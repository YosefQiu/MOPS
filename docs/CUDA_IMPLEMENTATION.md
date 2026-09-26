# CUDA Implementation - GPU-Accelerated Ocean Visualization and Particle Tracing

## Overview

This document describes the CUDA GPU implementation of MOPS visualization and particle tracing algorithms. The CUDA backend provides significant performance improvements over CPU implementations, particularly for:
- Large-scale particle trajectory calculations (10,000+ particles)
- High-resolution field visualization (4K+ images)
- Real-time interactive visualization

**Key Features**:
- **Unified memory management**: Centralized CUDA utilities in `CUDAMemoryUtils.cuh`
- **Unified constants**: All mesh/kernel constants in `CUDAConstants.cuh`
- **Optimized interpolation**: Wachspress coordinates for unstructured mesh
- **Robust boundary handling**: Edge projection and fallback strategies
- **Attribute tracking**: Temperature, salinity, and custom fields

**Performance**: 100× faster than CPU for 10,000 particles over 1000 timesteps.

## File Organization

```
src/GPU/CUDA/Kernel/
├── CUDAConstants.cuh              # Unified constant definitions (NEW)
├── CUDAMemoryUtils.cuh            # Unified memory management (NEW)
├── MPASOPathLineKernel.cu         # Time-varying particle tracing
├── MPASOStreamLineKernel.cu       # Steady-state particle tracing
├── MPASOVisualizerKernels.cu      # Field visualization kernels
├── MPASOSolutionKernels.cu        # Solution attribute computation
└── VisualizerKernels_Common.cuh   # Shared device functions
```

---

## Unified Infrastructure (2026)

### CUDAConstants.cuh - Unified Constant Definitions

**Purpose**: Eliminate duplicate constant definitions across kernel files.

**Location**: `src/GPU/CUDA/Kernel/CUDAConstants.cuh`

**Namespace**: `MOPS::GPU::CUDABackend::Constants`

**Key Constants**:
```cpp
constexpr int MAX_VERTEX_NUM = 10;            // Max vertices per cell
constexpr int MAX_CELL_NEIGHBOR_NUM = 11;     // Max cell neighbors (10 + self)
constexpr int MAX_VERTICAL_LEVELS = 100;      // Max ocean vertical layers
constexpr int MAX_ZTOP_LAYERS = 101;          // MAX_VERTICAL_LEVELS + 1
constexpr int DEFAULT_BLOCK_SIZE_1D = 128;    // 1D kernel block size
constexpr int DEFAULT_BLOCK_SIZE_2D = 16;     // 2D kernel block size (16×16)
constexpr int MAX_GRID_DIM = 65535;           // CUDA grid dimension limit
```

**Usage**:
```cpp
#include "CUDAConstants.cuh"
using namespace MOPS::GPU::CUDABackend::Constants;

__device__ void myKernel() {
    double ztop[MAX_VERTICAL_LEVELS];  // Use unified constant
    // ...
}
```

**History**: Before 2026-09-25, constants were defined locally in each kernel file with inconsistent values (MAX_VERTEX_NUM ranged from 10 to 20). This caused subtle bugs and maintenance issues. Now unified in single header.

### CUDAMemoryUtils.cuh - Unified Memory Management

**Purpose**: Eliminate duplicate CUDA memory utility functions.

**Location**: `src/GPU/CUDA/Kernel/CUDAMemoryUtils.cuh`

**Namespace**: `MOPS::GPU::CUDABackend::MemoryUtils`

**Functions**:
```cpp
// Allocate device memory
template<typename T>
T* DeviceAlloc(size_t count);

// Copy host → device
template<typename T>
void CopyHostToDevice(T* dst_dev, const T* src_host, size_t count);

// Copy device → host
template<typename T>
void CopyDeviceToHost(T* dst_host, const T* src_dev, size_t count);

// Free device memory
template<typename T>
void FreeDev(T* dev_ptr);

// Allocate and copy in one step
template<typename T>
T* DeviceAllocAndCopy(const T* host_ptr, size_t count);

// Copy back and free in one step
template<typename T>
void CopyBackAndFree(T* host_ptr, T* dev_ptr, size_t count);
```

**Usage**:
```cpp
#include "CUDAMemoryUtils.cuh"
using namespace MOPS::GPU::CUDABackend::MemoryUtils;

// Allocate and copy
double* dev_data = DeviceAllocAndCopy(host_data, array_size);

// Use in kernel
myKernel<<<grid, block>>>(dev_data, array_size);

// Copy back and clean up
CopyBackAndFree(host_data, dev_data, array_size);
```

**History**: Before unification, `FreeDev`, `DeviceAllocAndCopy`, `CheckCuda` were duplicated in `MPASOVisualizerKernels.cu` and `MPASOSolutionKernels.cu` with slight variations. Now centralized with consistent error checking.

---

## CUDA Visualization and Particle Tracing Algorithm Analysis

---

## 1. CUDA VisualizeFixedLayer Algorithm

**Purpose**: Visualize ocean velocity at a specific vertical layer (constant depth layer index).

**Step-by-step process**:

1. **Initialize grid and pixel mapping**
   - Convert image pixel coordinates to geographic lat/lon using `convertPixelToLatLonToRadians()`
   - Convert lat/lon to 3D Cartesian coordinates using `convertRadianLatLonToXYZ()`
   - Use pre-computed `cell_id` array from KDTree search to identify which cell contains each pixel

2. **Validate cell location**
   - Check if `cell_id` is valid (>= 0 and < cell_size)
   - Use `IsInMesh()` to verify pixel position is inside the cell polygon
   - Uses Wachspress coordinate test for point-in-polygon

3. **Get cell geometry**
   - Retrieve number of vertices for the cell using `number_vertex_on_cell[cell]`
   - Get vertex indices using `GetCellVerticesIdx()` (reads from `vertices_on_cell` array)
   - Get vertex positions using `GetCellVertexPos()`

4. **Calculate Wachspress interpolation weights**
   - Call `CalcPolygonWachspress()` to compute barycentric-like weights for arbitrary polygon
   - Weights sum to 1.0 and are used for interpolation within the cell

5. **Interpolate velocity at fixed layer**
   - Access velocity data at specified layer: `cell_vertex_velocity[vid * vel_levels + fixed_layer]`
   - Weighted average using Wachspress weights: `velocity = Σ(weight[i] * velocity[vertex[i]])`
   - Result is velocity in 3D Cartesian coordinates

6. **Convert to East-North-Up coordinates**
   - Use `convertXYZVelocityToENU()` to transform Cartesian velocity to geographic frame
   - Returns (zonal_velocity, meridional_velocity, 0)

7. **Write to output image**
   - Store (u_east, v_north, 0) at pixel location
   - Invalid pixels get NaN values

---

## 2. CUDA VisualizeFixedDepth Algorithm

**Purpose**: Visualize ocean velocity at a specific depth (in meters), requiring vertical interpolation.

**Step-by-step process**:

1. **Initialize and validate inputs**
   - Same pixel-to-cell mapping as FixedLayer
   - Check if pixel is in valid ocean cell

2. **Get cell geometry and Wachspress weights**
   - Same as FixedLayer (steps 3-4)

3. **Compute ztop profile at current pixel position**
   - `ztop` represents depth of layer interfaces (negative values, decreasing with depth)
   - For each vertical level k:
     ```
     current_point_ztop[k] = Σ(weight[v] * cell_vertex_ztop[vertex[v] * levels + k])
     ```
   - Interpolates ztop values horizontally using Wachspress weights

4. **Enforce monotonicity of ztop profile**
   - Ensure ztop values decrease with depth: if `ztop[k] > ztop[k-1]`, set `ztop[k] = ztop[k-1] - 1e-9`

5. **Check if target depth is in valid range**
   - Surface: `z_surf = ztop[0]` (sea surface elevation)
   - Bottom: `z_bot = ztop[levels-1]` (seafloor depth)
   - If `fixed_depth` is outside [z_bot, z_surf], return NaN

6. **Find vertical layer containing the target depth**
   - Linear search through ztop array
   - Find layer k where `ztop[k-1] >= fixed_depth >= ztop[k]`
   - If depth <= surface level, use layer 0
   - Store result in `local_layer`

7. **Calculate vertical interpolation parameter**
   - `top_z = ztop[local_layer - 1]`
   - `bot_z = ztop[local_layer]`
   - `t = (fixed_depth - bot_z) / (top_z - bot_z)` (0 at bottom, 1 at top)

8. **Interpolate velocity vertically**
   - Get velocity at layer above: `v_top = CalcVelocity(..., layer=local_layer-1)`
   - Get velocity at layer below: `v_bot = CalcVelocity(..., layer=local_layer)`
   - Handle zero-velocity cases (if one is zero, use the other)
   - Interpolate: `final_vel = (1-t) * v_bot + t * v_top`

9. **Convert to ENU and compute speed**
   - Transform to East-North-Up coordinates
   - Compute horizontal speed: `spd = sqrt(u_east^2 + v_north^2)`
   - Store `(u_east, v_north, spd)`

10. **Optional: Interpolate additional attributes**
    - If attributes exist (temperature, salinity, etc.), interpolate them similarly
    - Store in second image buffer

---

## 3. CUDA VisualizeFixedLatitude Algorithm

**Purpose**: Visualize vertical cross-section at a fixed latitude (latitude-depth plane).

**Step-by-step process**:

1. **Setup vertical-longitudinal grid**
   - Rows represent depth levels (from surface to max depth)
   - Columns represent longitude values
   - Fixed latitude for entire cross-section

2. **For each grid point (depth, longitude)**:

3. **Convert to 3D position**
   - Lat/lon to radians: `latlon_r = (fixed_lat * π/180, lon * π/180)`
   - Convert to Cartesian: `convertRadianLatLonToXYZ(latlon_r, position)`

4. **Find containing cell**
   - Use KDTree search: `searchKDT(position, cell_id)`
   - Verify with `IsInMesh()`

5. **Get cell geometry and weights**
   - Same as previous algorithms

6. **Interpolate ztop profile**
   - Same horizontal interpolation as VisualizeFixedDepth

7. **Check if depth is in valid ocean range**
   - If `depth > ztop[0]` or `depth < ztop[n_vert-1]`, pixel is outside ocean → NaN

8. **Find vertical layer**
   - Binary-like search for layer where `ztop[layer-1] >= depth >= ztop[layer]`

9. **Vertical velocity interpolation**
   - `t = (depth - ztop[layer-1]) / (ztop[layer] - ztop[layer-1])`
   - `vel_up = CalcVelocity(..., layer-1)`
   - `vel_dn = CalcVelocity(..., layer)`
   - `final_vel = (1-t) * vel_up + t * vel_dn`

10. **Convert and store**
    - Transform to ENU coordinates
    - Store at pixel (row=depth_index, col=lon_index)

---

## 4. CUDA StreamLine Algorithm

**Purpose**: Trace particle trajectories in steady-state velocity field (single time snapshot).

**Step-by-step process**:

### **Initialization (per particle)**

1. **Setup particle state**
   - Position: `sample_points[particle_id]` (3D Cartesian)
   - Depth: `particle_depths[particle_id]` (positive meters below surface)
   - Cell: `default_cell_id[particle_id]` (initial containing cell)
   - Cell neighbors: `GetCellNeighborsIdx()` (for fast cell location)

### **Main time-stepping loop (for each timestep)**

2. **Update current position and cell**
   - If first step: use initial cell_id
   - Otherwise: find nearest cell among current cell and neighbors
     ```
     min_distance = infinity
     for each neighbor cell:
         distance = ||cell_center - particle_position||
         if distance < min_distance:
             cell_id = this_cell
     ```
   - Update neighbor list for new cell

3. **Calculate velocity at current position**
   
   **3a. Get cell geometry**
   - Number of vertices: `number_vertex_on_cell[cell_id]`
   - Vertex indices: `GetCellVerticesIdx()`
   - Vertex positions: `GetCellVertexPos()`
   
   **3b. Check if particle is in mesh**
   - Use `IsInMesh()` with Wachspress coordinate test
   - If outside mesh, terminate (or handle boundary)
   
   **3c. Calculate Wachspress weights**
   - `CalcPolygonWachspress(position, vertex_pos, weights, n_vertices)`
   
   **3d. Interpolate ztop profile at particle position**
   - For each level k: `ztop[k] = Σ(weight[v] * ztop_vertex[v][k])`
   - Enforce monotonicity: `if ztop[k] > ztop[k-1]: ztop[k] = ztop[k-1] - 1e-9`
   
   **3e. Find vertical layer containing particle**
   - Binary search: find layer where `ztop[layer-1] >= current_depth >= ztop[layer]`
   - Handle boundaries:
     - Above surface: `local_layer = 1`
     - Below bottom: `local_layer = ztop_layers - 1`
   
   **3f. Vertical interpolation parameter**
   - `ztop_up = ztop[local_layer - 1]`
   - `ztop_dn = ztop[local_layer]`
   - `t = (current_depth - ztop_dn) / (ztop_up - ztop_dn)`
   - Clamp depth to valid range
   
   **3g. Interpolate horizontal velocity**
   - `vel_up = CalcVelocity(..., layer=local_layer-1)` (Wachspress interpolation horizontally)
   - `vel_dn = CalcVelocity(..., layer=local_layer)`
   - Handle zero-velocity cases
   - `final_vel = t * vel_up + (1-t) * vel_dn`
   
   **3h. Interpolate vertical velocity**
   - Same process using `cell_vertex_vert_velocity`
   - `w_up = CalcAttribute(..., layer=local_layer-1)`
   - `w_dn = CalcAttribute(..., layer=local_layer)`
   - `vertical_vel = t * w_up + (1-t) * w_dn`

4. **Time integration (choose method)**
   
   **Option A: Euler method** (`use_euler = true`)
   - Calculate rotation axis: `axis = cross(position, velocity)` (normalized)
   - Angular displacement: `theta = (speed * dt) / radius`
   - Rotate position: `CalcPositionAfterRotation(pos, axis, theta)`
   - Uses Rodrigues' rotation formula
   
   **Option B: RK4 method** (`use_euler = false`)
   - **Stage 1**: `s1 = CalcVelocityAt(pos, depth)`
   - **Stage 2**: `p2 = AdvectOnSphere(pos, s1.vel, dt*0.5)`, `s2 = CalcVelocityAt(p2, depth)`
   - **Stage 3**: `p3 = AdvectOnSphere(pos, s2.vel, dt*0.5)`, `s3 = CalcVelocityAt(p3, depth)`
   - **Stage 4**: `p4 = AdvectOnSphere(pos, s3.vel, dt)`, `s4 = CalcVelocityAt(p4, depth)`
   - Average: `vel_avg = (s1.vel + 2*s2.vel + 2*s3.vel + s4.vel) / 6`
   - Position: `new_pos = pos + vel_avg * dt`, normalize to sphere
   - Vertical: `vert_vel_avg = (s1.v_vel + 2*s2.v_vel + 2*s3.v_vel + s4.v_vel) / 6`

5. **Update depth**
   - `new_depth = old_depth - vertical_vel * dt`
   - Clamp to surface: `new_depth = max(0, new_depth)`
   - Update radius: `r_new = r + vertical_vel * dt`

6. **Project to sphere**
   - Normalize new position: `new_pos = (new_pos / ||new_pos||) * r_new`

7. **Record trajectory point**
   - If `(run_time % record_t) == 0`: store position and velocity
   - Increment output buffer index

8. **Update particle state**
   - `sample_points[particle_id] = new_position`
   - `particle_depths[particle_id] = new_depth`
   - Increment timestep counter

### **Finalization**

9. **Copy results from GPU to CPU**
   - Transfer `write_points` and `write_vels` arrays

10. **Filter invalid points**
    - Remove NaN/zero entries using `FinalizeTrajectoryLines()`

---

## 5. CUDA PathLine Algorithm

**Purpose**: Trace particle trajectories through time-varying velocity field (interpolates between two time snapshots).

**Step-by-step differences from StreamLine**:

### **Key additions beyond StreamLine**:

1. **Temporal interpolation**
   - Two velocity fields: `velocity_front` (time t0) and `velocity_back` (time t1)
   - Interpolation parameter: `alpha = current_step / total_steps` (0 to 1)
   - `velocity(t) = (1-alpha) * velocity_front + alpha * velocity_back`

2. **Enhanced velocity calculation** (`CalcVelocityAtPathlineCUDA`)
   
   **2a. Find vertical layer in BOTH time snapshots**
   - `ztop_front` and `ztop_back` may differ (evolving ocean surface/bathymetry)
   - Find `local_layer_front` using `ztop_front` profile
   - Find `local_layer_back` using `ztop_back` profile
   - **maxLevelCell clamping** (CRITICAL for seafloor avoidance):
     ```cpp
     if (max_level_cell != nullptr) {
         int max_valid_layer = max_level_cell[cell_id];
         if (max_valid_layer > 0 && max_valid_layer <= actual_ztop_layer) {
             // Clamp layer indices to stay above seafloor
             if (local_layer_front >= max_valid_layer) {
                 local_layer_front = max_valid_layer - 1;
             }
             if (local_layer_back >= max_valid_layer) {
                 local_layer_back = max_valid_layer - 1;
             }
         }
     }
     ```
   - **Purpose**: `maxLevelCell[cellID]` is the deepest valid ocean layer for each cell
   - **Effect**: Prevents particles from accessing velocity/attribute data below seafloor
   - **Example**: If `maxLevelCell[68814] = 51` and particle layer = 55, clamp to layer 50
   
   **2b. Interpolate velocity in space and time**
   - Front time: `vel_front = (1-t_front) * vel_dn_front + t_front * vel_up_front`
   - Back time: `vel_back = (1-t_back) * vel_dn_back + t_back * vel_up_back`
   - Combined: `final_vel = (1-alpha) * vel_front + alpha * vel_back`
   
   **2c. Same for vertical velocity and attributes**

3. **Vertical boundary conditions** (critical for realistic behavior)
   
   **3a. Surface boundary projection**
   - Compute surface depth at NEW horizontal position: `surface_ztop = CalcZTopAtLevel0(new_pos_horizontal)`
   - If `new_depth < surface_depth`: clamp `new_depth = surface_depth`
   
   **3b. Bottom-following behavior**
   - Compute bottom clearance at time t in CURRENT cell: `d = bottom_depth_t - old_depth`
   - Compute bottom depth at NEW position in TARGET cell: `bottom_depth_next`
   - Preserve clearance: if `new_depth > bottom_depth_next`: `new_depth = bottom_depth_next - d`
   - Prevents particles from penetrating seafloor
   
   **3c. Final clamping**
   - `new_depth = clamp(new_depth, surface_depth, bottom_depth_next)`

4. **Lateral boundary handling** (land/coastline)
   
   **4a. Detect boundary crossing**
   - Check if new position is in current cell OR neighbor cells using `FindContainingCellInCurrentOrNeighborsCUDA()`
   - If in neighbor: update cell_id (normal crossing)
   - If outside all cells: boundary hit detected
   
   **4b. Boundary projection strategy**
   - Find nearest cell edge using `FindNearestCellEdgeCUDA()`
   - Project velocity onto edge tangent: `ProjectVelocityOntoEdgeCUDA()` (flow parallel to coast)
   - Move particle along edge: `CalcPositionAfterRotation(pos, axis, theta)`
   - Verify projected position is still in mesh
   
   **4c. Fallback for failed projection**
   - If projection fails: jump to nearest cell center (current or neighbor)
   - Uses `FindContainingCellInCurrentOrNeighborsCUDA()` for nearest valid cell
   - Last resort: random tangent movement (`GenerateRandomTangentVelocityCUDA`)

5. **NaN/Inf velocity handling** (land cell detection)
   - If velocity components are NaN or Inf → particle entered land/boundary cell
   - Emergency relocation: jump to nearest valid ocean cell center
   - Recalculate velocity at new position (replaces invalid velocity)
   - Prevents stuck particles and maintains "NEVER STOP" policy

6. **Attribute tracking**
   - Track additional scalar fields (temperature, salinity, etc.)
   - Interpolate in space, depth, and time
   - Store alongside trajectory

7. **Boundary statistics**
   - Count boundary hits per particle using atomic operations
   - Report summary: particles affected, total hits, max hits per particle

---

## Key Helper Functions

### **IsInMesh()**
- Point-in-polygon test using Wachspress coordinates
- Returns true if sum of weights ≈ 1.0 and all weights >= 0

### **CalcPolygonWachspress()**
- Computes generalized barycentric coordinates for arbitrary convex polygons
- Weighted interpolation preserves linear functions

### **CalcVelocity()**
- Horizontal interpolation: `vel = Σ(weight[i] * velocity[vertex[i]][layer])`
- Returns 3D Cartesian velocity vector

### **CalcAttribute()**
- Same as CalcVelocity but for scalar fields (temperature, salinity, etc.)

### **AdvectOnSphereCUDA()**
- Spherical rotation using Rodrigues' formula
- `axis = normalize(cross(pos, vel))`
- `theta = (speed * dt) / radius`
- Exact geodesic motion on sphere

### **CalcRotationAxis()**
- `axis = cross(position, velocity)`
- Perpendicular to both position (radial) and velocity (tangent)

### **CalcPositionAfterRotation()**
- Rodrigues' rotation formula:
  ```
  new_pos = pos * cos(theta) + axis * sin(theta) + axis * dot(axis, pos) * (1 - cos(theta))
  ```

---

## Summary

The CUDA implementation uses:
- **Wachspress coordinates** for horizontal interpolation within unstructured polygonal cells
- **Linear interpolation** for vertical dimension (between ztop layers)
- **Temporal interpolation** (PathLine only) between time snapshots
- **Spherical geometry** via Rodrigues' rotation for exact geodesic advection
- **Robust boundary handling** with edge projection, cell center relocation, and random perturbation
- **Parallel execution** with one thread per particle or pixel

All algorithms share the same core interpolation infrastructure but differ in:
- Fixed layer: no vertical search, direct layer access
- Fixed depth: vertical search at each pixel, single time snapshot
- Fixed latitude: vertical cross-section, different grid layout
- StreamLine: particle advection, steady velocity field
- PathLine: particle advection, time-varying field + boundary conditions + attributes

---

## Configuration Parameters

### TrajectorySettings - PathLine Configuration

**Location**: `src/Core/MPASOVisualizer.h`

**Key Parameters**:

```cpp
struct TrajectorySettings {
    // Velocity perturbation (for low-velocity regions)
    double perturbation_sigma_h = 0.01;       // Horizontal perturbation (m/s)
    double perturbation_sigma_v = 0.00001;    // Vertical perturbation (m/s)
    
    // Data masking threshold
    double masked_attr_value = -1e34;         // Masked attribute value threshold
    
    // Integration parameters
    double dt = 600.0;                        // Timestep (seconds)
    int total_timesteps = 1000;               // Total integration steps
    IntegrationMethod method = RK4;           // Euler or RK4
};
```

### Velocity Perturbation

**Purpose**: Add small random noise to prevent particles from getting stuck in near-zero velocity regions.

**When Applied**: If `||velocity|| < threshold` (typically 1e-6 m/s)

**Algorithm**:
```cpp
if (velocity_magnitude < 1e-6) {
    // Add horizontal perturbation
    velocity.x += sigma_h * random_normal();
    velocity.y += sigma_h * random_normal();
    
    // Add vertical perturbation
    vertical_velocity += sigma_v * random_normal();
}
```

**Default Values**:
- `perturbation_sigma_h = 0.01` m/s (horizontal)
- `perturbation_sigma_v = 0.00001` m/s (vertical)

**Tuning**:
- **Too small**: Particles still get stuck
- **Too large**: Artificial diffusion dominates real advection
- **Typical range**: 0.001 to 0.1 m/s for horizontal

**Python API**:
```python
import pyMOPS

config = pyMOPS.TrajectorySettings()
config.perturbation_sigma_h = 0.01      # 1 cm/s horizontal noise
config.perturbation_sigma_v = 0.00001   # 0.01 mm/s vertical noise

pl = pyMOPS.PathLine(grid, solution_t0, solution_t1, config)
pl.run(seed_points, output_file)
```

### Masked Attribute Value

**Purpose**: Identify invalid/masked data in temperature, salinity, and other attributes.

**Detection**: Values below `masked_attr_value` are considered invalid.

**Behavior in Kernels**:
```cpp
double temperature = interpolate_temperature(position, layer);
if (temperature < masked_attr_value) {
    // Invalid data - use fallback strategy:
    // 1. Try neighboring cells
    // 2. Use last valid value
    // 3. Set to NaN if no valid data available
}
```

**Default Value**: `-1e34` (matches MPAS netCDF `_FillValue` convention)

**Common Values**:
- MPAS-Ocean: `-1e34`
- Some models: `-999.0` or `-9999.0`
- Check netCDF file: `ncdump -h ocean.nc | grep _FillValue`

**Python API**:
```python
config.masked_attr_value = -1e34  # MPAS standard
# or
config.masked_attr_value = -999.0  # Alternative convention
```

### maxLevelCell - Seafloor Layer Index

**Purpose**: Prevent particles from accessing invalid layers below the seafloor.

**Data Source**: MPAS netCDF variable `maxLevelCell` (integer array, size `nCells`)

**Meaning**: For cell `i`, `maxLevelCell[i]` is the deepest valid layer index (1-indexed in MPAS).

**Kernel Usage**:
```cpp
int max_valid_layer = max_level_cell[cell_id];
if (local_layer >= max_valid_layer) {
    // Particle below seafloor - clamp to bottom layer
    local_layer = max_valid_layer - 1;
}
```

**Effects**:
- **Without clamping**: Particles access invalid data below seafloor → NaN velocities → crashes
- **With clamping**: Particles stay in valid ocean layers → realistic bottom-following behavior

**Loading** (C++):
```cpp
// Read maxLevelCell from netCDF
std::vector<int> maxLevelCell_vec;
reader.readVariable("maxLevelCell", maxLevelCell_vec);

// Pass to PathLine kernel
KernelPathLine<<<grid, block>>>(
    /* ... other parameters ... */,
    maxLevelCell_vec.data(),
    /* ... */
);
```

**Python API** (automatic if available in netCDF):
```python
# pyMOPS automatically loads maxLevelCell if present
pl = pyMOPS.PathLine(grid, solution_t0, solution_t1)
# No manual configuration needed
```

---

## Performance Optimization

### Memory Management

**Unified Utilities** (`CUDAMemoryUtils.cuh`):
- **Before**: Each kernel file had 3-4 copies of `cudaMalloc`/`cudaMemcpy` wrappers
- **After**: Single implementation with consistent error checking
- **Benefit**: ~20% reduction in code duplication, easier debugging

**Pattern**:
```cpp
// Allocate and copy multiple arrays
double* dev_velocity = DeviceAllocAndCopy(host_velocity, vel_size);
double* dev_ztop = DeviceAllocAndCopy(host_ztop, ztop_size);
int* dev_maxLevel = DeviceAllocAndCopy(host_maxLevel, cell_size);

// Launch kernel
PathLineKernel<<<grid, block>>>(dev_velocity, dev_ztop, dev_maxLevel, ...);

// Copy back and free
CopyBackAndFree(host_positions, dev_positions, particle_size);
FreeDev(dev_velocity);
FreeDev(dev_ztop);
FreeDev(dev_maxLevel);
```

### Constant Memory

**Unified Constants** (`CUDAConstants.cuh`):
- **Before**: `MAX_VERTEX_NUM` defined 20+ times with values 10, 20 (inconsistent)
- **After**: Single definition `MAX_VERTEX_NUM = 10` used everywhere
- **Benefit**: Compiler can optimize array allocations, fewer register spills

**Stack Arrays**:
```cpp
__device__ void PathLineKernel() {
    double ztop[MAX_VERTICAL_LEVELS];        // 100 doubles = 800 bytes
    vec3 vertex_pos[MAX_VERTEX_NUM];         // 10 vec3 = 240 bytes
    double weights[MAX_VERTEX_NUM];          // 10 doubles = 80 bytes
    // Total stack usage: ~1.2 KB per thread
}
```

**Tuning**: Reducing `MAX_VERTEX_NUM` from 20 → 10 saves 80 bytes/thread → 40 KB per 512-thread block.

### Block Size Selection

**1D Kernels** (particles, image pixels):
```cpp
dim3 block(DEFAULT_BLOCK_SIZE_1D);  // 128 threads
dim3 grid((particle_count + 127) / 128);
```

**2D Kernels** (image rendering):
```cpp
dim3 block(DEFAULT_BLOCK_SIZE_2D, DEFAULT_BLOCK_SIZE_2D);  // 16×16 = 256 threads
dim3 grid((width + 15) / 16, (height + 15) / 16);
```

**Tuning**:
- **Too small** (32 threads): Low occupancy, underutilizes SM
- **Too large** (1024 threads): High register pressure, stack overflow risk
- **Optimal**: 128-256 threads per block for this workload

### Kernel Launch Overhead

**Batch Operations**: Group particles into chunks to reduce launch overhead.

**Example**:
```cpp
// Bad: Launch kernel per particle (high overhead)
for (int i = 0; i < num_particles; ++i) {
    PathLineKernel<<<1, 1>>>(particle[i], ...);  // 10,000 launches!
}

// Good: Launch once for all particles
PathLineKernel<<<grid, block>>>(all_particles, num_particles, ...);  // 1 launch
```

**Speedup**: 100× faster for 10,000 particles.

---

## Debugging and Validation

### Printf Debugging

**Enabled in Kernels**:
```cpp
__device__ void CalcVelocityAtPathlineCUDA(...) {
    if (particle_id == 0 && timestep % 100 == 0) {
        printf("[PathLine] Particle 0: cell=%d, depth=%.2f, layer=%d, vel=(%.6f,%.6f,%.6f)\n",
               cell_id, current_depth, local_layer, velocity.x, velocity.y, velocity.z);
    }
}
```

**Output**: Appears in CPU stdout (buffered, may delay until kernel completion).

**Use Cases**:
- Track particle 0 trajectory
- Verify maxLevelCell clamping
- Debug NaN propagation

### maxLevelCell Validation

**Check Loading**:
```cpp
printf("[PathLine] Particle 0: cell_id=%d, max_level_cell[%d]=%d, actual_ztop_layer=%d\n",
       cell_id, cell_id, max_level_cell[cell_id], actual_ztop_layer);
```

**Expected Output**:
```
[PathLine] Particle 0: cell_id=68814, max_level_cell[68814]=51, actual_ztop_layer=60
```

**Interpretation**:
- `max_level_cell = 51`: Seafloor is at layer 51 (layers 52-60 are below seafloor, invalid)
- `actual_ztop_layer = 60`: Full vertical grid has 60 layers
- **Clamping needed**: If particle reaches layer 55, must clamp to layer 50

### Attribute Masking Validation

**Check for Invalid Values**:
```cpp
double temp = interpolate_temperature(pos, layer);
if (temp < masked_attr_value) {
    printf("[WARNING] Particle %d: Invalid temperature %.2e at cell %d layer %d\n",
           particle_id, temp, cell_id, layer);
}
```

**Common Causes**:
- Particle entered land cell (velocity pushed it ashore)
- Layer below seafloor (maxLevelCell not clamped)
- Corrupted netCDF data

---

## Related Documentation

- **[MPAS_SOLUTION.md](MPAS_SOLUTION.md)** - Data structures and velocity computation
- **[MPAS_FIELD.md](MPAS_FIELD.md)** - Field binding and spatial queries
- **[MPAS_GRID.md](MPAS_GRID.md)** - Mesh topology and KD-tree indexing
- **[pathline_cuda_summary.md](pathline_cuda_summary.md)** - Detailed PathLine implementation notes
- **[CUDA_CODE_AUDIT_REPORT.md](../CUDA_CODE_AUDIT_REPORT.md)** - Code cleanup history

---

## References

- **MPAS-Ocean Documentation**: https://mpas-dev.github.io/ocean/ocean.html
- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **Wachspress Coordinates**: Meyer et al., "Generalized Barycentric Coordinates on Irregular Polygons", 2002
- **Source Files**: 
  - `/pscratch/sd/q/qiuyf/MOPS/src/GPU/CUDA/Kernel/MPASOPathLineKernel.cu`
  - `/pscratch/sd/q/qiuyf/MOPS/src/GPU/CUDA/Kernel/CUDAConstants.cuh`
  - `/pscratch/sd/q/qiuyf/MOPS/src/GPU/CUDA/Kernel/CUDAMemoryUtils.cuh`