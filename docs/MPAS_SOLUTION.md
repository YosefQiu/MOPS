# MPASOSolution - Time-Varying Ocean State

## Overview

`MPASOSolution` manages time-varying ocean state fields on the MPAS-Ocean unstructured mesh. It stores velocity components, layer thickness, vertical coordinate information (zTop), and extensible scalar attributes like temperature and salinity. Each solution object represents a single time snapshot of the ocean state.

The class serves three primary purposes:
1. **Storage**: Maintains all physical fields at a specific time (velocity, thickness, attributes)
2. **Data Transformation**: Converts between different velocity representations (edge-normal → cell-centered → vertex-centered)
3. **Attribute Management**: Provides extensible framework for arbitrary scalar/vector fields

This class is essential for visualization and particle trajectory calculations, as it provides the velocity and attribute fields needed for spatiotemporal interpolation.

## Key Concepts

### Time Snapshot

Each `MPASOSolution` object represents the ocean state at a single moment in time:
- **Timestamp**: String identifier (e.g., "2020-01-15_00:00:00")
- **Timestep**: Integer index in the simulation sequence
- **SolutionID**: Unique identifier combining timestamp and timestep

Multiple solutions can be paired (front/back) in `MPASOField` for temporal interpolation in time-varying particle trajectories.

### Velocity Representations

MPAS-Ocean stores velocity in **edge-normal** form (perpendicular to cell edges). MOPS converts this to more useful representations:

1. **Edge-Normal Velocity** (`cellNormalVelocity_vec`)
   - Original MPAS format: velocity perpendicular to each edge
   - Storage: `[mEdgesSize × mVertLevels]`
   - Direct from netCDF variable `normalVelocity`

2. **Zonal/Meridional Velocity** (`cellZonalVelocity_vec`, `cellMeridionalVelocity_vec`)
   - Geographic components: east-west (zonal), north-south (meridional)
   - Cell-centered: `[mCellsSize × mVertLevels]`
   - Vertex-centered: `[mVertexSize × mVertLevels]` (derived)
   - Computed from edge-normal via `calcCellCenterVelocityByZM()`

3. **3D Cartesian Velocity** (`cellCenterVelocity_vec`, `cellVertexVelocity_vec`)
   - Tangent to sphere surface in 3D (x,y,z) coordinates
   - Cell-centered: `[mCellsSize × mVertLevels]`
   - Vertex-centered: `[mVertexSize × mVertLevels]`
   - Computed via `calcCellCenterVelocity()` or `calcCellVertexVelocity()`
   - **Used by particle tracing kernels** for RK4 integration

4. **Vertical Velocity** (`cellVertVelocity_vec`, `cellVertexVertVelocity_vec`)
   - Velocity in z-direction (positive upward)
   - Interface values: `[mCellsSize × mVertLevelsP1]` or `[mVertexSize × mVertLevelsP1]`
   - Computed from layer thickness changes via `calcCellVertexVertVelocity()`

### Vertical Coordinate System (zTop)

The ocean's vertical structure is defined by **zTop** - the depth of layer interfaces:

```
Surface (SSH)  ━━━━━━━━━━━━━━━  zTop[0] = SSH (e.g., +0.5 m)
Layer 1        ┊┊┊┊┊┊┊┊┊┊┊┊┊┊
Interface      ─────────────────  zTop[1] (e.g., -10 m)
Layer 2        ┊┊┊┊┊┊┊┊┊┊┊┊┊┊
Interface      ─────────────────  zTop[2] (e.g., -30 m)
...
Interface      ─────────────────  zTop[nVertLevels-1]
Layer N        ┊┊┊┊┊┊┊┊┊┊┊┊┊┊
Bottom         ━━━━━━━━━━━━━━━  zTop[nVertLevels] = seafloor depth
```

**Key Features**:
- **Variable in space and time**: Seafloor depth varies by location; sea surface height varies with tides/currents
- **Negative convention**: Depths are negative (e.g., -100 m = 100 meters below surface)
- **Cell-centered** (`cellZTop_vec`): Interpolated from layer thickness
- **Vertex-centered** (`cellVertexZTop_vec`): Used for particle tracing interpolation
- **Layer count**: `mVertLevels` layers → `mVertLevelsP1 = mVertLevels + 1` interfaces

### Extensible Attributes

Beyond velocity and zTop, solutions can store arbitrary scalar/vector fields:

**Cell-Centered Attributes**:
- Temperature, salinity, density, vorticity, etc.
- Stored in type-specific maps: `mDoubleAttributes`, `mFloatAttributes`, `mCharAttributes`, `mVec3Attributes`
- Example: `mDoubleAttributes["temperature"]` → `[mCellsSize × mVertLevels]`

**Vertex-Centered Attributes** (derived):
- Interpolated from cell centers for particle tracing
- Maps: `mDoubleAttributes_CtoV`, `mFloatAttributes_CtoV`
- Suffix `_CtoV` = "Cell to Vertex"

**Invalid Value Handling**:
- MPAS uses large negative values (< -1e20) to mask invalid data (e.g., land cells)
- Converted to standardized `INVALID_VALUE = -999.0` during loading
- Cleaned via `cleanInvalidCellCenterValues()`

### maxLevelCell - Seafloor Layer Index

**Critical for trajectory calculations**:
- `maxLevelCell[cellID]`: Deepest valid layer index for each cell
- Values: 1 to `nVertLevels` (or 0 for land cells)
- **Purpose**: Prevents particles from accessing layers below the seafloor
- **Usage in kernels**: Clamp vertical layer index before accessing velocity/attribute arrays
- Stored in cell attribute: `cellBottomDepth_vec` or via `mDoubleAttributes["maxLevelCell"]`

## Data Structures

### Solution Identification
```cpp
std::string mTimeStamp;         // Time identifier (e.g., "2020-01-15_00:00:00")
std::string mDataName;          // Dataset name
SolutionID mID;                 // Unique identifier (timestamp + timestep)
```

**SolutionID Structure**:
```cpp
struct SolutionID {
    std::string timeStamp;
    int timestep;
};
```

### Dimensions
```cpp
int mCellsSize;        // Number of cells (same as MPASOGrid)
int mEdgesSize;        // Number of edges
int mMaxEdgesSize;     // Maximum edges per cell (typically 7)
int mVertexSize;       // Number of vertices
int mTimesteps;        // Total timesteps in dataset
int mVertLevels;       // Number of vertical layers (60-100 typically)
int mVertLevelsP1;     // mVertLevels + 1 (layer interfaces)
```

**zTop-Specific Dimensions**:
```cpp
int mTotalZTopLayer;    // Actual number of zTop layers (= mVertLevels)
int mTotalZTopLayerP1;  // mTotalZTopLayer + 1
```

### Cell-Centered Velocity Fields

#### Edge-Normal Velocity (Native MPAS Format)
```cpp
std::vector<double> cellNormalVelocity_vec;  // [mEdgesSize × mVertLevels]
                                              // Velocity perpendicular to edges
```

#### Zonal/Meridional Velocity
```cpp
std::vector<double> cellZonalVelocity_vec;       // [mCellsSize × mVertLevels]
                                                  // East-west component (cell-centered)
                                                  
std::vector<double> cellMeridionalVelocity_vec;  // [mCellsSize × mVertLevels]
                                                  // North-south component (cell-centered)
```

#### 3D Cartesian Velocity
```cpp
std::vector<vec3> cellCenterVelocity_vec;   // [mCellsSize × mVertLevels]
                                             // Velocity in 3D Cartesian (x,y,z)
                                             // Used by particle tracing
```

**Deprecated**:
```cpp
std::vector<vec3> cellVelocity_vec;  // Legacy, not used
```

#### Vertical Velocity
```cpp
std::vector<double> cellVertVelocity_vec;   // [mCellsSize × mVertLevelsP1]
                                             // Vertical velocity at layer interfaces
```

### Vertex-Centered Velocity Fields (Derived)

Computed from cell-centered values for interpolation during particle tracing:

```cpp
std::vector<vec3> cellVertexVelocity_vec;            // [mVertexSize × mVertLevels]
                                                      // 3D Cartesian velocity at vertices
                                                      
std::vector<double> cellVertexZonalVelocity_vec;     // [mVertexSize × mVertLevels]
std::vector<double> cellVertexMeridionalVelocity_vec;// [mVertexSize × mVertLevels]
std::vector<double> cellVertexVertVelocity_vec;      // [mVertexSize × mVertLevelsP1]
```

### Vertical Coordinate (zTop)

```cpp
std::vector<double> cellZTop_vec;         // [mCellsSize × mTotalZTopLayerP1]
                                           // Layer interface depths (cell-centered)
                                           
std::vector<double> cellVertexZTop_vec;   // [mVertexSize × mTotalZTopLayerP1]
                                           // Layer interface depths (vertex-centered)
                                           // Used in particle tracing kernels
```

**Indexing**: Cell `c` layer interface `k` depth = `cellZTop_vec[c * mTotalZTopLayerP1 + k]`

### Layer Thickness and Topography

```cpp
std::vector<double> cellLayerThickness_vec;  // [mCellsSize × mVertLevels]
                                              // Thickness of each layer (meters)
                                              
std::vector<double> cellBottomDepth_vec;     // [mCellsSize]
                                              // Depth of seafloor (negative value)
                                              
std::vector<double> cellSurfaceHeight_vec;   // [mCellsSize]
                                              // Sea surface height (SSH, can be positive/negative)
```

### Extensible Attributes

#### Cell-Centered Attributes
```cpp
std::map<std::string, std::vector<char>>   mCharAttributes;
std::map<std::string, std::vector<float>>  mFloatAttributes;
std::map<std::string, std::vector<double>> mDoubleAttributes;
std::map<std::string, std::vector<vec3>>   mVec3Attributes;
```

**Examples**:
- `mDoubleAttributes["temperature"]` → `[mCellsSize × mVertLevels]`
- `mDoubleAttributes["salinity"]` → `[mCellsSize × mVertLevels]`
- `mFloatAttributes["density"]` → `[mCellsSize × mVertLevels]`

#### Vertex-Centered Attributes (Derived)
```cpp
std::map<std::string, std::vector<double>> mDoubleAttributes_CtoV;
std::map<std::string, std::vector<float>>  mFloatAttributes_CtoV;
```

**Usage**: Interpolated to vertices for particle tracing attribute tracking.

## Methods

### Initialization

#### `void initSolution(MPASOReader* reader)`
Load solution data from MPAS netCDF file via `MPASOReader`.

**Parameters**:
- `reader`: Pointer to initialized `MPASOReader` with open MPAS file at specific timestep

**Behavior**:
- Reads dimensions (`nCells`, `nVertices`, `nVertLevels`, etc.)
- Loads edge-normal velocity (`normalVelocity`)
- Loads layer thickness (`layerThickness`)
- Loads optional attributes (temperature, salinity, SSH, etc.)
- Sets `mTimeStamp` from netCDF `xtime` variable
- Converts masked values (< -1e20) to `INVALID_VALUE`

**Example**:
```cpp
MPASOReader reader("data/ocean.nc");
reader.setTimestep(0);  // Load first timestep
MPASOSolution solution;
solution.initSolution(&reader);
```

#### `void initSolution(ftk::ndarray_group* g, MPASOReader* reader = nullptr)`
Load solution from FTK `ndarray_group` (for preprocessed data pipelines).

**Parameters**:
- `g`: Pointer to FTK ndarray group with MPAS variables
- `reader`: Optional reader for metadata

#### `void initSolution_FromBin(const char* prefix)`
Load solution from cached binary files (faster than netCDF).

**Parameters**:
- `prefix`: Path prefix for binary files (e.g., `"/cache/ocean_t0"`)

**Expected Files**:
- `{prefix}_cellNormalVelocity.bin`
- `{prefix}_cellLayerThickness.bin`
- `{prefix}_cellZTop.bin`
- etc.

#### `void initSolution_DemoLoading(const char* yaml_path, int timestep)`
Load solution from YAML configuration (for testing).

**Parameters**:
- `yaml_path`: Path to YAML config file
- `timestep`: Which timestep to load

### Velocity Computation

#### `void calcCellCenterVelocity(MPASOGrid* grid, std::string& dataDir, const RuntimeContext& ctx)`
Compute 3D Cartesian velocity at cell centers from edge-normal velocity.

**Parameters**:
- `grid`: Pointer to `MPASOGrid` with mesh topology
- `dataDir`: Directory for caching intermediate results
- `ctx`: Runtime context (CPU/GPU selection)

**Output**: Populates `cellCenterVelocity_vec`

**Algorithm**:
1. For each cell, iterate over edges
2. Get edge-normal velocity and edge tangent vector
3. Reconstruct velocity: `v_cell = Σ(edge_normal_vel[e] * edge_tangent[e])`
4. Convert to 3D Cartesian coordinates

**Example**:
```cpp
solution.calcCellCenterVelocity(&grid, cacheDir, ctx);
// Now cellCenterVelocity_vec[cell*mVertLevels + layer] contains vec3 velocity
```

#### `void calcCellCenterVelocityByZM(MPASOGrid* grid, std::string& dataDir, const RuntimeContext& ctx)`
Compute cell-centered velocity from zonal/meridional components.

**Parameters**: Same as `calcCellCenterVelocity()`

**Output**: Populates `cellCenterVelocity_vec` from `cellZonalVelocity_vec` and `cellMeridionalVelocity_vec`

**Use Case**: When MPAS file directly provides zonal/meridional velocity (some output configurations)

#### `void calcCellVertexVelocity(MPASOGrid* grid, std::string& dataDir, const RuntimeContext& ctx)`
Interpolate velocity from cell centers to vertices.

**Parameters**:
- `grid`: Contains connectivity (`cellsOnVertex`)
- `dataDir`: Cache directory
- `ctx`: Runtime context

**Output**: Populates `cellVertexVelocity_vec`

**Algorithm**:
1. For each vertex, find 3 adjacent cells
2. Average velocity: `v_vertex = (v_cell1 + v_cell2 + v_cell3) / 3`
3. Handle boundary vertices with fewer cells

**Why Needed**: Particle tracing uses Wachspress interpolation within cells, which requires velocity at cell vertices.

**Example**:
```cpp
solution.calcCellVertexVelocity(&grid, cacheDir, ctx);
// Now cellVertexVelocity_vec[vertex*mVertLevels + layer] contains interpolated velocity
```

#### `void calcCellVertexVelocityByZM(MPASOGrid* grid, std::string& dataDir, const RuntimeContext& ctx)`
Interpolate zonal/meridional velocity to vertices, then convert to Cartesian.

**Output**: Populates `cellVertexVelocity_vec`, `cellVertexZonalVelocity_vec`, `cellVertexMeridionalVelocity_vec`

#### `void calcCellVertexVertVelocity(MPASOGrid* grid, std::string& dataDir, const RuntimeContext& ctx)`
Compute vertical velocity at vertices.

**Output**: Populates `cellVertexVertVelocity_vec`

**Algorithm**: Interpolate `cellVertVelocity_vec` from cells to vertices using 3-cell average.

### Vertical Coordinate Computation

#### `void calcCellCenterZtop()`
Compute zTop (layer interface depths) at cell centers from layer thickness.

**No parameters required** (operates on `cellLayerThickness_vec`)

**Output**: Populates `cellZTop_vec`

**Algorithm**:
```
zTop[0] = cellSurfaceHeight[cell]  // Sea surface height
for k = 1 to mVertLevels:
    zTop[k] = zTop[k-1] - layerThickness[k-1]
zTop[mVertLevels] = cellBottomDepth[cell]  // Seafloor
```

**Example**:
```cpp
solution.calcCellCenterZtop();
// Now cellZTop_vec contains cumulative depth profile
```

#### `void calcCellVertexZtop(MPASOGrid* grid, std::string& dataDir, const RuntimeContext& ctx)`
Interpolate zTop from cell centers to vertices.

**Parameters**:
- `grid`: Contains vertex-to-cell connectivity
- `dataDir`: Cache directory (can save/load results)
- `ctx`: Runtime context

**Output**: Populates `cellVertexZTop_vec`

**Algorithm**: Average zTop from 3 adjacent cells for each vertex and layer.

**Critical for**: Particle tracing velocity interpolation (needs zTop at vertices for Wachspress weights).

**Example**:
```cpp
solution.calcCellVertexZtop(&grid, cacheDir, ctx);
// Now cellVertexZTop_vec ready for particle kernel
```

### Attribute Management

#### `void addAttribute(std::string name, AttributeFormat type)`
Register a new custom attribute.

**Parameters**:
- `name`: Attribute identifier (e.g., "temperature")
- `type`: Data type (`kDouble`, `kFloat`, `kChar`, `kVec3`)

**Effect**: Creates entry in appropriate map (`mDoubleAttributes[name]`, etc.)

**Example**:
```cpp
solution.addAttribute("temperature", AttributeFormat::kDouble);
solution.mDoubleAttributes["temperature"].resize(mCellsSize * mVertLevels);
// Fill with data from netCDF or computation
```

#### `void cleanInvalidCellCenterValues(const std::string& name, MPASOGrid* grid)`
Replace invalid/masked values with interpolated estimates from neighbors.

**Parameters**:
- `name`: Attribute name in `mDoubleAttributes` or `mFloatAttributes`
- `grid`: Grid with connectivity for neighbor lookup

**Algorithm**:
1. Scan attribute array for values < `INVALID_VALUE_THRESHOLD` (-1e20)
2. For each invalid cell, average valid neighbors
3. Replace invalid value with neighbor average
4. If all neighbors invalid, mark as `INVALID_VALUE` (-999.0)

**Use Case**: Fill gaps in temperature/salinity near land boundaries.

**Example**:
```cpp
solution.cleanInvalidCellCenterValues("temperature", &grid);
```

#### `void calcCellCenterToVertex(const std::string& name, const std::vector<double>& vec, MPASOGrid* grid, std::string& dataDir, const RuntimeContext& ctx)`
Interpolate arbitrary cell-centered attribute to vertices.

**Parameters**:
- `name`: Attribute name (will be stored in `mDoubleAttributes_CtoV[name]`)
- `vec`: Cell-centered data `[mCellsSize × mVertLevels]`
- `grid`: Grid with connectivity
- `dataDir`: Cache directory
- `ctx`: Runtime context

**Output**: Populates `mDoubleAttributes_CtoV[name]` with `[mVertexSize × mVertLevels]` data

**Example**:
```cpp
solution.calcCellCenterToVertex("temperature", 
                                 solution.mDoubleAttributes["temperature"],
                                 &grid, cacheDir, ctx);
// Now mDoubleAttributes_CtoV["temperature"] has vertex-interpolated values
```

### Attribute Setters

#### `void setAttribute(GridAttributeType type, int val)`
Set dimensional attribute.

**Supported Types**:
- `kCellSize`, `kVertexSize`, `kEdgeSize`, `kMaxEdgeSize`, `kVertLevels`

**Example**:
```cpp
solution.setAttribute(GridAttributeType::kVertLevels, 80);
```

#### `void setAttributesVec3(AttributeType type, const std::vector<vec3>& vec)`
Set vec3 field (e.g., velocity).

**Supported Types**:
- `kVelocity` → `cellCenterVelocity_vec`

#### `void setAttributesDouble(AttributeType type, const std::vector<double>& vec)`
Set scalar field.

**Supported Types**:
- `kZonalVelocity`, `kMeridionalVelocity`, `kNormalVelocity`, `kZTop`, `kLayerThickness`, `kBottomDepth`

### Utilities

#### `int getID() const`
Get unique integer hash of this solution's timestamp + timestep.

**Returns**: 32-bit FNV-1a hash

**Use Case**: Fast solution comparison or caching key generation.

#### `std::string getTimeStamp() const`
Get timestamp string.

**Returns**: `mTimeStamp` (e.g., "2020-01-15_00:00:00")

#### `bool checkAttribute()`
Validate that all required fields are properly initialized.

**Returns**: `true` if dimensions consistent and critical arrays non-empty.

#### `void setTimestep(int timestep)`
Set timestep index.

**Parameters**:
- `timestep`: Integer timestep identifier

### Deprecated Methods

These methods access data with older indexing conventions. **Do not use in new code**:
- `getCellVelocity()`, `getCellVertVelocity()`, `getCellZTop()`
- `getEdgeNormalVelocity()`, `getCellSurfaceMeridionalVelocity()`
- `getCellCenterZTop()`, `getCellVertexZTop()`
- `getCellCenterVelocity()`, `getCellVertexVelocity()`

**Reason**: These assume CPU-only access and don't work with GPU-resident data.

## Usage Examples

### Loading and Preparing a Solution

```cpp
#include "Core/MPASOSolution.h"
#include "Core/MPASOGrid.h"
#include "IO/MPASOReader.h"

// Initialize grid
MPASOReader reader("data/ocean.nc");
MPASOGrid grid;
grid.initGrid(&reader);

// Load solution at timestep 0
reader.setTimestep(0);
MPASOSolution solution;
solution.initSolution(&reader);

// Compute derived velocity fields
RuntimeContext ctx(RuntimeBackend::CUDA, 0);  // GPU 0
std::string cacheDir = "cache/";

solution.calcCellCenterZtop();  // Vertical coordinate
solution.calcCellCenterVelocity(&grid, cacheDir, ctx);  // 3D velocity
solution.calcCellVertexZtop(&grid, cacheDir, ctx);       // Vertex zTop
solution.calcCellVertexVelocity(&grid, cacheDir, ctx);   // Vertex velocity
solution.calcCellVertexVertVelocity(&grid, cacheDir, ctx); // Vertical velocity

// Now solution is ready for particle tracing or visualization
```

### Loading Temperature and Interpolating to Vertices

```cpp
// Add temperature attribute
solution.addAttribute("temperature", AttributeFormat::kDouble);

// Load from netCDF (assuming reader supports it)
reader.readVariable("temperature", solution.mDoubleAttributes["temperature"]);

// Clean invalid values (land/masked cells)
solution.cleanInvalidCellCenterValues("temperature", &grid);

// Interpolate to vertices for particle tracking
solution.calcCellCenterToVertex("temperature",
                                 solution.mDoubleAttributes["temperature"],
                                 &grid, cacheDir, ctx);

// Access interpolated temperature
double temp_at_vertex = solution.mDoubleAttributes_CtoV["temperature"][vertex * mVertLevels + layer];
```

### Time-Varying Field Setup (for PathLine)

```cpp
// Load two consecutive timesteps
reader.setTimestep(10);
MPASOSolution solution_t0;
solution_t0.initSolution(&reader);
solution_t0.calcCellCenterVelocity(&grid, cacheDir, ctx);
solution_t0.calcCellVertexVelocity(&grid, cacheDir, ctx);

reader.setTimestep(11);
MPASOSolution solution_t1;
solution_t1.initSolution(&reader);
solution_t1.calcCellCenterVelocity(&grid, cacheDir, ctx);
solution_t1.calcCellVertexVelocity(&grid, cacheDir, ctx);

// Pair them in MPASOField for temporal interpolation
MPASOField field;
field.initField(&grid, &solution_t0, &solution_t1);
// PathLine can now interpolate velocity between t0 and t1
```

### Custom Attribute Pipeline

```cpp
// Register custom attribute (e.g., computed vorticity)
solution.addAttribute("vorticity", AttributeFormat::kFloat);
solution.mFloatAttributes["vorticity"].resize(solution.mCellsSize * solution.mVertLevels);

// Compute vorticity from velocity (pseudocode)
for (int cell = 0; cell < solution.mCellsSize; ++cell) {
    for (int layer = 0; layer < solution.mVertLevels; ++layer) {
        float vort = computeVorticity(cell, layer, &grid, &solution);
        solution.mFloatAttributes["vorticity"][cell * solution.mVertLevels + layer] = vort;
    }
}

// Interpolate to vertices (if needed for visualization)
// (Convert float vector to double for calcCellCenterToVertex)
std::vector<double> vort_double(solution.mFloatAttributes["vorticity"].begin(),
                                  solution.mFloatAttributes["vorticity"].end());
solution.calcCellCenterToVertex("vorticity", vort_double, &grid, cacheDir, ctx);
```

### Accessing Velocity Components

```cpp
// Access 3D Cartesian velocity at cell center
int cellID = 1000;
int layer = 10;
vec3 velocity = solution.cellCenterVelocity_vec[cellID * solution.mVertLevels + layer];
std::cout << "Velocity at cell " << cellID << " layer " << layer 
          << ": (" << velocity.x << ", " << velocity.y << ", " << velocity.z << ")\n";

// Access vertex velocity (for interpolation)
int vertexID = 500;
vec3 vertexVel = solution.cellVertexVelocity_vec[vertexID * solution.mVertLevels + layer];

// Access zTop at vertex (for depth finding)
double ztop_top = solution.cellVertexZTop_vec[vertexID * solution.mTotalZTopLayerP1 + layer];
double ztop_bot = solution.cellVertexZTop_vec[vertexID * solution.mTotalZTopLayerP1 + layer + 1];
std::cout << "Layer " << layer << " at vertex " << vertexID 
          << " spans [" << ztop_bot << ", " << ztop_top << "] meters\n";
```

## Implementation Notes

### Memory Layout

**Strided 2D arrays** (cell/vertex × layers):
- Cell attribute at (cell `c`, layer `k`): `array[c * mVertLevels + k]`
- Vertex attribute at (vertex `v`, layer `k`): `array[v * mVertLevels + k]`
- zTop at (entity, interface `k`): `array[entity * mTotalZTopLayerP1 + k]`

**GPU-friendly**: Consecutive layers for same cell/vertex are contiguous (cache-friendly for vertical interpolation).

### Invalid Value Convention

- **MPAS files**: Use `_FillValue` or values < -1e20 (e.g., -1e34) for masked cells
- **MOPS internal**: Convert to `INVALID_VALUE = -999.0` for consistency
- **Threshold**: `INVALID_VALUE_THRESHOLD = -1e20` separates valid from invalid data
- **Particle kernels**: Check against `masked_attr_value` parameter (user-configurable)

### Velocity Computation Order

**Recommended pipeline**:
1. `initSolution()` - loads edge-normal velocity
2. `calcCellCenterZtop()` - compute vertical coordinate
3. `calcCellCenterVelocity()` or `calcCellCenterVelocityByZM()` - cell-centered velocity
4. `calcCellVertexZtop()` - vertex zTop (needed for next step)
5. `calcCellVertexVelocity()` - vertex velocity (depends on step 4 for correct interpolation)
6. `calcCellVertexVertVelocity()` - vertical velocity at vertices

**Why this order**: Vertex computations require cell-centered data and zTop for spatial interpolation weights.

### RuntimeContext vs GPUContext

Modern code should use `RuntimeContext`:
```cpp
RuntimeContext ctx(RuntimeBackend::CUDA, 0);  // GPU device 0
solution.calcCellVertexVelocity(&grid, cacheDir, ctx);
```

Legacy `GPUContext` still supported for compatibility:
```cpp
GPUContext gpu_ctx(0);  // Device 0
solution.calcCellVertexVelocity(&grid, cacheDir, gpu_ctx);
```

### Performance Considerations

- **CPU vs GPU**: Velocity computations are GPU-accelerated when `RuntimeBackend::CUDA` or `RuntimeBackend::SYCL`
- **Caching**: Binary files in `dataDir` cache results of expensive computations (zTop, vertex interpolation)
- **Cache hit**: If `{dataDir}/{operation}_result.bin` exists, computation is skipped
- **Large meshes**: 1M cells × 100 layers = 100M values → ~400 MB per velocity field

### Thread Safety

**NOT thread-safe**. Each solution object should be:
- Initialized in single thread
- Passed to GPU kernels via pointers (read-only after preparation)
- Not modified during particle tracing or visualization

## Related Classes

- **MPASOGrid**: Provides mesh topology and connectivity for interpolation operations
- **MPASOField**: Combines grid + solution(s) for spatiotemporal queries
- **MPASOVisualizer**: Uses solution data for rendering and particle tracing
- **MPASOReader**: Reads MPAS netCDF files and populates solution arrays

## References

- [MPAS-Ocean User's Guide](https://mpas-dev.github.io/ocean/ocean.html) - Variable definitions
- [MPAS Output Fields](https://mpas-dev.github.io/ocean/ocean_variables.html) - Standard variables
- Source files: `/pscratch/sd/q/qiuyf/MOPS/src/Core/MPASOSolution.{h,cpp}`
