Now I have a complete view of the CUDA implementation. Let me provide detailed step-by-step algorithms for each GPU function.

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