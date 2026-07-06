# CUDA Pathline Computation and Boundary-Condition Logic

This note summarizes how `MPASOVisualizerKernels.cu` computes pathlines and how the CUDA kernel handles different boundary conditions. The focus is on the pathline-related functions and the boundary-decision logic.

---

## 1. Main Function Map

| Function name | Role in the pathline pipeline |
|---|---|
| `PathLine(...)` | CPU-side driver. It validates inputs, prepares seed points and depths, finds initial cells, allocates/copies GPU buffers, launches `KernelPathLine(...)`, copies results back, and finalizes trajectory lines. |
| `KernelPathLine(...)` | GPU kernel. One CUDA thread integrates one particle over all time steps. This is the core pathline integration loop. |
| `CalcVelocityAtPathlineCUDA(...)` | Computes the interpolated horizontal velocity, vertical velocity, and attributes at a given particle position, depth, cell, and time interpolation factor `alpha`. |
| `CalcVelocityAtPathlineWithConstraintsCUDA(...)` | Wrapper around `CalcVelocityAtPathlineCUDA(...)`. It additionally tries edge projection when the horizontal velocity is extremely small but nonzero. |
| `FindContainingCellInCurrentOrNeighborsCUDA(...)` | Checks whether a candidate position is inside the current cell or one of its neighboring cells. This distinguishes normal cell crossing from true lateral boundary hits. |
| `FindNearestCellEdgeCUDA(...)` | Finds the closest edge of the current cell to the particle position. Used when a particle hits a lateral boundary. |
| `ProjectVelocityOntoEdgeCUDA(...)` | Projects the particle velocity onto the tangent direction of the nearest boundary edge. This allows the particle to slide along the boundary instead of crossing it. |
| `GenerateRandomTangentVelocityCUDA(...)` | Generates a deterministic pseudo-random tangent velocity on the spherical surface. Used as a fallback escape direction. |
| `CalcZTopAtLevel0CUDA(...)` | Computes the interpolated surface `zTop` at the current horizontal position. Used for surface boundary clamping. |
| `CalcZTopAtBottomCUDA(...)` | Computes the interpolated bottom `zTop` at the current horizontal position. Used for bottom boundary clamping. |
| `AdvectOnSphereCUDA(...)`, `CalcRotationAxis(...)`, `CalcPositionAfterRotation(...)` | Move the particle along the spherical surface using a rotation-based update instead of a simple Cartesian `pos += v * dt`. |

---

## 2. High-Level Pathline Pipeline

The pathline computation is driven by `PathLine(...)` on the CPU side and `KernelPathLine(...)` on the GPU side.

At the CPU level, `PathLine(...)` does the following:

```cpp
PathLine(mpasoF, points, config, default_cell_id)
{
    validate mpasoF, config, mGrid, mSol_Front, mSol_Back;

    stable_points = input seed points;

    host_buffers = InitTrajectoryOutputBuffers(...);
    effective_depths = BuildEffectiveDepths(stable_points, config, "PathLine");
    trajectory_lines = InitTrajectoryLines(stable_points, effective_depths, config);

    if default_cell_id is not provided:
        for each seed point:
            default_cell_id[i] = mGrid->searchKDT(seed_point[i]);

    build grid_info:
        actual_cell_size
        actual_max_edge_size
        actual_vertex_size
        actual_ztop_layer
        actual_ztop_layer_p1

    copy seed points, depths, grid arrays, velocity arrays, zTop arrays,
    vertical velocity arrays, and optional attributes to GPU;

    launch KernelPathLine<<<grid, block>>>(...);

    copy output points, velocities, attributes, and boundary-hit counts back;

    print lateral-boundary statistics;

    return FinalizeTrajectoryLinesWithAttrs(...);
}
```

The actual numerical integration happens inside `KernelPathLine(...)`. The model is:

```text
one CUDA thread = one particle
```

Each thread integrates its particle from the initial seed position through `n_steps`.

---

## 3. Time Interpolation Between `mSol_Front` and `mSol_Back`

This is a pathline, not a fixed-time streamline. The velocity field is time-dependent between two snapshots:

```cpp
mSol_Front
mSol_Back
```

Inside `KernelPathLine(...)`, each step computes:

```cpp
double alpha = static_cast<double>(i_step) / static_cast<double>(n_steps);
```

Then `CalcVelocityAtPathlineCUDA(...)` interpolates between front and back fields:

```cpp
v = (1.0 - alpha) * v_front + alpha * v_back;
w = (1.0 - alpha) * w_front + alpha * w_back;
attr = (1.0 - alpha) * attr_front + alpha * attr_back;
```

Conceptually:

```text
alpha = 0    -> use mSol_Front
alpha = 1    -> use mSol_Back
0 < alpha < 1 -> temporal interpolation between the two snapshots
```

Note: in the current loop, `i_step` runs from `0` to `n_steps - 1`, so `alpha` approaches but does not exactly reach `1.0` in the Euler path.

---

## 4. Cell Initialization and Cell Update

### 4.1 Initial Cell

The initial cell is prepared in `PathLine(...)`:

```cpp
if default_cell_id.size() != stable_points.size():
    default_cell_id.assign(stable_points.size(), -1);
    for each seed point:
        mGrid->searchKDT(stable_points[i], cell_id);
        default_cell_id[i] = cell_id;
```

Inside `KernelPathLine(...)`, the first loop uses this initial cell:

```cpp
cell_id = default_cell_id[global_id];

if (cell_id < 0 || cell_id >= actual_cell_size) {
    return;
}
```

So if the initial cell is invalid, the particle is terminated immediately.

### 4.2 Cell Update During Integration

After the first step, the kernel does not perform a global KD-tree search. Instead, it searches the current cell and its neighbors.

The logic is approximately:

```cpp
if not first_loop:
    current_cell_vertices_number = number_vertex_on_cell[cell_id];

    best_cell = cell_id;
    min_distance = INF;

    for cid in cell_neig_vec:
        if cid is valid:
            d = length(cell_coord[cid] - sample_point_position);
            if d < min_distance:
                min_distance = d;
                best_cell = cid;

    cell_id = best_cell;

    // Important bug fix in the new version:
    current_cell_vertices_number = number_vertex_on_cell[cell_id];

    GetCellNeighborsIdx(cell_id, current_cell_vertices_number, cell_neig_vec, ...);
```

This update assumes that one time step is small enough that a particle usually remains in the current cell or moves into a one-ring neighboring cell.

---

## 5. Velocity Computation in `CalcVelocityAtPathlineCUDA(...)`

The velocity computation has three interpolation layers:

```text
1. Horizontal interpolation inside the MPAS-O polygonal cell
2. Vertical interpolation between zTop layers
3. Temporal interpolation between front and back snapshots
```

### 5.1 Early Failure Cases

`CalcVelocityAtPathlineCUDA(...)` returns a zero state with `ok = false` when it cannot compute a reliable velocity:

```cpp
return {vec3{0,0,0}, 0.0, vec3{0,0,0}, false};
```

Typical failure cases include:

```text
cell_id < 0
invalid number of vertical layers
invalid number of cell vertices
position is not inside the given cell
invalid vertex id
Wachspress interpolation input failure
cannot find a valid local vertical layer
vertical interpolation denominator is too small
```

Pseudo-code:

```cpp
CalcVelocityAtPathlineCUDA(pos, cell_id, current_depth, alpha)
{
    if cell_id is invalid:
        return zero_state(false);

    if vertical layer count is invalid:
        return zero_state(false);

    current_cell_vertices_number = number_vertex_on_cell[cell_id];
    if vertex count is invalid:
        return zero_state(false);

    if !IsInMesh(cell_id, pos):
        return zero_state(false);

    vertices = GetCellVerticesIdx(cell_id);
    vertex_positions = GetCellVertexPos(vertices);

    weights = CalcPolygonWachspress(pos, vertex_positions);

    compute zTop_front[k] and zTop_back[k] by vertex-weighted interpolation;
    enforce monotonic decreasing zTop profiles;

    local_layer_front = find vertical layer containing current_depth in zTop_front;
    local_layer_back  = find vertical layer containing current_depth in zTop_back;

    if layer search fails:
        return zero_state(false);

    interpolate velocity vertically in front snapshot;
    interpolate velocity vertically in back snapshot;

    h_vel = (1 - alpha) * h_vel_front + alpha * h_vel_back;
    v_vel = (1 - alpha) * v_vel_front + alpha * v_vel_back;
    attr  = (1 - alpha) * attr_front  + alpha * attr_back;

    return {h_vel, v_vel, attr, true};
}
```

### 5.2 Horizontal Interpolation

The code obtains the current cell vertices and computes Wachspress weights:

```cpp
GetCellVerticesIdx(...);
GetCellVertexPos(...);
Interpolator::CalcPolygonWachspress(...);
```

Then any vertex-defined field is interpolated as:

```cpp
Q(pos) = sum_i weight_i(pos) * Q(vertex_i);
```

This is used for:

```text
cell vertex horizontal velocity
cell vertex zTop
cell vertex vertical velocity
cell vertex attributes
```

### 5.3 Vertical Interpolation

The kernel first computes the local vertical `zTop` profile at the particle position:

```cpp
zTop_front[k] = sum_i weight_i * cell_vertex_ztop_front[vertex_i, k];
zTop_back[k]  = sum_i weight_i * cell_vertex_ztop_back [vertex_i, k];
```

Then it enforces monotonicity:

```cpp
if zTop[k] > zTop[k-1]:
    zTop[k] = zTop[k-1] - 1e-9;
```

The code searches for the layer `k` such that:

```cpp
zTop[k-1] >= current_depth >= zTop[k]
```

Then it computes the interpolation ratio:

```cpp
t = (current_depth - zTop_down) / (zTop_up - zTop_down);
```

And vertically interpolates:

```cpp
velocity = t * velocity_up + (1 - t) * velocity_down;
```

### 5.4 Depth Sign Convention

Inside `KernelPathLine(...)`, the code uses:

```cpp
double current_depth = -1.0 * particle_depths[global_id];
```

This implies the following convention:

```text
particle_depths[global_id] : positive downward depth, e.g. 20 m
current_depth              : negative z-coordinate, e.g. -20 m
zTop arrays                : likely negative downward vertical coordinates
```

This sign convention is important for both vertical interpolation and surface/bottom boundary checks.

---

## 6. Position Integration: Euler and RK4

The kernel supports two integration modes:

```cpp
use_euler == true   -> Euler update
use_euler == false  -> RK4-like update
```

### 6.1 Euler Update

For Euler, the code computes the velocity once at the current position:

```cpp
s = CalcVelocityAtPathlineWithConstraintsCUDA(current_position, cell_id, current_depth, alpha, ...);
current_horizontal_velocity = s.h_vel;
current_vertical_velocity   = s.v_vel;
current_attrs               = s.attr;
```

Then the horizontal position is advanced on the sphere:

```cpp
rotation_axis = CalcRotationAxis(current_position, current_horizontal_velocity);
speed = length(current_horizontal_velocity);
theta = speed * delta_t / r;
new_position = CalcPositionAfterRotation(current_position, rotation_axis, theta);
```

Pseudo-code:

```cpp
EulerStep(pos, h_vel, dt)
{
    r = length(pos);
    speed = length(h_vel);
    axis = CalcRotationAxis(pos, h_vel);
    theta = speed * dt / r;
    return CalcPositionAfterRotation(pos, axis, theta);
}
```

This is not a simple Cartesian update. It is a rotation-based spherical update, which keeps the particle on a spherical shell.

### 6.2 RK4-Like Update

For RK4, the code evaluates velocity four times:

```cpp
s1 = V(pos, depth, alpha);

p2 = AdvectOnSphereCUDA(pos, s1.h_vel, dt/2);
s2 = V(p2, depth, alpha + dalpha/2);

p3 = AdvectOnSphereCUDA(pos, s2.h_vel, dt/2);
s3 = V(p3, depth, alpha + dalpha/2);

p4 = AdvectOnSphereCUDA(pos, s3.h_vel, dt);
s4 = V(p4, depth, alpha + dalpha);
```

Then it averages the states:

```cpp
h_vel = (s1.h_vel + 2*s2.h_vel + 2*s3.h_vel + s4.h_vel) / 6;
v_vel = (s1.v_vel + 2*s2.v_vel + 2*s3.v_vel + s4.v_vel) / 6;
attr  = (s1.attr  + 2*s2.attr  + 2*s3.attr  + s4.attr ) / 6;
```

Finally it advances the particle on the sphere using the averaged horizontal velocity.

Important note: the intermediate RK4 positions `p2`, `p3`, and `p4` are evaluated using the same `cell_id` and the same `current_depth`. Therefore, this is not a fully coupled 3D RK4 over `(position, depth, cell_id, time)`. It is a practical RK4-like improvement mainly for the horizontal position and time interpolation.

---

## 7. Vertical Boundary Conditions

After the horizontal position is advanced, the kernel updates the depth and checks surface/bottom constraints.

### 7.1 Depth Update

The code stores particle depth as a positive downward quantity:

```cpp
old_depth = particle_depths[global_id];
new_depth = old_depth - current_vertical_velocity * delta_t;
```

Pseudo-code:

```cpp
old_depth = particle_depths[id];
new_depth = old_depth - w * dt;
```

### 7.2 Surface Boundary Projection

The surface boundary is handled as a **projection/clamping step** after the particle has been tentatively advanced by the velocity field.

The idea is not to simply discard the step. Instead, the algorithm first computes a tentative next position at time `t + 1`. If this tentative position is above the sea surface, the particle is pulled back onto the sea surface.

Conceptually:

```text
                  t+1 (old)
                      ●
                      │  projection back to surface
                      ↓
sea surface   ─────── ● ───────
                  t+1 (new)
                 /
                /
              ●
              t
```

Therefore, the surface boundary condition is:

```txt
If the tentative particle depth is above the local sea surface,
move the particle back to the sea-surface depth at the new horizontal location.
```

In the code, the local sea-surface level is obtained by interpolating the level-0 `zTop` field at the tentative horizontal position:

```cpp
surface_ztop = CalcZTopAtLevel0CUDA(
    new_position,
    cell_id,
    ...,
    cell_vertex_ztop_front
);
```

If `particle_depths` uses positive-downward depth, while `surface_ztop` uses the model vertical coordinate, the corresponding surface depth is:

```c++
surface_depth = -surface_ztop;
```

Then the tentative depth is computed from the vertical velocity:

```c++
old_depth = particle_depths[global_id];
new_depth_raw = old_depth - current_vertical_velocity * delta_t;
```

The surface-boundary projection should be interpreted as:

```c++
if (new_depth_raw < surface_depth) {
    // The tentative particle position is above the sea surface.
    // Project it back onto the sea surface.
    new_depth = surface_depth;
} else {
    // The particle remains inside the water column.
    new_depth = new_depth_raw;
}

new_depth = max(new_depth, 0.0);
```

A more explicit pseudocode version is:

```c++
// 1. First compute the tentative next position using the velocity field.
new_position_raw = RK4(current_position);
new_depth_raw = old_depth - vertical_velocity * delta_t;

// 2. Evaluate the sea-surface depth at the tentative horizontal location.
surface_ztop = CalcZTopAtLevel0CUDA(new_position_raw, cell_id, ...);
surface_depth = -surface_ztop;

// 3. If the tentative point is above the surface, project it back.
if (new_depth_raw < surface_depth) {
    new_depth = surface_depth;

    // Keep the horizontal movement, but correct the vertical/radial coordinate.
    new_position = ProjectPositionToDepth(new_position_raw, new_depth);
} else {
    new_depth = new_depth_raw;
    new_position = new_position_raw;
}
```

### 7.3 Bottom Boundary: Bottom-Following Projection

The key idea is to preserve the particle's relative distance from the bottom while it moves horizontally from the current cell to a new cell.

At time `t`, the particle is located inside the current cell. Before applying the bottom correction, the algorithm first measures the particle's vertical clearance from the local bottom:

```cpp
bottom_ztop_t = CalcZTopAtBottomCUDA(
    current_position,
    cell_id,
    ...,
    cell_vertex_ztop_front
);

bottom_depth_t = -bottom_ztop_t;
d = bottom_depth_t - old_depth;
```

Here, `old_depth` is assumed to be a positive-downward depth. Therefore, `d` represents the distance between the particle and the local seafloor at time `t`:

```text
d > 0  : the particle is above the seafloor
d = 0  : the particle is exactly on the seafloor
d < 0  : the particle is already below the seafloor, which should be corrected
```

After the velocity integration, the algorithm obtains a tentative next position:

```cpp
new_position_raw = RK4(current_position);

new_depth_raw = old_depth - vertical_velocity * delta_t;
```

This tentative point is the point labeled `t+1 (old)` in the diagram. It may move into another cell, and the bottom depth in that target cell may be very different from the bottom depth in the original cell.

Therefore, before applying the bottom correction, the algorithm should determine which cell contains the tentative horizontal position:

```cpp
target_cell_id = FindContainingCellInCurrentOrNeighborsCUDA(
    new_position_raw,
    cell_id,
    ...
);
```

Then the local bottom depth at the tentative horizontal location is evaluated in the target cell:

```cpp
bottom_ztop_next = CalcZTopAtBottomCUDA(
    new_position_raw,
    target_cell_id,
    ...,
    cell_vertex_ztop_front
);

bottom_depth_next = -bottom_ztop_next;
```

Instead of directly clamping the particle to the bottom, the particle is projected to a new depth that preserves the previous bottom clearance `d`:

```cpp
new_depth_projected = bottom_depth_next - d;
```

In other words, if the particle was `d` meters above the bottom at time `t`, then after moving to the new horizontal position, it is placed `d` meters above the bottom of the target cell.

Conceptually:

```text
At time t in the current cell:

    particle depth = old_depth
    local bottom   = bottom_depth_t

    d = bottom_depth_t - old_depth


At tentative t+1 in the target cell:

    local bottom at new horizontal position = bottom_depth_next

    projected depth = bottom_depth_next - d
```

The final bottom-boundary rule is therefore:

```cpp
if (new_depth_raw > bottom_depth_next) {
    // The tentative particle position is below the seafloor.
    // Project it back using the previous bottom clearance.
    new_depth = bottom_depth_next - d;
} else {
    // The particle remains inside the water column.
    new_depth = new_depth_raw;
}
```

For robustness, the projected depth should also be clamped into the valid water column:

```cpp
new_depth = max(new_depth, surface_depth);
new_depth = min(new_depth, bottom_depth_next);
```

A more complete pseudocode version is:

```cpp
// 1. Compute the distance from the current particle to the bottom
//    in the current cell at time t.
bottom_ztop_t = CalcZTopAtBottomCUDA(
    current_position,
    cell_id,
    ...,
    cell_vertex_ztop_front
);

bottom_depth_t = -bottom_ztop_t;
d = bottom_depth_t - old_depth;

// Make sure the clearance is non-negative.
d = max(d, 0.0);


// 2. Compute the tentative t+1 point.
new_position_raw = AdvectOnSphere(
    current_position,
    horizontal_velocity,
    delta_t
);

new_depth_raw = old_depth - vertical_velocity * delta_t;


// 3. Find which cell contains the tentative horizontal position.
target_cell_id = FindContainingCellInCurrentOrNeighborsCUDA(
    new_position_raw,
    cell_id,
    ...
);


// 4. Compute the bottom depth in the target cell.
bottom_ztop_next = CalcZTopAtBottomCUDA(
    new_position_raw,
    target_cell_id,
    ...,
    cell_vertex_ztop_front
);

bottom_depth_next = -bottom_ztop_next;


// 5. If the tentative point goes below the bottom,
//    project it upward while preserving the old bottom clearance d.
if (new_depth_raw > bottom_depth_next) {
    new_depth = bottom_depth_next - d;
} else {
    new_depth = new_depth_raw;
}


// 6. Keep the particle inside the water column.
new_depth = max(new_depth, surface_depth);
new_depth = min(new_depth, bottom_depth_next);


// 7. Keep the tentative horizontal movement, but update the radial/depth position.
new_position = ProjectPositionToDepth(
    new_position_raw,
    new_depth
);
```

This behavior matches the intended bottom-boundary handling in the diagram:

```text
1. At time t, compute the distance d from the particle to the bottom
   in the current cell.

2. First let the particle move horizontally to its tentative t+1 position.

3. Determine the cell that contains the tentative t+1 horizontal position.

4. Evaluate the bottom depth in that target cell.

5. If the tentative particle penetrates the seafloor, move it upward
   at the same horizontal location so that it keeps the same distance d
   above the new local bottom.
```

This is different from a simple bottom clamp. A simple clamp would set:

```cpp
new_depth = bottom_depth_next;
```

which places the particle directly on the seafloor. The bottom-following projection instead sets:

```cpp
new_depth = bottom_depth_next - d;
```

so the particle remains inside the water column while preserving its relative height above the bottom.

## 8. Lateral Boundary Conditions

The most important improvement in the new code is that a candidate position is no longer immediately treated as a boundary hit just because it is outside the old cell.

The new code calls:

```cpp
FindContainingCellInCurrentOrNeighborsCUDA(new_position, cell_id, ...);
```

This distinguishes:

```text
normal cell crossing
vs.
true lateral boundary / land / outside-ocean-domain hit
```

### 8.1 Normal Cell Crossing

The logic of `FindContainingCellInCurrentOrNeighborsCUDA(...)` is:

```cpp
FindContainingCellInCurrentOrNeighborsCUDA(pos, current_cell_id)
{
    if IsInMesh(current_cell_id, pos):
        return current_cell_id;

    neighbors = GetCellNeighborsIdx(current_cell_id);

    for each neighbor_cell_id:
        if IsInMesh(neighbor_cell_id, pos):
            return neighbor_cell_id;

    return -1;
}
```

Then `KernelPathLine(...)` uses:

```cpp
containing_cell = FindContainingCellInCurrentOrNeighborsCUDA(new_position, cell_id, ...);

if (containing_cell >= 0) {
    if (containing_cell != cell_id) {
        cell_id = containing_cell;
        current_cell_vertices_number = number_vertex_on_cell[cell_id];
        GetCellNeighborsIdx(cell_id, current_cell_vertices_number, cell_neig_vec, ...);
    }

    // Accept new_position.
}
```

This means:

```text
If the new position is inside the current cell or a neighbor cell, it is a valid ocean position.
The particle is allowed to cross from one MPAS cell to another.
```

This fixes the previous problem where ordinary cell crossing could be mistaken for a lateral boundary hit.

### 8.2 True Lateral Boundary Hit

If `FindContainingCellInCurrentOrNeighborsCUDA(...)` returns `-1`, the code treats the position as truly outside the valid ocean mesh:

```cpp
if (containing_cell < 0) {
    boundary_hit_count[global_id]++;
    handle lateral boundary;
}
```

The boundary-handling sequence is:

```text
1. Count the boundary hit.
2. Find the nearest edge of the current cell.
3. Project the horizontal velocity onto that edge direction.
4. Recompute a candidate position using the projected velocity.
5. Validate the candidate position.
6. If projection fails, try multiple random tangent escape directions.
7. If all attempts fail, keep the particle at the old position.
```

### 8.3 Edge Projection

The nearest edge is found with:

```cpp
FindNearestCellEdgeCUDA(current_position, cell_id, ...,
                        edge_va_idx, edge_vb_idx);
```

Then the velocity is projected onto the edge tangent direction:

```cpp
projected_vel = ProjectVelocityOntoEdgeCUDA(
    current_horizontal_velocity,
    current_position,
    va,
    vb);
```

Conceptually:

```cpp
edge_vec = vb - va;
radial = normalize(current_position);
edge_tangent = edge_vec - dot(edge_vec, radial) * radial;
edge_tangent = normalize(edge_tangent);
projected_vel = dot(velocity, edge_tangent) * edge_tangent;
```

This removes the component that would push the particle out of the ocean domain and keeps only the along-boundary component.

Then the kernel tries to move along that projected velocity:

```cpp
candidate_position = AdvectOnSphere(current_position, projected_vel, delta_t);
candidate_position = normalize(candidate_position) * r_new;
```

The candidate is accepted only if:

```cpp
IsInMesh(cell_id, candidate_position, ...)
```

### 8.4 Random Tangent Escape

If edge projection fails, the kernel calls:

```cpp
TryMultipleRandomDirectionsCUDA(...);
```

This function tries up to 20 tangent directions. The speed increases every five attempts:

```cpp
BASE_KICK_SPEED = 0.01;
kick_speed = BASE_KICK_SPEED * (1.0 + (attempt / 5) * 2.0);
```

So the speed schedule is approximately:

```text
attempt 0-4:    0.01
attempt 5-9:    0.03
attempt 10-14:  0.05
attempt 15-19:  0.07
```

Pseudo-code:

```cpp
TryMultipleRandomDirectionsCUDA(...)
{
    for attempt in 0..19:
        kick_speed = progressively_larger_speed(attempt);

        random_vel = GenerateRandomTangentVelocityCUDA(
            current_position,
            global_id,
            i_step,
            kick_speed,
            attempt);

        random_candidate = AdvectOnSphere(current_position, random_vel, delta_t);
        random_candidate = normalize(random_candidate) * r_new;

        if IsInMesh(cell_id, random_candidate):
            out_position = random_candidate;
            return true;

    return false;
}
```

If random escape succeeds:

```cpp
new_position = escaped_position;
```

If all attempts fail:

```cpp
new_position = current_position;
```

So the particle is kept in a safe valid position instead of producing NaNs or escaping the mesh.

---

## 9. Stuck-Particle Velocity Constraint

`CalcVelocityAtPathlineWithConstraintsCUDA(...)` adds an extra check for extremely small horizontal velocity:

```cpp
state = CalcVelocityAtPathlineCUDA(...);

if (!state.ok):
    return state;

speed = length(state.h_vel);

if speed < 1e-9 and speed > 1e-20:
    nearest_edge = FindNearestCellEdgeCUDA(...);
    state.h_vel = ProjectVelocityOntoEdgeCUDA(state.h_vel, pos, edge);

return state;
```

The intention is:

```text
If a particle is nearly stuck near a boundary, try to move it along the nearest edge direction.
```

This is different from the lateral-boundary handler. This wrapper acts before the particle has actually crossed the boundary, while the lateral-boundary handler acts after a candidate position is found to be outside the current/neighbor ocean cells.

---

## 10. Complete Kernel-Level Pseudocode

```cpp
KernelPathLine(...)
{
    id = blockIdx.x * blockDim.x + threadIdx.x;

    if id >= particle_count:
        return;

    boundary_hit_count[id] = 0;

    initialize cell_id = -1;
    initialize neighbor list;

    for i_step in 0 .. n_steps-1:
    {
        alpha = i_step / n_steps;

        current_position = sample_points[id];
        current_depth = -particle_depths[id];

        if first_loop:
        {
            cell_id = default_cell_id[id];

            if cell_id is invalid:
                return;

            current_cell_vertices_number = number_vertex_on_cell[cell_id];
            GetCellNeighborsIdx(cell_id, current_cell_vertices_number, cell_neig_vec, ...);

            write initial point;
        }
        else:
        {
            if cell_id is invalid:
                return;

            // Update cell by nearest center among current/neighbor cells.
            best_cell = nearest cell center in cell_neig_vec;
            cell_id = best_cell;

            // Important fix: update vertex count after cell_id changes.
            current_cell_vertices_number = number_vertex_on_cell[cell_id];
            GetCellNeighborsIdx(cell_id, current_cell_vertices_number, cell_neig_vec, ...);
        }

        if use_euler:
        {
            s = CalcVelocityAtPathlineWithConstraintsCUDA(
                    current_position, cell_id, current_depth, alpha, ...);

            h_vel = s.h_vel;
            w_vel = s.v_vel;
            attr  = s.attr;

            new_position = AdvectOnSphere(current_position, h_vel, delta_t);
        }
        else:
        {
            s1 = V(current_position, current_depth, alpha);

            p2 = AdvectOnSphere(current_position, s1.h_vel, dt/2);
            s2 = V(p2, current_depth, alpha + dalpha/2);

            p3 = AdvectOnSphere(current_position, s2.h_vel, dt/2);
            s3 = V(p3, current_depth, alpha + dalpha/2);

            p4 = AdvectOnSphere(current_position, s3.h_vel, dt);
            s4 = V(p4, current_depth, alpha + dalpha);

            h_vel = (s1.h_vel + 2*s2.h_vel + 2*s3.h_vel + s4.h_vel) / 6;
            w_vel = (s1.v_vel + 2*s2.v_vel + 2*s3.v_vel + s4.v_vel) / 6;
            attr  = (s1.attr  + 2*s2.attr  + 2*s3.attr  + s4.attr ) / 6;

            new_position = AdvectOnSphere(current_position, h_vel, dt);
        }

        // Vertical update.
        old_depth = particle_depths[id];
        new_depth = old_depth - w_vel * delta_t;

        // Surface boundary.
        surface_ztop = CalcZTopAtLevel0CUDA(new_position, cell_id, ...);
        if particle is above surface:
            new_depth = -surface_ztop;
            new_depth = max(new_depth, 0.0);

        // Bottom boundary.
        bottom_ztop = CalcZTopAtBottomCUDA(new_position, cell_id, ...);
        if particle is below bottom:
            new_depth = bottom depth;

        new_depth = max(new_depth, 0.0);
        particle_depths[id] = new_depth;

        // Radius update.
        r_new = r + w_vel * delta_t;
        r_new = max(r_new, 1.0);
        new_position = normalize(new_position) * r_new;

        // Lateral boundary check.
        containing_cell = FindContainingCellInCurrentOrNeighborsCUDA(new_position, cell_id, ...);

        if containing_cell >= 0:
        {
            // Normal ocean-domain position.
            if containing_cell != cell_id:
                cell_id = containing_cell;
                current_cell_vertices_number = number_vertex_on_cell[cell_id];
                GetCellNeighborsIdx(cell_id, current_cell_vertices_number, cell_neig_vec, ...);
        }
        else:
        {
            // True lateral boundary hit.
            boundary_hit_count[id]++;

            edge = FindNearestCellEdgeCUDA(current_position, cell_id, ...);

            if edge found:
            {
                projected_vel = ProjectVelocityOntoEdgeCUDA(h_vel, current_position, edge);
                candidate = AdvectOnSphere(current_position, projected_vel, delta_t);
                candidate = normalize(candidate) * r_new;

                if IsInMesh(cell_id, candidate):
                    new_position = candidate;
                else if TryMultipleRandomDirectionsCUDA(...):
                    new_position = escaped_position;
                else:
                    new_position = current_position;
            }
            else:
            {
                if TryMultipleRandomDirectionsCUDA(...):
                    new_position = escaped_position;
                else:
                    new_position = current_position;
            }
        }

        sample_points[id] = new_position;

        if this is a record step:
            write_points[...] = new_position;
            write_vels[...]   = h_vel;
            write_attrs[...]  = attr;
    }
}
```

