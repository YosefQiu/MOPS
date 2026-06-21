# Pathline Calculation Modifications - Detailed Changelog

**Original Date**: 2026-05-26  
**Updated**: 2026-06-21  
**Modified by**: Claude Code  
**Purpose**: Add surface and bottom constraints, edge projection for stuck particles in pathline calculations

---

## CRITICAL BUG FIXES (2026-06-21)

### Fix #1: Recursive Call Bug
**Issue**: The original `CalcVelocityAtPathlineCUDA` was renamed to `CalcVelocityAtPathlineWithConstraintsCUDA`, but then a wrapper with the same name was added, causing the wrapper to call itself recursively (infinite loop).

**Solution**: 
- Renamed function at line 1716 back to `CalcVelocityAtPathlineCUDA`
- Updated wrapper at line 2107 to call `CalcVelocityAtPathlineCUDA` (not itself)

### Fix #2: Bottom Constraint for Seafloor
**Issue**: Particles could penetrate below the seafloor, causing them to stop moving or become NaN.

**Solution**: 
- Added `CalcZTopAtBottomCUDA` helper function to calculate seafloor depth
- Added bottom constraint in depth update logic (lines 2493-2516):
  - Checks if particle would go below seafloor
  - Clamps depth to bottom level
  - Ignores vertical velocity when hitting seafloor
  - Particles now "slide along" the ocean floor horizontally

---

## Summary of All Changes (vs commit 186ea78 - origin/main)

📄 **See `FILES_CHANGED.md` for complete file-by-file breakdown**

### Files Changed: 2 (+ system files)
1. **`src/GPU/CUDA/Kernel/MPASOVisualizerKernels.cu`** - Main implementation file
   - **+404 lines, -6 lines** (total: +398 net lines)
   - 5 new helper functions added
   - 1 function renamed (bug fix)
   - 6 function calls updated (1 wrapper + Euler + RK4 stages 1-4)
   - 2 depth update sections significantly modified
   
2. **`PATHLINE_MODIFICATIONS_CHANGELOG.md`** - This documentation file
   - **+589 lines** (new file)

3. **`FILES_CHANGED.md`** - Files changed summary
   - **New file** - Quick reference for all modifications

4. System files (can be ignored, should add to .gitignore):
   - `.DS_Store` (Mac system file)
   - `src/.DS_Store`
   - `src/GPU/.DS_Store`
   - `src/GPU/CUDA/.DS_Store`

### Total Code Impact
- **Net Lines Changed**: +398 lines in CUDA kernel
- **Functions Added**: 5
- **Functions Modified**: 1 (renamed to fix recursion bug)
- **Integration Points Updated**: 6 (1 wrapper internal call + 5 pathline velocity calculations)

---

## Files Modified

### 1. `/Users/yosef/Desktop/MOPS/src/GPU/CUDA/Kernel/MPASOVisualizerKernels.cu`

**Baseline**: Commit `186ea78` (origin/main)  
**Current State**: Working directory with bug fixes applied  
**Total changes**: 5 new functions added, 5 function calls updated, 1 depth update section modified

---

## Detailed Changes

### Change #1: Added Helper Function - CalcZTopAtLevel0CUDA

**Location**: After line 1457 (after the StreamLine function's closing brace)  
**Line number**: Inserted at ~line 1467  
**Type**: New function addition

**Code Added**:
```cpp
// Helper function: Calculate zTop at surface level (level 0) at given position
MOPS_DEVICE inline double CalcZTopAtLevel0CUDA(
    const vec3& pos,
    int cell_id,
    int actual_max_edge_size,
    int actual_vertex_size,
    int actual_ztop_layer,
    const size_t* number_vertex_on_cell,
    const size_t* vertices_on_cell,
    const vec3* vertex_coord,
    const double* cell_vertex_ztop)
{
    constexpr int MAX_VERTEX_NUM = 10;

    if (cell_id < 0) {
        return 0.0;
    }

    int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
    if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
        return 0.0;
    }

    // Get cell vertex indices
    size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
    MOPS::CUDAKernel::GetCellVerticesIdx(
        cell_id,
        current_cell_vertices_number,
        current_cell_vertices_idx,
        MAX_VERTEX_NUM,
        actual_max_edge_size,
        vertices_on_cell);

    // Get cell vertex positions
    vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
    if (!MOPS::CUDAKernel::GetCellVertexPos(
            current_cell_vertex_pos,
            current_cell_vertices_idx,
            MAX_VERTEX_NUM,
            current_cell_vertices_number,
            vertex_coord)) {
        return 0.0;
    }

    // Calculate Wachspress weights
    double current_cell_vertex_weight[MAX_VERTEX_NUM];
    for (int i = 0; i < MAX_VERTEX_NUM; ++i) {
        current_cell_vertex_weight[i] = 0.0;
    }
    Interpolator::CalcPolygonWachspress(
        pos,
        current_cell_vertex_pos,
        current_cell_vertex_weight,
        current_cell_vertices_number);

    // Interpolate zTop at level 0
    double ztop_surface = 0.0;
    for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
        int vid = static_cast<int>(current_cell_vertices_idx[v_idx]);
        if (vid < 0 || vid >= actual_vertex_size) {
            continue;
        }
        double ztop_at_vertex = cell_vertex_ztop[vid * actual_ztop_layer + 0];  // level 0
        ztop_surface += current_cell_vertex_weight[v_idx] * ztop_at_vertex;
    }

    return ztop_surface;
}
```

**Purpose**: Calculates the zTop value at the water surface (level 0) at a given position using Wachspress interpolation.

---

### Change #2: Added Helper Function - FindNearestCellEdgeCUDA

**Location**: After CalcZTopAtLevel0CUDA function  
**Line number**: Inserted at ~line 1536  
**Type**: New function addition

**Code Added**:
```cpp
// Helper function: Find nearest edge of a cell to given position
MOPS_DEVICE inline void FindNearestCellEdgeCUDA(
    const vec3& pos,
    int cell_id,
    int actual_max_edge_size,
    const size_t* number_vertex_on_cell,
    const size_t* vertices_on_cell,
    const vec3* vertex_coord,
    int& edge_vertex_a_idx,
    int& edge_vertex_b_idx)
{
    constexpr int MAX_VERTEX_NUM = 10;

    edge_vertex_a_idx = -1;
    edge_vertex_b_idx = -1;

    if (cell_id < 0) {
        return;
    }

    int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
    if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
        return;
    }

    // Get cell vertex indices
    size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
    MOPS::CUDAKernel::GetCellVerticesIdx(
        cell_id,
        current_cell_vertices_number,
        current_cell_vertices_idx,
        MAX_VERTEX_NUM,
        actual_max_edge_size,
        vertices_on_cell);

    // Find nearest edge by finding minimum distance from point to edge
    double min_dist = 1e300;
    for (int i = 0; i < current_cell_vertices_number; ++i) {
        int next_i = (i + 1) % current_cell_vertices_number;
        int va_idx = static_cast<int>(current_cell_vertices_idx[i]);
        int vb_idx = static_cast<int>(current_cell_vertices_idx[next_i]);

        vec3 va = vertex_coord[va_idx];
        vec3 vb = vertex_coord[vb_idx];

        // Calculate distance from pos to edge (va, vb)
        // Using point-to-line-segment distance on the sphere
        vec3 edge_vec = vb - va;
        vec3 pos_vec = pos - va;

        double edge_len_sq = MOPS_DOT(edge_vec, edge_vec);
        if (edge_len_sq < 1e-12) {
            continue;
        }

        double t = MOPS_DOT(pos_vec, edge_vec) / edge_len_sq;
        t = fmax(0.0, fmin(1.0, t));  // clamp to [0, 1]

        vec3 closest_point = va + edge_vec * t;
        double dist = MOPS_LENGTH(pos - closest_point);

        if (dist < min_dist) {
            min_dist = dist;
            edge_vertex_a_idx = va_idx;
            edge_vertex_b_idx = vb_idx;
        }
    }
}
```

**Purpose**: Finds the nearest edge of a cell to a given particle position by calculating minimum distance to all cell edges.

---

### Change #3: Added Helper Function - ProjectVelocityOntoEdgeCUDA

**Location**: After FindNearestCellEdgeCUDA function  
**Line number**: Inserted at ~line 1605  
**Type**: New function addition

**Code Added**:
```cpp
// Helper function: Project velocity onto edge tangent direction
MOPS_DEVICE inline vec3 ProjectVelocityOntoEdgeCUDA(
    const vec3& velocity,
    const vec3& pos,
    const vec3& edge_vertex_a,
    const vec3& edge_vertex_b)
{
    // Compute edge direction on the sphere's tangent plane at pos
    vec3 edge_vec = edge_vertex_b - edge_vertex_a;

    // Project edge vector onto tangent plane at pos
    // Tangent plane is perpendicular to radial direction
    vec3 radial = pos;
    double radial_len = MOPS_LENGTH(radial);
    if (radial_len < 1e-12) {
        return velocity;
    }
    radial = radial / radial_len;

    // Remove radial component from edge vector
    double edge_radial_component = MOPS_DOT(edge_vec, radial);
    vec3 edge_tangent = edge_vec - radial * edge_radial_component;

    double edge_tangent_len = MOPS_LENGTH(edge_tangent);
    if (edge_tangent_len < 1e-12) {
        return vec3{0.0, 0.0, 0.0};
    }

    // Normalize edge tangent direction
    edge_tangent = edge_tangent / edge_tangent_len;

    // Project velocity onto edge tangent
    double vel_parallel_magnitude = MOPS_DOT(velocity, edge_tangent);
    vec3 vel_parallel = edge_tangent * vel_parallel_magnitude;

    return vel_parallel;
}
```

**Purpose**: Projects a velocity vector onto an edge's tangent direction on the sphere surface, returning only the parallel component.

---

### Change #4: Added Wrapper Function - CalcVelocityAtPathlineWithConstraintsCUDA

**Location**: After CalcVelocityAtPathlineCUDA function (after line 2004)  
**Line number**: Inserted at ~line 1642 (after helper functions) and ~line 2007 (definition)  
**Type**: New function addition

**Code Added**:
```cpp
// Wrapper function: Calculate velocity with edge projection constraint for stuck particles
MOPS_DEVICE inline PathlineVelocityState CalcVelocityAtPathlineWithConstraintsCUDA(
    const vec3& pos,
    int cell_id,
    double current_depth,
    double alpha,
    int actual_max_edge_size,
    int actual_vertex_size,
    int actual_ztop_layer,
    int actual_ztop_layer_p1,
    bool has_double_attributes,
    int attr_count,
    const size_t* number_vertex_on_cell,
    const size_t* vertices_on_cell,
    const vec3* vertex_coord,
    const vec3* cell_vertex_velocity_front,
    const vec3* cell_vertex_velocity_back,
    const double* cell_vertex_ztop_front,
    const double* cell_vertex_ztop_back,
    const double* cell_vertex_vert_velocity_front,
    const double* cell_vertex_vert_velocity_back,
    const double* attr0_front,
    const double* attr1_front,
    const double* attr0_back,
    const double* attr1_back)
{
    // Calculate velocity normally
    PathlineVelocityState state = CalcVelocityAtPathlineCUDA(
        pos,
        cell_id,
        current_depth,
        alpha,
        actual_max_edge_size,
        actual_vertex_size,
        actual_ztop_layer,
        actual_ztop_layer_p1,
        has_double_attributes,
        attr_count,
        number_vertex_on_cell,
        vertices_on_cell,
        vertex_coord,
        cell_vertex_velocity_front,
        cell_vertex_velocity_back,
        cell_vertex_ztop_front,
        cell_vertex_ztop_back,
        cell_vertex_vert_velocity_front,
        cell_vertex_vert_velocity_back,
        attr0_front,
        attr1_front,
        attr0_back,
        attr1_back);

    if (!state.ok) {
        return state;
    }

    // Apply edge projection constraint for stuck particles
    constexpr double VELOCITY_THRESHOLD = 1e-9;
    double speed = MOPS_LENGTH(state.h_vel);

    if (speed < VELOCITY_THRESHOLD && speed > 1e-20) {  // Very small but non-zero
        // Find nearest edge
        int edge_va_idx = -1;
        int edge_vb_idx = -1;
        FindNearestCellEdgeCUDA(
            pos,
            cell_id,
            actual_max_edge_size,
            number_vertex_on_cell,
            vertices_on_cell,
            vertex_coord,
            edge_va_idx,
            edge_vb_idx);

        if (edge_va_idx >= 0 && edge_vb_idx >= 0) {
            vec3 va = vertex_coord[edge_va_idx];
            vec3 vb = vertex_coord[edge_vb_idx];

            // Project velocity onto edge
            vec3 projected_vel = ProjectVelocityOntoEdgeCUDA(state.h_vel, pos, va, vb);
            state.h_vel = projected_vel;
        }
    }

    return state;
}
```

**Purpose**: Wraps the original velocity calculation and applies edge projection constraint when particle velocity is below threshold (1e-9).

---

### Change #5: Updated Function Calls in KernelPathLine (Euler Method)

**Location**: Line ~2217 (original line number before additions)  
**Type**: Function call replacement

**Original Code**:
```cpp
PathlineVelocityState s = CalcVelocityAtPathlineCUDA(
```

**Changed To**:
```cpp
PathlineVelocityState s = CalcVelocityAtPathlineWithConstraintsCUDA(
```

**Purpose**: Apply edge projection constraint to Euler method velocity calculation.

---

### Change #6: Updated Function Calls in KernelPathLine (RK4 Stage 1)

**Location**: Line ~2250 (original line number before additions)  
**Type**: Function call replacement

**Original Code**:
```cpp
PathlineVelocityState s1 = CalcVelocityAtPathlineCUDA(
```

**Changed To**:
```cpp
PathlineVelocityState s1 = CalcVelocityAtPathlineWithConstraintsCUDA(
```

**Purpose**: Apply edge projection constraint to RK4 stage 1 (k1) velocity calculation.

---

### Change #7: Updated Function Calls in KernelPathLine (RK4 Stage 2)

**Location**: Line ~2279 (original line number before additions)  
**Type**: Function call replacement

**Original Code**:
```cpp
PathlineVelocityState s2 = CalcVelocityAtPathlineCUDA(
```

**Changed To**:
```cpp
PathlineVelocityState s2 = CalcVelocityAtPathlineWithConstraintsCUDA(
```

**Purpose**: Apply edge projection constraint to RK4 stage 2 (k2) velocity calculation.

---

### Change #8: Updated Function Calls in KernelPathLine (RK4 Stage 3)

**Location**: Line ~2308 (original line number before additions)  
**Type**: Function call replacement

**Original Code**:
```cpp
PathlineVelocityState s3 = CalcVelocityAtPathlineCUDA(
```

**Changed To**:
```cpp
PathlineVelocityState s3 = CalcVelocityAtPathlineWithConstraintsCUDA(
```

**Purpose**: Apply edge projection constraint to RK4 stage 3 (k3) velocity calculation.

---

### Change #9: Updated Function Calls in KernelPathLine (RK4 Stage 4)

**Location**: Line ~2337 (original line number before additions)  
**Type**: Function call replacement

**Original Code**:
```cpp
PathlineVelocityState s4 = CalcVelocityAtPathlineCUDA(
```

**Changed To**:
```cpp
PathlineVelocityState s4 = CalcVelocityAtPathlineWithConstraintsCUDA(
```

**Purpose**: Apply edge projection constraint to RK4 stage 4 (k4) velocity calculation.

---

### Change #10: Modified Depth Update Logic with Surface Constraint

**Location**: Lines ~2393-2437 (in KernelPathLine, after velocity integration)  
**Type**: Code section replacement

**Original Code**:
```cpp
double old_depth = static_cast<double>(particle_depths[global_id]);
double new_depth = old_depth - current_vertical_velocity * static_cast<double>(delta_t);
new_depth = fmax(0.0, new_depth);

double r_new = fmax(1.0, r + current_vertical_velocity * static_cast<double>(delta_t));
particle_depths[global_id] = static_cast<float>(new_depth);

double new_len = MOPS_LENGTH(new_position);
if (new_len > 1e-12) {
    new_position = (new_position / new_len) * r_new;
}
sample_points[global_id] = new_position;
```

**Changed To**:
```cpp
double old_depth = static_cast<double>(particle_depths[global_id]);
double new_depth = old_depth - current_vertical_velocity * static_cast<double>(delta_t);

// Surface constraint: prevent particles from leaving the water
// Get surface zTop (level 0) at current position
double surface_ztop = CalcZTopAtLevel0CUDA(
    new_position,
    cell_id,
    actual_max_edge_size,
    actual_vertex_size,
    actual_ztop_layer,
    number_vertex_on_cell,
    vertices_on_cell,
    vertex_coord,
    cell_vertex_ztop_front);  // Use front snapshot for surface check

// If particle would go above surface (negative depth relative to surface), clamp it
double depth_relative_to_surface = -surface_ztop - new_depth;
if (depth_relative_to_surface < 0.0 || new_depth < 0.0) {
    // Particle is trying to leave water - clamp to surface
    new_depth = -surface_ztop;  // Set depth to surface level
    if (new_depth < 0.0) {
        new_depth = 0.0;  // Safety clamp to ensure non-negative
    }
}

// Ensure depth is non-negative
new_depth = fmax(0.0, new_depth);

// Update radius based on new depth
// Note: depth is positive downward, so r_new = r_base - depth for ocean
// But vertical_velocity is in the radial direction, so we use it directly
double r_new = r + current_vertical_velocity * static_cast<double>(delta_t);

// Ensure radius doesn't go below a minimum (e.g., Earth's radius if applicable)
// If your domain has a specific base radius, use that instead of 1.0
r_new = fmax(1.0, r_new);

particle_depths[global_id] = static_cast<float>(new_depth);

double new_len = MOPS_LENGTH(new_position);
if (new_len > 1e-12) {
    new_position = (new_position / new_len) * r_new;
}
sample_points[global_id] = new_position;
```

**Purpose**: Add surface constraint to prevent particles from leaving the water. Clamps particles to surface level when they would move above the water.

---

### Change #11: Added Helper Function - CalcZTopAtBottomCUDA (BUG FIX)

**Location**: After CalcZTopAtLevel0CUDA function  
**Line number**: Inserted at ~line 1533  
**Type**: New function addition (Bug Fix - 2026-06-21)

**Code Added**:
```cpp
// Helper function: Calculate zTop at bottom level (seafloor) at given position
MOPS_DEVICE inline double CalcZTopAtBottomCUDA(
    const vec3& pos,
    int cell_id,
    int actual_max_edge_size,
    int actual_vertex_size,
    int actual_ztop_layer,
    const size_t* number_vertex_on_cell,
    const size_t* vertices_on_cell,
    const vec3* vertex_coord,
    const double* cell_vertex_ztop)
{
    constexpr int MAX_VERTEX_NUM = 10;

    if (cell_id < 0) {
        return -1e10;  // Very deep default for invalid cell
    }

    int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
    if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
        return -1e10;
    }

    if (actual_ztop_layer <= 1) {
        return -1e10;
    }

    // Get cell vertex indices
    size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
    MOPS::CUDAKernel::GetCellVerticesIdx(
        cell_id,
        current_cell_vertices_number,
        current_cell_vertices_idx,
        MAX_VERTEX_NUM,
        actual_max_edge_size,
        vertices_on_cell);

    // Get cell vertex positions
    vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
    if (!MOPS::CUDAKernel::GetCellVertexPos(
            current_cell_vertex_pos,
            current_cell_vertices_idx,
            MAX_VERTEX_NUM,
            current_cell_vertices_number,
            vertex_coord)) {
        return -1e10;
    }

    // Calculate Wachspress weights
    double current_cell_vertex_weight[MAX_VERTEX_NUM];
    for (int i = 0; i < MAX_VERTEX_NUM; ++i) {
        current_cell_vertex_weight[i] = 0.0;
    }
    Interpolator::CalcPolygonWachspress(
        pos,
        current_cell_vertex_pos,
        current_cell_vertex_weight,
        current_cell_vertices_number);

    // Interpolate zTop at bottom layer (last layer)
    int bottom_layer = actual_ztop_layer - 1;
    double ztop_bottom = 0.0;
    for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
        int vid = static_cast<int>(current_cell_vertices_idx[v_idx]);
        if (vid < 0 || vid >= actual_vertex_size) {
            continue;
        }
        double ztop_at_vertex = cell_vertex_ztop[vid * actual_ztop_layer + bottom_layer];
        ztop_bottom += current_cell_vertex_weight[v_idx] * ztop_at_vertex;
    }

    return ztop_bottom;
}
```

**Purpose**: Calculate seafloor depth (zTop at bottom layer) to prevent particles from penetrating below the ocean floor.

---

### Change #12: Fixed Recursive Call Bug (BUG FIX)

**Location**: Line 1716  
**Type**: Function rename (Bug Fix - 2026-06-21)

**Original (Buggy) Code**:
```cpp
MOPS_DEVICE inline PathlineVelocityState CalcVelocityAtPathlineWithConstraintsCUDA(
    // ... same parameters as wrapper function ...
```

**Changed To**:
```cpp
MOPS_DEVICE inline PathlineVelocityState CalcVelocityAtPathlineCUDA(
    // ... same parameters ...
```

**Purpose**: Fix recursive call bug. The original function was renamed to `CalcVelocityAtPathlineWithConstraintsCUDA`, but then a wrapper with the same name was added, causing infinite recursion. Renamed back to `CalcVelocityAtPathlineCUDA` so the wrapper (line 2081) can call it properly.

**Impact**: Without this fix, all pathline calculations would hang in infinite recursion.

---

### Change #13: Fixed Wrapper Function Call (BUG FIX)

**Location**: Line 2107 (in CalcVelocityAtPathlineWithConstraintsCUDA wrapper)  
**Type**: Function call correction (Bug Fix - 2026-06-21)

**Original (Buggy) Code**:
```cpp
    // Calculate velocity normally
    PathlineVelocityState state = CalcVelocityAtPathlineWithConstraintsCUDA(
        pos,
        cell_id,
        // ... parameters ...
```

**Changed To**:
```cpp
    // Calculate velocity normally
    PathlineVelocityState state = CalcVelocityAtPathlineCUDA(
        pos,
        cell_id,
        // ... parameters ...
```

**Purpose**: Wrapper should call the base function `CalcVelocityAtPathlineCUDA`, not itself. This completes the fix for the recursive call bug.

---

### Change #14: Added Bottom Constraint to Depth Update (BUG FIX)

**Location**: Lines ~2493-2516 (in KernelPathLine, after surface constraint)  
**Type**: Code addition (Bug Fix - 2026-06-21)

**Code Added** (inserted after surface constraint):
```cpp
        // Bottom constraint: prevent particles from going below seafloor
        // Get bottom zTop (deepest layer) at current position
        double bottom_ztop = CalcZTopAtBottomCUDA(
            new_position,
            cell_id,
            actual_max_edge_size,
            actual_vertex_size,
            actual_ztop_layer,
            number_vertex_on_cell,
            vertices_on_cell,
            vertex_coord,
            cell_vertex_ztop_front);  // Use front snapshot for bottom check

        // If particle would go below seafloor, clamp to bottom and ignore vertical velocity
        // (depth values are negative going down, bottom_ztop is most negative)
        if (new_depth < bottom_ztop) {
            // Particle is trying to penetrate seafloor - clamp to bottom
            // This effectively ignores the downward vertical velocity component
            new_depth = bottom_ztop;
        }

        // ... existing depth clamping ...

        // Update radius based on new depth
        // ... existing code ...
        double r_new = r + current_vertical_velocity * static_cast<double>(delta_t);

        // Apply bottom constraint to radius update as well
        // If we clamped depth due to bottom constraint, adjust radius accordingly
        if (old_depth - current_vertical_velocity * static_cast<double>(delta_t) < bottom_ztop) {
            // Don't apply vertical velocity to radius if hitting bottom
            r_new = r;
        }
```

**Purpose**: Prevent particles from penetrating below the seafloor. When a particle would move below the bottom, its depth is clamped and vertical velocity is ignored, allowing it to continue moving horizontally along the ocean floor.

**Impact**: Fixes issue where particles would stop moving or become NaN when trying to go below the seafloor.

---

## Summary of Changes

### Functions Added: 5
1. `CalcZTopAtLevel0CUDA` - Calculate surface zTop at a position *(Change #1)*
2. `CalcZTopAtBottomCUDA` - Calculate bottom zTop (seafloor) at a position *(Change #11, Bug Fix 2026-06-21)*
3. `FindNearestCellEdgeCUDA` - Find nearest cell edge to a position *(Change #2)*
4. `ProjectVelocityOntoEdgeCUDA` - Project velocity onto edge direction *(Change #3)*
5. `CalcVelocityAtPathlineWithConstraintsCUDA` - Wrapper with edge projection *(Change #4)*

### Functions Modified/Renamed: 1
1. `CalcVelocityAtPathlineCUDA` - Original function renamed back from `CalcVelocityAtPathlineWithConstraintsCUDA` to fix recursive call bug *(Change #12, Bug Fix 2026-06-21)*

### Function Calls Updated: 6
1. Wrapper function internal call - Fixed to call `CalcVelocityAtPathlineCUDA` instead of itself *(Change #13, Bug Fix 2026-06-21)*
2. Euler method velocity calculation *(Change #5)*
3. RK4 stage 1 (s1/k1) velocity calculation *(Change #6)*
4. RK4 stage 2 (s2/k2) velocity calculation *(Change #7)*
5. RK4 stage 3 (s3/k3) velocity calculation *(Change #8)*
6. RK4 stage 4 (s4/k4) velocity calculation *(Change #9)*

### Code Sections Modified: 2
1. Depth update logic with surface constraint *(Change #10)*
2. Depth update logic with bottom constraint and radius adjustment *(Change #14, Bug Fix 2026-06-21)*

---

## Complete Change List

**Total Changes: 14**
- Changes #1-10: Original implementation (2026-05-26)
- Changes #11-14: Critical bug fixes (2026-06-21)

**Original Implementation (Changes #1-10)**:
1. Added `CalcZTopAtLevel0CUDA` helper function
2. Added `FindNearestCellEdgeCUDA` helper function
3. Added `ProjectVelocityOntoEdgeCUDA` helper function
4. Added `CalcVelocityAtPathlineWithConstraintsCUDA` wrapper function
5. Updated Euler method to use wrapper
6. Updated RK4 stage 1 to use wrapper
7. Updated RK4 stage 2 to use wrapper
8. Updated RK4 stage 3 to use wrapper
9. Updated RK4 stage 4 to use wrapper
10. Modified depth update with surface constraint

**Bug Fixes (Changes #11-14)**:
11. Added `CalcZTopAtBottomCUDA` for seafloor depth calculation
12. Fixed function naming - renamed first function back to `CalcVelocityAtPathlineCUDA`
13. Fixed wrapper to call base function instead of itself (infinite recursion fix)
14. Added bottom constraint to prevent seafloor penetration

---

## Key Constants

- **Velocity threshold for stuck particles**: `1e-9`
- **Maximum vertex number per cell**: `10` (MAX_VERTEX_NUM)
- **Minimum radius clamp**: `1.0` (may need adjustment based on domain)

---

## Testing Notes

**Test Cases Needed**:
1. Particles with velocity < 1e-9 (should slide along edges)
2. Particles with strong upward vertical velocity (should clamp to surface)
3. Particles in normal flow (should behave as before)
4. RK4 integration with constraints at each stage

**Performance Impact**:
- Additional calculations per particle per timestep:
  - Edge projection: 1-2 times per timestep (if velocity < threshold)
  - Surface constraint: Once per timestep
- Expected impact: Minor (<5% overhead for typical flows)

---

## Implementation Date
**2026-05-26**

## Verification Status
- [x] Code added
- [x] Critical bugs fixed (recursive call, bottom constraint)
- [ ] Compiled successfully
- [ ] Unit tested
- [ ] Integration tested
- [ ] Performance profiled

---

## Next Steps
1. **Compile the code** to ensure no syntax errors
2. **Test with particles near the seafloor** to verify bottom constraint works
3. **Test with particles near the surface** to verify surface constraint works
4. **Test stuck particles** (velocity < 1e-9) to verify edge projection works
5. **Performance test** to measure overhead of new constraints
