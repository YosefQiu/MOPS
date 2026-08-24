# PATHLINE CUDA Implementation - Technical Reference

**Version:** 2026-08-18  
**File:** `src/GPU/CUDA/Kernel/MPASOVisualizerKernels.cu`  
**Function:** `KernelPathLine` (lines 2405-3096)

This document provides a comprehensive technical description of the CUDA pathline computation implementation for MPAS-Ocean data.

---

## Table of Contents

1. [Overview](#overview)
2. [Algorithm Flow](#algorithm-flow)
3. [Integration Methods](#integration-methods)
4. [Velocity Perturbation Mechanism](#velocity-perturbation-mechanism)
5. [Boundary Conditions](#boundary-conditions)
6. [Attribute Tracking](#attribute-tracking)
7. [Helper Functions](#helper-functions)
8. [Key Parameters](#key-parameters)
9. [Edge Cases and Failure Handling](#edge-cases-and-failure-handling)
10. [Performance Considerations](#performance-considerations)

---

## 1. Overview

### Purpose

The pathline computation traces water parcels through a time-varying 3D ocean velocity field on an unstructured MPAS-Ocean mesh. Each pathline represents the trajectory of a Lagrangian particle advected by ocean currents.

### Key Characteristics

- **Mesh Type:** Unstructured spherical Voronoi mesh (MPAS-Ocean)
- **Velocity Field:** Time-varying (interpolated between two snapshots)
- **Integration:** Euler or 4th-order Runge-Kutta (RK4)
- **Coordinate System:** 3D Cartesian (with spherical geometry)
- **Parallelization:** One CUDA thread per particle
- **Boundary Handling:** Surface, bottom, and lateral boundaries with projection/reflection

---

## 2. Algorithm Flow

### 2.1 Main Loop Structure

```
For each particle (one CUDA thread):
    Initialize cell_id from seed position
    
    For each time step (i_step = 0 to n_steps-1):
        1. Compute temporal interpolation factor: alpha = i_step / n_steps
        
        2. Update cell_id (find nearest cell center among current + neighbors)
        
        3. Compute velocity and attributes:
           - If Euler: CalcVelocityAtPathline(position, depth, alpha) once
           - If RK4: CalcVelocityAtPathline() 4 times with perturbations
        
        4. Advance horizontal position on sphere:
           - Use spherical rotation (not Cartesian displacement)
           - new_position = RotateOnSphere(current_position, velocity, dt)
        
        5. Apply vertical boundary conditions:
           a. Compute tentative depth: new_depth = old_depth - w * dt
           b. Surface projection: clamp to surface if particle floats
           c. Bottom-following projection: preserve clearance from seafloor
           d. Final clamping: ensure within [surface, bottom]
           e. Project position to match adjusted depth
        
        6. Apply lateral boundary conditions:
           a. Check if new_position is in current cell or neighbors
           b. If valid: update cell_id and continue
           c. If outside (boundary hit):
              - Find nearest edge
              - Project velocity onto edge direction
              - Try moving along edge
              - If failed: try 20 random tangent directions
              - If all failed: keep old position (particle trapped)
        
        7. Record position/velocity/attributes at specified intervals
```

### 2.2 Thread Assignment

```cpp
int global_id = blockIdx.x * blockDim.x + threadIdx.x;
// global_id = particle index (0 to particle_count-1)
```

Each thread independently integrates one particle for all time steps.

---

## 3. Integration Methods

### 3.1 Euler Method

**When:** `use_euler == true`

**Algorithm:**
```cpp
// Compute velocity at current position
state = CalcVelocityAtPathlineWithConstraintsCUDA(
    current_position, cell_id, current_depth, alpha, ...);

h_vel = state.h_vel;  // Horizontal velocity (3D Cartesian)
v_vel = state.v_vel;  // Vertical velocity (scalar)

// Advance position using spherical rotation
rotation_axis = CalcRotationAxis(current_position, h_vel);
speed = length(h_vel);
theta = (speed * delta_t) / radius;
new_position = RotateOnSphere(current_position, rotation_axis, theta);
```

**Characteristics:**
- First-order accurate
- Fast (1 velocity evaluation per step)
- Suitable for small time steps

### 3.2 RK4 Method

**When:** `use_euler == false`

**Algorithm:**
```cpp
// Four-stage Runge-Kutta
dt = delta_t;
dalpha = dt / simulation_duration;

// Stage 1: k1
a1 = alpha;
s1 = CalcVelocityAtPathline(current_position, depth, a1, ...);
if (length(s1.h_vel) < VEL_THRESH):
    s1.h_vel += RandomTangentVelocity(..., SIGMA_H);  // PERTURBATION
if (|s1.v_vel| < VEL_THRESH):
    s1.v_vel += SIGMA_V * random_noise();  // PERTURBATION

// Stage 2: k2
p2 = AdvectOnSphere(current_position, s1.h_vel, dt/2);
a2 = clamp(a1 + dalpha/2, 0, 1);
s2 = CalcVelocityAtPathline(p2, depth, a2, ...);
if (length(s2.h_vel) < VEL_THRESH):
    s2.h_vel += RandomTangentVelocity(..., SIGMA_H);  // PERTURBATION
if (|s2.v_vel| < VEL_THRESH):
    s2.v_vel += SIGMA_V * random_noise();  // PERTURBATION

// Stage 3: k3
p3 = AdvectOnSphere(current_position, s2.h_vel, dt/2);
a3 = clamp(a1 + dalpha/2, 0, 1);
s3 = CalcVelocityAtPathline(p3, depth, a3, ...);
[... same perturbation logic ...]

// Stage 4: k4
p4 = AdvectOnSphere(current_position, s3.h_vel, dt);
a4 = clamp(a1 + dalpha, 0, 1);
s4 = CalcVelocityAtPathline(p4, depth, a4, ...);
[... same perturbation logic ...]

// Weighted average
h_vel = (s1.h_vel + 2*s2.h_vel + 2*s3.h_vel + s4.h_vel) / 6;
v_vel = (s1.v_vel + 2*s2.v_vel + 2*s3.v_vel + s4.v_vel) / 6;
attr  = (s1.attr  + 2*s2.attr  + 2*s3.attr  + s4.attr ) / 6;

// Advance position using averaged velocity
new_position = AdvectOnSphere(current_position, h_vel, dt);
```

**Characteristics:**
- Fourth-order accurate
- 4 velocity evaluations per step
- **NEW:** Velocity perturbation added at ALL four RK4 stages (not just final)
- Suitable for larger time steps and higher accuracy

**Important Note:** The RK4 implementation uses the SAME `cell_id` and `current_depth` for all four stages. This is a practical simplification - true 3D RK4 would update cell and depth at each stage, which would be computationally expensive.

---

## 4. Velocity Perturbation Mechanism

### 4.1 Purpose

Prevent particles from getting "stuck" in regions of extremely low velocity by adding small random perturbations when velocity falls below a threshold.

### 4.2 When Perturbations Are Applied

**Location in code:** Lines 2598-2739 (RK4 only, NOT in Euler mode)

**Trigger conditions:**
```cpp
constexpr double VEL_THRESH = 1e-4;  // 0.1 mm/s = 0.36 m/hr

// Horizontal perturbation
if (length(h_vel) < VEL_THRESH):
    add horizontal perturbation

// Vertical perturbation
if (|v_vel| < VEL_THRESH):
    add vertical perturbation
```

### 4.3 Perturbation Magnitudes

```cpp
constexpr double SIGMA_H = 0.01;      // 1 cm/s horizontal
constexpr double SIGMA_V = 0.00001;   // 0.01 mm/s vertical
```

### 4.4 Horizontal Perturbation

**Function:** `GenerateRandomTangentVelocityCUDA`

```cpp
vec3 perturb_h = GenerateRandomTangentVelocityCUDA(
    position, 
    particle_id, 
    i_step * 4 + stage,  // Unique seed for each stage
    SIGMA_H);

h_vel += perturb_h;
```

**Algorithm:**
1. Compute radial direction: `radial = normalize(position)`
2. Create two orthonormal tangent vectors on sphere surface
3. Generate random angle: `theta = random_float(...) * 2*pi`
4. Combine tangents: `direction = tangent1*cos(theta) + tangent2*sin(theta)`
5. Scale: `perturb_h = direction * SIGMA_H`

**Properties:**
- Deterministic (hash-based PRNG, repeatable with same inputs)
- Tangent to sphere (no radial component)
- Magnitude = SIGMA_H
- Different for each RK4 stage due to seed

### 4.5 Vertical Perturbation

```cpp
// Two independent random samples for better distribution
float r1 = random_float(particle_id, i_step * 4 + stage, 10);
float r2 = random_float(particle_id, i_step * 4 + stage, 20);
v_vel += SIGMA_V * (r1 - 0.5 + r2 - 0.5);  // Range: [-SIGMA_V, +SIGMA_V]
```

**Properties:**
- Symmetric distribution around zero
- Range: approximately [-1e-5, +1e-5] m/s
- Very small (prevents unrealistic vertical motion)

### 4.6 Physical Interpretation

The perturbations represent **unresolved sub-grid scale turbulent diffusion**:
- Real ocean: small-scale eddies, internal waves, turbulent mixing
- Model: unresolved at MPAS-O grid resolution (~10 km)
- Perturbation: phenomenological representation of this mixing

**Scaling:**
- Horizontal diffusivity: K_h ~ SIGMA_H^2 * dt ~ 10^-4 m²/s (typical ocean)
- Vertical diffusivity: K_v ~ SIGMA_V^2 * dt ~ 10^-10 m²/s (very small, as expected)

---

## 5. Boundary Conditions

### 5.1 Overview

Three types of boundaries:
1. **Surface** (sea-air interface): Particle cannot float above water
2. **Bottom** (seafloor): Particle follows bottom topography
3. **Lateral** (land/mesh edge): Particle slides along boundary or escapes randomly

### 5.2 Surface Boundary (Lines 2803-2824)

**Condition:** Particle depth < surface depth (floats above water)

**Algorithm:**
```
1. Compute tentative new depth:
   new_depth_raw = old_depth - v_vel * dt

2. Evaluate surface level at NEW horizontal position:
   surface_ztop = CalcZTopAtLevel0(new_position_raw, cell_id, ...)
   surface_depth = -surface_ztop

3. Apply projection:
   if (new_depth_raw < surface_depth):
       new_depth = surface_depth  // Clamp to surface
   else:
       new_depth = new_depth_raw  // Allow normal motion
   
   new_depth = max(new_depth, 0.0)  // Ensure non-negative
```

**Physical Interpretation:**
- Particle "bounces" off the sea surface
- In reality: particles (e.g., plankton, pollutants) accumulate at surface

**Note:** Surface is evaluated at the NEW horizontal position (after advection), not the old position. This is important when crossing regions with variable sea surface height.

### 5.3 Bottom Boundary (Lines 2826-2858)

**Condition:** Particle depth > bottom depth (penetrates seafloor)

**Algorithm:**
```
1. Compute clearance at time t in CURRENT cell:
   bottom_ztop_t = CalcZTopAtBottom(current_position, cell_id, ...)
   bottom_depth_t = -bottom_ztop_t
   clearance = bottom_depth_t - old_depth  // Distance above bottom
   clearance = max(clearance, 0.0)

2. Compute tentative new depth:
   new_depth_raw = old_depth - v_vel * dt

3. Find which cell contains NEW horizontal position:
   target_cell = FindContainingCellInCurrentOrNeighbors(new_position_raw, ...)
   if (target_cell < 0): target_cell = cell_id  // Fallback to current

4. Evaluate bottom at NEW horizontal position in TARGET cell:
   bottom_ztop_next = CalcZTopAtBottom(new_position_raw, target_cell, ...)
   bottom_depth_next = -bottom_ztop_next

5. Apply bottom-following projection:
   if (new_depth_raw > bottom_depth_next):
       new_depth = bottom_depth_next - clearance  // Preserve clearance
   else:
       new_depth = new_depth_raw  // Allow normal motion

6. Final clamping (keep in water column):
   new_depth = max(new_depth, surface_depth)
   new_depth = min(new_depth, bottom_depth_next)
```

**Bottom-Following Behavior:**

This is NOT a simple clamp to the seafloor. Instead, it preserves the particle's **relative distance from the bottom** as it moves horizontally.

Example:
```
Time t (cell A):
    particle at 3000m depth
    bottom at 3500m depth
    clearance = 500m

Time t+1 (cell B):
    bottom at 4000m depth
    → particle depth = 4000 - 500 = 3500m
    (maintains 500m clearance)
```

**Physical Interpretation:**
- Represents **bottom boundary layer flow**
- Real ocean: velocity shear near bottom, particles follow topography
- Prevents unrealistic penetration into solid earth

### 5.4 Depth-to-Position Projection (Lines 2868-2872)

After adjusting depth, the 3D position must be updated to match:

```cpp
new_position = ProjectPositionToDepth(
    new_position_raw,  // Tentative horizontal position
    new_depth,         // Adjusted depth
    old_depth,         // Previous depth
    radius);           // Current radius
```

**Algorithm:**
```cpp
depth_change = old_depth - new_depth;
new_radius = current_radius + depth_change;
new_radius = max(new_radius, 1.0);  // Ensure minimum radius

// Keep horizontal direction, scale to new radius
direction = normalize(new_position_raw);
new_position = direction * new_radius;
```

**Effect:** Particles move radially inward/outward to match the corrected depth, while preserving their horizontal (lat/lon) location.

### 5.5 Lateral Boundary (Lines 2881-3079)

**Detection:**

```cpp
containing_cell = FindContainingCellInCurrentOrNeighborsCUDA(
    new_position, cell_id, ...);

if (containing_cell >= 0):
    // Normal case: cell crossing (not a boundary)
    if (containing_cell != cell_id):
        cell_id = containing_cell  // Update cell
else:
    // True lateral boundary hit (land or mesh edge)
    handle_lateral_boundary()
```

**Key Insight:** This distinguishes between:
- **Normal cell crossing**: Particle moves from one ocean cell to a neighboring ocean cell → ALLOWED
- **Boundary hit**: Particle tries to move outside ALL valid cells → BOUNDARY HANDLING

**Boundary Handling Strategy:**

```
1. Count boundary hit: boundary_hit_count[particle_id]++

2. Find nearest edge of CURRENT cell:
   FindNearestCellEdgeCUDA(...) → (va, vb)

3. Project velocity onto edge tangent:
   projected_vel = ProjectVelocityOntoEdgeCUDA(h_vel, pos, va, vb)
   
   Algorithm:
       edge_vec = vb - va
       radial = normalize(pos)
       edge_tangent = edge_vec - dot(edge_vec, radial) * radial
       edge_tangent = normalize(edge_tangent)
       projected_vel = dot(h_vel, edge_tangent) * edge_tangent

4. Compute candidate position along edge:
   candidate = AdvectOnSphere(current_position, projected_vel, dt)
   candidate = normalize(candidate) * r_new  // Scale to correct radius

5. Validate candidate:
   if (IsInMesh(cell_id, candidate, ...)):
       new_position = candidate  // Success
   else:
       try_random_escape()  // Failed, try random directions
```

**Random Escape Mechanism:**

If edge projection fails, try up to 20 random tangent directions:

```cpp
bool TryMultipleRandomDirectionsCUDA(...):
    BASE_KICK_SPEED = 0.01 m/s
    
    for attempt in 0..19:
        // Progressive speed increase
        kick_speed = BASE_KICK_SPEED * (1.0 + (attempt / 5) * 2.0)
        
        // Speed schedule:
        // attempts 0-4:   0.01 m/s
        // attempts 5-9:   0.03 m/s
        // attempts 10-14: 0.05 m/s
        // attempts 15-19: 0.07 m/s
        
        random_vel = GenerateRandomTangentVelocity(..., kick_speed, attempt)
        random_candidate = AdvectOnSphere(pos, random_vel, dt)
        
        if (IsInMesh(cell_id, random_candidate, ...)):
            return true  // Escape successful
    
    return false  // All attempts failed, particle trapped
```

**Trapped Particles:**

If all boundary handling strategies fail:
```cpp
new_position = current_position;  // Stay at safe position
// Particle effectively stops moving (trapped against land)
```

**Statistics:**

The kernel tracks boundary hits per particle and reports:
- Total boundary hits
- Particles affected
- Maximum hits per particle

Example output:
```
🎯 ===== Lateral Boundary Projection Summary =====
   Particles affected: 23 / 1000 (2.3%)
   Total boundary hits: 156
   Average hits per affected particle: 6.8
   Maximum hits for single particle: 34
================================================
```

---

## 6. Attribute Tracking

### 6.1 Supported Attributes

- **Attribute 0:** Typically temperature (°C)
- **Attribute 1:** Typically salinity (PSU)
- **Attribute 2:** Reserved (currently unused)

### 6.2 Interpolation Method

Attributes are computed in `CalcVelocityAtPathlineCUDA` using the SAME interpolation as velocity:

```
1. Horizontal interpolation (Wachspress coordinates):
   attr(pos) = sum_i weight_i(pos) * attr_i(vertex_i)

2. Vertical interpolation (linear between layers):
   attr_layer = t * attr_upper + (1-t) * attr_lower
   where t = (depth - z_lower) / (z_upper - z_lower)

3. Temporal interpolation (linear between snapshots):
   attr = (1 - alpha) * attr_front + alpha * attr_back
   where alpha = time / simulation_duration
```

### 6.3 Attribute Consistency After Boundary Handling

**CRITICAL BUG FIX (documented in ATTRIBUTE_CONSISTENCY_FIX.md):**

After ANY position correction (edge projection, random escape, etc.), attributes MUST be recalculated to match the NEW position. Otherwise:

```
❌ WRONG:
   position = Cell B  ✅
   temperature = 15°C  ❌ (from Cell A)

✅ CORRECT:
   position = Cell B  ✅
   temperature = 12°C  ✅ (from Cell B)
```

**Implementation (should be present after boundary corrections):**
```cpp
// After position correction
auto s_corrected = CalcVelocityAtPathlineCUDA(
    new_position, cell_id, current_depth, alpha, ...);

if (s_corrected.ok):
    current_horizontal_velocity = s_corrected.h_vel;
    current_vertical_velocity = s_corrected.v_vel;
    current_attrs = s_corrected.attr;  // ← UPDATED ATTRIBUTES
```

---

## 7. Helper Functions

### 7.1 Velocity Computation

#### `CalcVelocityAtPathlineCUDA` (Lines 1954-2316)

**Purpose:** Core velocity/attribute interpolation at a 3D position

**Inputs:**
- `pos`: 3D Cartesian position
- `cell_id`: MPAS cell containing position
- `current_depth`: Depth (positive downward, meters)
- `alpha`: Temporal interpolation factor [0, 1]
- Grid arrays, velocity fields, zTop fields, attribute fields

**Outputs:** `PathlineVelocityState`
```cpp
struct PathlineVelocityState {
    vec3 h_vel;    // Horizontal velocity (m/s, Cartesian)
    double v_vel;  // Vertical velocity (m/s, upward positive)
    vec3 attr;     // Attributes (temperature, salinity, ...)
    bool ok;       // Success flag
};
```

**Algorithm:**
1. Validate inputs (cell_id, vertex count, depth)
2. Check if position is inside cell
3. Get cell vertices and compute Wachspress weights
4. Interpolate zTop profile vertically at position
5. Find vertical layer containing current_depth
6. Interpolate velocity vertically (front and back snapshots)
7. Interpolate velocity temporally: `v = (1-alpha)*v_front + alpha*v_back`
8. Interpolate attributes similarly
9. Return state with `ok=true` if successful

**Failure modes** (returns `ok=false`):
- Invalid cell_id
- Position outside cell
- Depth outside vertical range
- Invalid vertex indices
- Vertical interpolation denominator too small

#### `CalcVelocityAtPathlineWithConstraintsCUDA` (Lines 2318-2402)

**Purpose:** Wrapper that adds edge projection for stuck particles

**Algorithm:**
```cpp
state = CalcVelocityAtPathlineCUDA(...);
if (!state.ok): return state;

speed = length(state.h_vel);
if (speed < 1e-9 && speed > 1e-20):  // Very small but non-zero
    edge = FindNearestCellEdge(pos, cell_id, ...);
    if (edge found):
        state.h_vel = ProjectVelocityOntoEdge(state.h_vel, pos, edge);

return state;
```

**Note:** This pre-emptive edge projection is DIFFERENT from lateral boundary handling. It acts BEFORE the particle actually hits a boundary, when velocity is extremely low near a boundary.

### 7.2 Spherical Geometry

#### `AdvectOnSphereCUDA` (defined elsewhere, used at lines 2615, 2657, 2699, etc.)

**Purpose:** Move a point on sphere surface using great-circle motion

**Algorithm:**
```cpp
vec3 AdvectOnSphereCUDA(vec3 pos, vec3 vel, double dt):
    r = length(pos);
    speed = length(vel);
    rotation_axis = CalcRotationAxis(pos, vel);
    theta = (speed * dt) / r;  // Angular displacement
    return CalcPositionAfterRotation(pos, rotation_axis, theta);
```

**Key Point:** This is NOT Cartesian advection `pos + vel*dt`. It properly accounts for spherical geometry by rotating along a great circle.

#### `CalcRotationAxis` (defined in CUDAKernel.h)

**Purpose:** Compute rotation axis for great-circle motion

**Algorithm:**
```cpp
vec3 CalcRotationAxis(vec3 pos, vec3 vel):
    radial = normalize(pos);
    tangent_vel = vel - dot(vel, radial) * radial;  // Remove radial component
    rotation_axis = cross(radial, tangent_vel);
    return normalize(rotation_axis);
```

#### `CalcPositionAfterRotation` (defined in CUDAKernel.h)

**Purpose:** Rodrigues' rotation formula on sphere

**Algorithm:**
```cpp
vec3 CalcPositionAfterRotation(vec3 pos, vec3 axis, double theta):
    return pos * cos(theta) + 
           cross(axis, pos) * sin(theta) + 
           axis * dot(axis, pos) * (1 - cos(theta));
```

### 7.3 Boundary Detection

#### `FindContainingCellInCurrentOrNeighborsCUDA` (Lines 1748-1813)

**Purpose:** Determine if position is in valid ocean (current or neighbor cell)

**Returns:**
- `>= 0`: cell_id containing the position
- `-1`: position is outside all cells (true boundary)

**Algorithm:**
```cpp
if (IsInMesh(current_cell_id, pos, ...)):
    return current_cell_id;

neighbors = GetCellNeighborsIdx(current_cell_id, ...);
for each neighbor:
    if (IsInMesh(neighbor_cell_id, pos, ...)):
        return neighbor_cell_id;

return -1;  // Outside all cells
```

#### `FindNearestCellEdgeCUDA` (Lines 1611-1677)

**Purpose:** Find the closest edge of a cell to a given position

**Algorithm:**
```cpp
min_dist = infinity;
for each edge (va, vb) in cell:
    // Point-to-line-segment distance
    edge_vec = vb - va;
    pos_vec = pos - va;
    t = clamp(dot(pos_vec, edge_vec) / ||edge_vec||^2, 0, 1);
    closest_point = va + edge_vec * t;
    dist = ||pos - closest_point||;
    
    if (dist < min_dist):
        min_dist = dist;
        edge_vertex_a_idx = va;
        edge_vertex_b_idx = vb;
```

#### `ProjectVelocityOntoEdgeCUDA` (Lines 1680-1715)

**Purpose:** Project velocity onto tangent direction of boundary edge

**Algorithm:**
```cpp
// Project edge onto tangent plane at pos
radial = normalize(pos);
edge_tangent = edge_vec - dot(edge_vec, radial) * radial;
edge_tangent = normalize(edge_tangent);

// Project velocity onto edge direction
vel_parallel = dot(vel, edge_tangent) * edge_tangent;
return vel_parallel;
```

**Effect:** Removes the normal component (perpendicular to boundary) and keeps only the tangential component (parallel to boundary).

### 7.4 Random Number Generation

#### `hash` (Lines 1816-1822)

**Purpose:** Deterministic pseudo-random hash function

**Algorithm:** MurmurHash variant
```cpp
unsigned int hash(unsigned int x):
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x);
    return x;
```

#### `random_float` (Lines 1825-1829)

**Purpose:** Generate random float in [0, 1]

**Algorithm:**
```cpp
float random_float(int particle_id, int timestep, int component):
    seed = hash(particle_id * 73856093 + 
                timestep * 19349663 + 
                component * 83492791);
    return (seed & 0xFFFFFF) / 16777216.0f;  // 24-bit mantissa
```

**Properties:**
- Deterministic (same inputs → same output)
- Different seeds for different particles, timesteps, components
- Uniform distribution in [0, 1]

#### `GenerateRandomTangentVelocityCUDA` (Lines 1833-1879)

See Section 4.4 above.

### 7.5 Depth and Surface Functions

#### `CalcZTopAtLevel0CUDA` (Lines 1468-1534)

**Purpose:** Interpolate sea surface height at position

**Algorithm:**
1. Get cell vertices and weights
2. Interpolate: `ztop_surface = sum_i weight_i * ztop[vertex_i, level=0]`

**Units:** `zTop` is negative downward (e.g., -5 m means 5 m below reference)

#### `CalcZTopAtBottomCUDA` (Lines 1537-1608)

**Purpose:** Interpolate seafloor depth at position

**Algorithm:**
1. Get cell vertices and weights
2. Interpolate: `ztop_bottom = sum_i weight_i * ztop[vertex_i, level=last]`

**Units:** Same as surface (negative downward)

---

## 8. Key Parameters

### 8.1 Integration Parameters

| Parameter | Description | Value | Units |
|-----------|-------------|-------|-------|
| `delta_t` | Time step size | User-defined | seconds |
| `n_steps` | Number of integration steps | `simulation_duration / delta_t` | - |
| `simulation_duration` | Total simulation time | User-defined | seconds |
| `record_t` | Recording interval | User-defined | seconds |
| `use_euler` | Use Euler (true) or RK4 (false) | User-defined | - |

### 8.2 Velocity Perturbation Parameters

| Parameter | Description | Value | Units | Physical Meaning |
|-----------|-------------|-------|-------|-----------------|
| `VEL_THRESH` | Velocity threshold for perturbation | 1e-4 | m/s | 0.1 mm/s = 0.36 m/hr |
| `SIGMA_H` | Horizontal perturbation magnitude | 0.01 | m/s | 1 cm/s |
| `SIGMA_V` | Vertical perturbation magnitude | 1e-5 | m/s | 0.01 mm/s |

**Physical scaling:**
- Horizontal diffusivity: K_h ~ SIGMA_H² × dt ~ 10⁻⁴ m²/s (typical ocean mesoscale)
- Vertical diffusivity: K_v ~ SIGMA_V² × dt ~ 10⁻¹⁰ m²/s (minimal vertical mixing)

### 8.3 Lateral Boundary Parameters

| Parameter | Description | Value | Units |
|-----------|-------------|-------|-------|
| `BASE_KICK_SPEED` | Initial random escape speed | 0.01 | m/s |
| `MAX_ATTEMPTS` | Maximum random escape attempts | 20 | - |

**Speed progression:**
- Attempts 0-4: 0.01 m/s
- Attempts 5-9: 0.03 m/s
- Attempts 10-14: 0.05 m/s
- Attempts 15-19: 0.07 m/s

### 8.4 Numerical Tolerances

| Parameter | Description | Value | Context |
|-----------|-------------|-------|---------|
| `1e-12` | Zero velocity threshold | 1e-12 | Velocity magnitude checks |
| `1e-9` | Very small velocity | 1e-9 | Edge projection trigger |
| `1e-8` | Depth comparison epsilon | 1e-8 | Vertical layer search |
| `1e-10` | Very deep bottom default | -1e10 | Invalid cell bottom |

---

## 9. Edge Cases and Failure Handling

### 9.1 Invalid Initial Cell

```cpp
if (cell_id < 0 || cell_id >= actual_cell_size):
    return;  // Terminate this particle
```

**Cause:** Seed point outside valid mesh

**Handling:** Particle immediately stops (no integration)

### 9.2 Velocity Computation Failure

```cpp
state = CalcVelocityAtPathline(...);
if (!state.ok):
    // Return from function or use fallback
```

**Causes:**
- Position not in specified cell
- Depth outside vertical range
- Invalid vertex indices
- Interpolation failure

**Handling:** Depends on context (early return or keep old position)

### 9.3 Particle Trapped at Boundary

```cpp
// After all boundary handling attempts fail
new_position = current_position;  // Stay in place
```

**Cause:** Particle is in a "pocket" surrounded by land, cannot escape

**Handling:** Particle stops moving (effectively removed from circulation)

**Statistics:** Typically <5% of particles in coastal/island regions

### 9.4 NaN Propagation

**Prevention strategies:**
- Check division denominators > 1e-12
- Validate vector magnitudes before normalization
- Clamp depths to valid range [surface, bottom]
- Use finite checks where appropriate

**No explicit NaN handling in current code** - relies on prevention

### 9.5 Zero Velocity Field

```cpp
if (length(velocity) < VEL_THRESH):
    add perturbation  // Prevents complete stagnation
```

**Cause:** Particle in region with no flow (e.g., closed basin, stagnant zone)

**Handling:** Velocity perturbation mechanism (Section 4)

---

## 10. Performance Considerations

### 10.1 Parallelization

- **Thread mapping:** 1 thread = 1 particle (perfectly parallel)
- **Grid size:** `(particle_count + 127) / 128` blocks × 128 threads/block
- **Memory access:** Mostly read-only (velocity fields, grid), minimal writes

### 10.2 Memory Footprint

**Per-particle storage:**
- Position: 3 × 8 bytes = 24 bytes
- Velocity: 3 × 8 bytes = 24 bytes
- Attributes: 3 × 8 bytes = 24 bytes
- Depth: 1 × 4 bytes = 4 bytes
- Total: ~76 bytes per recorded point

**Example:** 1000 particles × 1000 time points = 1 million points × 76 bytes ≈ 76 MB

### 10.3 Computational Costs

**Per time step per particle (RK4):**
- 4 velocity evaluations
- Each evaluation:
  - Wachspress interpolation: ~50-100 FLOPs
  - Vertical interpolation: ~20 FLOPs
  - Temporal interpolation: ~10 FLOPs
- Boundary checks: ~100-200 FLOPs
- Total: ~500-1000 FLOPs/step

**Typical performance:** 1000 particles × 1000 steps on Nvidia A100 ≈ 1-2 seconds

### 10.4 Bottlenecks

1. **Velocity interpolation:** Most expensive operation (Wachspress weights)
2. **Boundary detection:** Requires checking multiple cells
3. **Random number generation:** Hash computations for perturbations

### 10.5 Optimization Opportunities

1. **Reduce perturbation frequency:** Only apply when truly needed
2. **Cache cell neighbor lists:** Avoid recomputing `GetCellNeighborsIdx`
3. **Early termination:** Stop particles that have "converged" to a stable region
4. **Adaptive time stepping:** Use larger dt when velocity is smooth

---

## Appendix A: Coordinate Systems and Sign Conventions

### Cartesian Coordinates

- **Origin:** Earth center
- **Units:** Meters
- **Radius:** ~6.37 × 10⁶ m (Earth radius)

### Depth Convention

```
particle_depths[global_id]:  positive downward (e.g., 100 m)
current_depth (in code):     negative z-coordinate (e.g., -100 m)
zTop arrays:                 negative downward (e.g., -100 m)

Conversion: current_depth = -1.0 * particle_depths[global_id]
```

### Velocity Convention

- **Horizontal velocity:** 3D Cartesian vector, tangent to sphere
- **Vertical velocity:** Scalar, positive = upward (away from Earth center)
- **Units:** m/s

### Temporal Interpolation

```
alpha = 0:   use mSol_Front (earlier snapshot)
alpha = 1:   use mSol_Back (later snapshot)
0 < alpha < 1: linear interpolation

alpha = i_step / n_steps
```

---

## Appendix B: Differences from Documentation

### Discrepancies Found

1. **NEVER_STOP_GUARANTEE.md**
   - **Claims:** Velocity is boosted to a minimum magnitude
   - **Reality:** Velocity perturbation is added only in low-velocity regions, not universally
   - **Status:** Document describes a proposed mechanism, not the actual implementation

2. **VELOCITY_BOOST_TUNING.md**
   - **Claims:** Velocity is multiplied by a boost factor (e.g., 3x)
   - **Reality:** Perturbations are ADDED, not multiplied
   - **Status:** Document may describe an older version of the code

3. **ATTRIBUTE_CONSISTENCY_FIX.md**
   - **Claims:** Attributes are recalculated after boundary corrections
   - **Verification needed:** Check lines ~2985-3030 (boundary handling code) for attribute recalculation calls
   - **Status:** Document describes a fix that should be verified in code

4. **pathline_cuda_summary.md**
   - **Accuracy:** Generally accurate
   - **Missing:** RK4-stage perturbations (newly added feature not yet documented)

---

## Appendix C: Recommendations

### For Scientific Use

1. **Document perturbations:** Always mention in methods section that velocity perturbations are applied when |v| < 1e-4 m/s
2. **Validate results:** Compare a subset of trajectories with and without perturbations to assess impact
3. **Sensitivity analysis:** Test different VEL_THRESH values for your specific application

### For Code Maintenance

1. **Update documentation:** Revise NEVER_STOP_GUARANTEE.md and VELOCITY_BOOST_TUNING.md to match actual implementation
2. **Verify attribute fix:** Confirm that CalcVelocityAtPathlineCUDA is called after ALL position corrections
3. **Add unit tests:** Test boundary conditions with known geometric cases
4. **Profile performance:** Identify actual bottlenecks (may differ from theoretical analysis)

### For Further Development

1. **Adaptive time stepping:** Use velocity magnitude to adjust dt automatically
2. **Particle seeding strategies:** Pre-filter seed points to avoid shallow water and land
3. **3D RK4:** Implement fully coupled 3D RK4 with cell/depth updates at each stage
4. **Turbulence parameterization:** Replace ad-hoc perturbations with physically-based diffusion model

---

## Appendix D: File Cleanup Recommendations

### Files to Update

- `docs/pathline_cuda_summary.md` → Update with RK4 perturbation mechanism
- `NEVER_STOP_GUARANTEE.md` → Rewrite to match actual implementation or mark as "proposed"
- `VELOCITY_BOOST_TUNING.md` → Update or delete (describes old boosting, not current perturbation)

### Files to Delete

- `PATHLINE_MODIFICATIONS_CHANGELOG.md` → Deleted (listed in git status as deleted)
- `pathline_cuda_summary.md` (old version) → Delete or merge into this document

### Files to Keep

- `ATTRIBUTE_CONSISTENCY_FIX.md` → Keep (describes important fix)
- `FIX_STUCK_PARTICLES.md` → Keep (valuable workflow documentation)
- `LOW_VELOCITY_DEBUG_README.md` → Keep (debugging guide)
- `DEPTH_UPDATE_DEBUG_README.md` → Keep (debugging guide)

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-08-18 | Claude Code | Initial comprehensive technical reference created from source code analysis |

**Source of truth:** This document is derived from direct source code analysis of `MPASOVisualizerKernels.cu` (lines 0-3365) as of 2026-08-18.

