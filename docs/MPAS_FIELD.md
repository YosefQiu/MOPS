# MPASOField - Spatiotemporal Ocean State Interface

## Overview

`MPASOField` is a lightweight container that binds together an `MPASOGrid` (mesh topology) with one or two `MPASOSolution` objects (ocean state at specific times). It serves as the unified interface for spatiotemporal queries and particle trajectory calculations in MOPS.

The class serves three primary purposes:
1. **Binding**: Links mesh topology with time-varying ocean state
2. **Temporal Interpolation**: Supports queries between two time snapshots (for PathLine tracing)
3. **Spatial Queries**: Provides high-level methods to locate points within the mesh

This class is the primary input to `MPASOVisualizer` methods, enabling both steady-state (StreamLine) and time-varying (PathLine) particle tracing.

## Key Concepts

### Field Configurations

`MPASOField` supports two operational modes:

#### 1. Single-Solution Mode (Steady-State)
```cpp
MPASOField field;
field.initField(grid, solution_single);
```

**Use Cases**:
- **Visualization**: Rendering velocity/temperature at a single time snapshot
- **StreamLine**: Particle tracing in steady-state velocity field
- **Snapshot analysis**: Computing diagnostics at one time

**Temporal behavior**: No temporal interpolation; all queries use `mSol_Front` data.

#### 2. Dual-Solution Mode (Time-Varying)
```cpp
MPASOField field;
field.initField(grid, solution_t0, solution_t1);
```

**Use Cases**:
- **PathLine**: Time-evolving particle trajectories
- **Temporal interpolation**: Velocity at time `t = (1-α)*t0 + α*t1` where `α ∈ [0,1]`
- **Transient analysis**: Studying time-varying phenomena

**Temporal behavior**: Queries interpolate between `mSol_Front` (earlier time) and `mSol_Back` (later time).

### Spatial Queries

Field provides methods to answer:
- "Which cell contains point (x, y, z)?" → `calcInWhichCells()`
- "Is this point in the ocean (not land)?" → `isOnOcean()`

These queries combine:
- **KD-tree search** (from `MPASOGrid`) → nearest cell
- **Point-in-polygon test** (Wachspress coordinates) → verify containment
- **Land masking** (from `MPASOSolution`) → check validity

### Shared Pointers and Ownership

Field uses `std::shared_ptr` for resource management:
```cpp
std::shared_ptr<MPASOGrid> mGrid;
std::shared_ptr<MPASOSolution> mSol_Front;
std::shared_ptr<MPASOSolution> mSol_Back;  // nullptr in single-solution mode
```

**Advantages**:
- **Shared ownership**: Multiple fields can reference same grid/solution
- **Automatic cleanup**: Objects deleted when last field referencing them is destroyed
- **Memory efficiency**: Avoid copying large arrays

**Example** (multiple fields sharing grid):
```cpp
auto grid = std::make_shared<MPASOGrid>();
grid->initGrid(&reader);

auto sol1 = std::make_shared<MPASOSolution>();
sol1->initSolution(&reader);

auto sol2 = std::make_shared<MPASOSolution>();
sol2->initSolution(&reader);

MPASOField field1, field2;
field1.initField(grid, sol1);  // field1 shares grid
field2.initField(grid, sol2);  // field2 shares same grid
// Grid is deleted only when both field1 and field2 are destroyed
```

## Data Structures

### Core Components
```cpp
std::shared_ptr<MPASOGrid> mGrid;          // Mesh topology and spatial index
std::shared_ptr<MPASOSolution> mSol_Front; // Primary solution (or earlier time)
std::shared_ptr<MPASOSolution> mSol_Back;  // Optional second solution (later time)
```

**Nullability**:
- `mGrid`: **Must not be nullptr** (checked in `initField`)
- `mSol_Front`: **Must not be nullptr** (primary data source)
- `mSol_Back`: **Can be nullptr** (single-solution mode)

## Methods

### Initialization

#### `void initField(MPASOGrid* grid, MPASOSolution* sol_1, MPASOSolution* sol_2 = nullptr)`
Initialize field with raw pointers (creates shared_ptr internally).

**Parameters**:
- `grid`: Pointer to initialized `MPASOGrid`
- `sol_1`: Pointer to first solution (or only solution for steady-state)
- `sol_2`: Optional pointer to second solution (for temporal interpolation)

**Behavior**:
- Wraps raw pointers in `std::shared_ptr` (non-owning if pointer not null)
- Sets `mGrid`, `mSol_Front`, `mSol_Back`
- No validation performed (assumes inputs are valid)

**Example**:
```cpp
MPASOGrid grid;
MPASOSolution sol1, sol2;
// ... initialize grid, sol1, sol2 ...

MPASOField field;
field.initField(&grid, &sol1, &sol2);
// field now references grid, sol1, sol2
```

**Note**: Caller retains ownership of grid/solutions if allocated on stack.

#### `void initField(std::shared_ptr<MPASOGrid> grid, std::shared_ptr<MPASOSolution> sol_1, std::shared_ptr<MPASOSolution> sol_2 = nullptr)`
Initialize field with shared pointers (recommended for dynamic allocation).

**Parameters**:
- `grid`: Shared pointer to `MPASOGrid`
- `sol_1`: Shared pointer to first solution
- `sol_2`: Optional shared pointer to second solution

**Behavior**:
- Directly assigns shared pointers (increments reference count)
- Supports shared ownership between multiple fields

**Example**:
```cpp
auto grid = std::make_shared<MPASOGrid>();
grid->initGrid(&reader);

auto sol = std::make_shared<MPASOSolution>();
sol->initSolution(&reader);

MPASOField field;
field.initField(grid, sol);
// field shares ownership of grid and sol
```

**Recommended**: Use this overload when creating fields dynamically or managing lifetimes explicitly.

### Spatial Queries

#### `void calcInWhichCells(const vec3& position, int& cell_id)`
Find which cell contains a 3D Cartesian point.

**Parameters**:
- `position`: Query point in 3D Cartesian coordinates (x, y, z) on sphere
- `cell_id`: Output parameter receiving the containing cell index

**Behavior**:
1. Query KD-tree: `mGrid->searchKDT(position, cell_id)`
2. Returns nearest cell center index
3. **Does not verify** point-in-polygon (may return cell even if point is outside it)

**Returns**: Modifies `cell_id` to nearest cell index (0-based).

**Use Case**: Fast approximate lookup when exact containment not critical.

**Example**:
```cpp
vec3 point = {0.5, 0.5, 0.707};  // Some point on sphere
int cellID;
field.calcInWhichCells(point, cellID);
std::cout << "Nearest cell: " << cellID << std::endl;
```

#### `void calcInWhichCells(std::vector<CartesianCoord>& points_vec, std::vector<int>& cell_id_vec)`
Batch version: find cells for multiple points.

**Parameters**:
- `points_vec`: Vector of query points (3D Cartesian coordinates)
- `cell_id_vec`: Output vector of cell indices (must be pre-sized to `points_vec.size()`)

**Behavior**: Calls `searchKDT()` for each point independently.

**Example**:
```cpp
std::vector<vec3> particles = { /* 1000 particle positions */ };
std::vector<int> cellIDs(particles.size());
field.calcInWhichCells(particles, cellIDs);
// cellIDs[i] now contains nearest cell for particles[i]
```

**Performance**: O(N log M) where N = number of points, M = number of cells.

#### `bool isOnOcean(const vec3& position, int& cell_id, std::vector<size_t>& current_cell_vertices_idx)`
Check if a point is inside the ocean (not land) and verify polygon containment.

**Parameters**:
- `position`: Query point in 3D Cartesian coordinates
- `cell_id`: Output nearest cell index (modified)
- `current_cell_vertices_idx`: Output vertex indices of the containing cell

**Returns**:
- `true`: Point is inside an ocean cell (valid polygon containment)
- `false`: Point is on land, outside mesh, or near boundary

**Behavior**:
1. Find nearest cell: `calcInWhichCells(position, cell_id)`
2. Get cell vertices: `mGrid->getVerticesOnCell(cell_id, ...)`
3. Store vertex indices in `current_cell_vertices_idx`
4. **Verify containment**: Check if point is inside cell polygon using Wachspress coordinates
   - Sum of weights ≈ 1.0 and all weights ≥ 0 → inside
5. **Land check**: Verify cell has valid ocean data (not masked)

**Example**:
```cpp
vec3 queryPoint = {0.5, 0.5, 0.707};
int cellID;
std::vector<size_t> vertexIndices;

if (field.isOnOcean(queryPoint, cellID, vertexIndices)) {
    std::cout << "Point is in ocean cell " << cellID << std::endl;
    std::cout << "Cell has " << vertexIndices.size() << " vertices" << std::endl;
} else {
    std::cout << "Point is on land or outside mesh" << std::endl;
}
```

**Use Case**:
- Particle initialization: Verify seed points are in water
- Boundary detection: Check if trajectory has exited ocean domain
- Land masking: Filter visualization pixels

## Usage Examples

### Single-Solution Field (Steady-State Visualization)

```cpp
#include "Core/MPASOField.h"
#include "Core/MPASOGrid.h"
#include "Core/MPASOSolution.h"
#include "IO/MPASOReader.h"

// Initialize grid
MPASOReader reader("data/ocean.nc");
MPASOGrid grid;
grid.initGrid(&reader);
grid.createKDTree("cache/kdtree.kdt", sycl_queue);

// Load single solution
reader.setTimestep(0);
MPASOSolution solution;
solution.initSolution(&reader);
solution.calcCellCenterVelocity(&grid, "cache/", ctx);
solution.calcCellVertexVelocity(&grid, "cache/", ctx);

// Create field
MPASOField field;
field.initField(&grid, &solution);

// Use field for visualization
MPASOVisualizer viz;
viz.initVisualizer(&field, ctx);
viz.VisualizeFixedDepth(/* parameters */);
```

### Dual-Solution Field (Time-Varying PathLine)

```cpp
// Load two consecutive timesteps
reader.setTimestep(10);
MPASOSolution solution_t0;
solution_t0.initSolution(&reader);
solution_t0.calcCellCenterVelocity(&grid, "cache/", ctx);
solution_t0.calcCellVertexVelocity(&grid, "cache/", ctx);
solution_t0.calcCellVertexZtop(&grid, "cache/", ctx);

reader.setTimestep(11);
MPASOSolution solution_t1;
solution_t1.initSolution(&reader);
solution_t1.calcCellCenterVelocity(&grid, "cache/", ctx);
solution_t1.calcCellVertexVelocity(&grid, "cache/", ctx);
solution_t1.calcCellVertexZtop(&grid, "cache/", ctx);

// Create time-varying field
MPASOField field;
field.initField(&grid, &solution_t0, &solution_t1);

// Use for PathLine tracing
MPASOVisualizer viz;
viz.initVisualizer(&field, ctx);

TrajectorySettings config;
config.dt = 600.0;  // 10 minutes
config.total_timesteps = 1440;  // 10 days
config.integration_method = IntegrationMethod::RK4;

viz.PathLine(/* seed positions */, config, /* outputs */);
// Particles advect through velocity field interpolating between t0 and t1
```

### Shared Grid Across Multiple Fields

```cpp
// Shared grid for multiple timesteps
auto grid = std::make_shared<MPASOGrid>();
grid->initGrid(&reader);
grid->createKDTree("cache/kdtree.kdt", sycl_queue);

// Load multiple solutions
std::vector<std::shared_ptr<MPASOSolution>> solutions;
for (int t = 0; t < 100; ++t) {
    reader.setTimestep(t);
    auto sol = std::make_shared<MPASOSolution>();
    sol->initSolution(&reader);
    sol->calcCellCenterVelocity(grid.get(), "cache/", ctx);
    sol->calcCellVertexVelocity(grid.get(), "cache/", ctx);
    solutions.push_back(sol);
}

// Create fields for time intervals
for (int t = 0; t < 99; ++t) {
    MPASOField field;
    field.initField(grid, solutions[t], solutions[t+1]);
    // Use field for PathLine from timestep t to t+1
    // Grid is shared (not duplicated) across all fields
}
```

### Particle Seed Point Validation

```cpp
// Generate potential seed points
std::vector<vec3> candidatePoints;
for (double lat = -60; lat <= 60; lat += 5) {
    for (double lon = -180; lon < 180; lon += 5) {
        double lat_r = lat * M_PI / 180.0;
        double lon_r = lon * M_PI / 180.0;
        vec3 point = {
            cos(lat_r) * cos(lon_r),
            cos(lat_r) * sin(lon_r),
            sin(lat_r)
        };
        candidatePoints.push_back(point);
    }
}

// Filter to ocean-only points
std::vector<vec3> oceanSeedPoints;
std::vector<int> cellIDs;
std::vector<std::vector<size_t>> cellVertices;

for (const auto& point : candidatePoints) {
    int cellID;
    std::vector<size_t> vertices;
    
    if (field.isOnOcean(point, cellID, vertices)) {
        oceanSeedPoints.push_back(point);
        cellIDs.push_back(cellID);
        cellVertices.push_back(vertices);
    }
}

std::cout << "Valid ocean seed points: " << oceanSeedPoints.size() 
          << " / " << candidatePoints.size() << std::endl;

// Use oceanSeedPoints for particle initialization
```

### Dynamic Field Switching (Time Series)

```cpp
// Simulate advancing through time with sliding window
auto grid = std::make_shared<MPASOGrid>();
grid->initGrid(&reader);
grid->createKDTree("cache/kdtree.kdt", sycl_queue);

for (int t = 0; t < 100; ++t) {
    // Load two solutions for current interval
    reader.setTimestep(t);
    auto sol_t0 = std::make_shared<MPASOSolution>();
    sol_t0->initSolution(&reader);
    sol_t0->calcCellCenterVelocity(grid.get(), "cache/", ctx);
    
    reader.setTimestep(t + 1);
    auto sol_t1 = std::make_shared<MPASOSolution>();
    sol_t1->initSolution(&reader);
    sol_t1->calcCellCenterVelocity(grid.get(), "cache/", ctx);
    
    // Create field for this interval
    MPASOField field;
    field.initField(grid, sol_t0, sol_t1);
    
    // Run PathLine for this time interval
    viz.PathLine(/* seeds */, config, /* outputs */);
    
    // field destroyed, sol_t0/sol_t1 released when out of scope
    // Next iteration loads new solutions
}
```

## Implementation Notes

### Memory Management

- **Shared pointers**: Reference counting prevents premature deletion
- **Lightweight**: Field object is small (~24 bytes on 64-bit) - just three pointers
- **No data copying**: Field references existing grid/solution objects
- **Destruction order**: Field can be destroyed before or after grid/solutions (safe due to shared_ptr)

### Thread Safety

- **Initialization**: NOT thread-safe (call `initField()` from single thread)
- **Queries**: Thread-safe for read-only operations (`calcInWhichCells`, `isOnOcean`) after initialization
- **Multiple fields**: Safe to create multiple fields referencing same grid/solutions from different threads

### Temporal Interpolation Details

When `mSol_Back != nullptr`, GPU kernels (PathLine) interpolate as:
```cpp
double alpha = current_step / total_steps;  // 0.0 to 1.0
vec3 velocity_t0 = /* query mSol_Front */;
vec3 velocity_t1 = /* query mSol_Back */;
vec3 velocity_interpolated = (1.0 - alpha) * velocity_t0 + alpha * velocity_t1;
```

**Alpha parameter**:
- `alpha = 0.0` → use `mSol_Front` only
- `alpha = 1.0` → use `mSol_Back` only
- `alpha = 0.5` → equal weight (midpoint)

**Handled automatically** by `PathLine` kernel based on `TrajectorySettings.dt` and `total_timesteps`.

### Point-in-Polygon Test (isOnOcean)

Uses **Wachspress coordinates**:
1. Compute generalized barycentric weights for point relative to cell polygon
2. Check validity:
   - All weights ≥ -ε (small negative tolerance for numerical errors)
   - Sum of weights ≈ 1.0 (within tolerance)
3. If valid → inside polygon
4. If invalid → outside polygon (point on land or in neighboring cell)

**Tolerance**: Typically `ε = 1e-6` for weight negativity.

### Performance Considerations

- **KD-tree queries**: O(log N) per point, ~200 ns for 1M cells
- **Batch queries**: No parallelization in `calcInWhichCells(vector)` - CPU sequential loop
- **isOnOcean overhead**: ~10× slower than `calcInWhichCells` due to polygon test and land masking
- **Field construction**: Negligible cost (~1 µs) - just pointer assignments

### Design Rationale

**Why separate Field from Grid and Solution?**
- **Flexibility**: Same grid used with different solutions (time series)
- **Reusability**: Same solution visualized with different grids (multi-resolution)
- **Modularity**: Grid loading, solution computation, and field queries are independent
- **Memory**: Avoid duplicating large arrays when creating multiple fields

**Why shared_ptr instead of raw pointers?**
- **Lifetime management**: Field can outlive grid/solutions in async workflows
- **Shared ownership**: Multiple fields or visualizers can reference same data
- **Safety**: Automatic cleanup prevents dangling pointers

## Related Classes

- **MPASOGrid**: Provides mesh topology and KD-tree spatial index
- **MPASOSolution**: Provides velocity fields and ocean state at specific time
- **MPASOVisualizer**: Consumes `MPASOField` for rendering and particle tracing
- **TrajectorySettings**: Configuration for PathLine temporal interpolation behavior

## References

- Point-in-polygon: [Wachspress Coordinates](https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Wachspress_coordinates)
- Source files: `/pscratch/sd/q/qiuyf/MOPS/src/Core/MPASOField.{h,cpp}`
