# MPASOGrid - Mesh Structure and Spatial Indexing

## Overview

`MPASOGrid` is the foundational data structure in MOPS that manages the MPAS-Ocean unstructured spherical mesh topology and provides efficient spatial indexing capabilities. MPAS-Ocean uses a Voronoi tessellation where computational cells are irregular polygons distributed across the sphere, unlike traditional structured latitude-longitude grids.

The class serves three primary purposes:
1. **Storage**: Maintains the mesh geometry (cell/vertex/edge coordinates) and connectivity relationships
2. **Topology Navigation**: Provides methods to traverse mesh relationships (e.g., finding neighbors of a cell)
3. **Spatial Queries**: Implements KD-tree-based indexing to efficiently locate which cell contains an arbitrary 3D point

This class is essential for all visualization and particle trajectory operations in MOPS, as it enables mapping between geographic coordinates and the unstructured mesh.

## Key Concepts

### Unstructured Spherical Mesh

Unlike regular grids with fixed latitude/longitude spacing, MPAS-Ocean uses an unstructured mesh where:
- **Cells** (computational elements) are irregular polygons with 5-7 edges typically
- **Vertices** are the corners where multiple cells meet
- **Edges** connect cells and define their boundaries

The mesh is represented in 3D Cartesian coordinates (x, y, z) on a unit sphere, which avoids singularities at poles that plague lat/lon grids.

### Connectivity Primitives

The mesh topology is defined through index arrays that answer questions like:
- "Which vertices form the boundary of cell i?" → `verticesOnCell_vec`
- "Which cells share an edge with cell i?" → `cellsOnCell_vec`
- "Which cells meet at vertex j?" → `cellsOnVertex_vec`

These connectivity arrays use MPAS-Ocean's native indexing conventions (typically 1-indexed in the netCDF file, converted to 0-indexed internally).

### Spatial Indexing

To determine which cell contains a point (lat, lon, depth):
1. Convert geographic coordinates to 3D Cartesian
2. Query the KD-tree to find the nearest cell center
3. Optionally verify the point lies within the cell polygon (land masking)

The KD-tree dramatically accelerates queries from O(N) to O(log N) for meshes with millions of cells.

## Data Structures

### Grid Dimensions
```cpp
int mCellsSize;       // Total number of computational cells in the mesh
int mEdgesSize;       // Total number of edges
int mVertexSize;      // Total number of vertices
int mMaxEdgesSize;    // Maximum edges per cell (typically 7 for MPAS)
int mVertLevels;      // Number of vertical layers (nVertLevels in MPAS)
int mVertLevelsP1;    // mVertLevels + 1 (layer interfaces)
int mTimesteps;       // Number of time snapshots in the dataset
```

**Note**: `mCellsSize`, `mVertexSize`, and `mEdgesSize` define the "horizontal" mesh resolution. `mVertLevels` defines vertical resolution (typically 60-100 layers in ocean models).

### Metadata
```cpp
std::string mMeshName;        // Identifier for the mesh (e.g., "EC60to30")
std::string mCachedDataDir;   // Directory for cached binary files (.bin)
std::string mFolderPath;      // Root path to MPAS dataset
```

### Coordinates

All spatial coordinates are stored as 3D Cartesian vectors on a unit sphere (Earth radius normalized to 1.0 in MPAS files):

```cpp
std::vector<vec3> cellCoord_vec;      // [mCellsSize] Cell centers (x,y,z)
std::vector<vec3> vertexCoord_vec;    // [mVertexSize] Vertex positions (x,y,z)
std::vector<vec3> edgeCoord_vec;      // [mEdgesSize] Edge midpoints (x,y,z)
std::vector<vec2> vertexLatLon_vec;   // [mVertexSize] Vertex positions (lat, lon in radians)
```

**Usage Note**: 
- `cellCoord_vec` is used for KD-tree construction and nearest-cell queries
- `vertexLatLon_vec` is used for geographic-to-Cartesian conversions
- Cartesian coordinates avoid singularities in spherical geometry calculations

### Connectivity Arrays

These arrays encode the mesh topology using index lists. MPAS uses ragged arrays (variable-length sublists), which are flattened into 1D vectors with fixed stride:

```cpp
std::vector<size_t> verticesOnCell_vec;   // [mCellsSize × mMaxEdgesSize] 
                                           // Indices of vertices forming each cell's boundary
                                           
std::vector<size_t> cellsOnCell_vec;      // [mCellsSize × mMaxEdgesSize]
                                           // Indices of cells sharing edges with each cell
                                           
std::vector<size_t> cellsOnVertex_vec;    // [mVertexSize × 3]
                                           // Indices of 3 cells meeting at each vertex
                                           
std::vector<size_t> edgesOnCell_vec;      // [mCellsSize × mMaxEdgesSize]
                                           // Indices of edges forming each cell's boundary
                                           
std::vector<size_t> cellsOnEdge_vec;      // [mEdgesSize × 2]
                                           // Indices of 2 cells sharing each edge
                                           
std::vector<size_t> verticesOnEdge_vec;   // [mEdgesSize × 2]
                                           // Indices of 2 vertices at each edge's endpoints
                                           
std::vector<size_t> numberVertexOnCell_vec; // [mCellsSize]
                                             // Actual number of vertices per cell (5-7)
```

**Indexing Convention**:
- Cell `i` has vertices at indices `verticesOnCell_vec[i*mMaxEdgesSize + j]` for `j = 0...numberVertexOnCell_vec[i]-1`
- Unused entries (when a cell has fewer than `mMaxEdgesSize` vertices) contain sentinel values (typically 0 or SIZE_MAX after conversion)

### Interpolation Weights
```cpp
std::vector<float> cellWeight_vec;  // [mVertexSize × 3]
                                     // Barycentric weights for vertex-to-cell interpolation
```

Used to interpolate values from vertices to cell centers (or vice versa) using the three cells meeting at each vertex.

### Topography
```cpp
std::vector<double> cellRefBottomDepth_vec;  // [mVertLevels]
                                              // Reference depth at each vertical layer interface (positive down)
```

Defines the nominal vertical layer structure. Actual layer depths vary per cell based on `layerThickness` in `MPASOSolution`.

### Spatial Index
```cpp
#if _WIN32 || __linux__
    std::unique_ptr<KDTree_t> mKDTree;      // nanoflann-based KD-tree
#elif __APPLE__
    std::unique_ptr<kdtreegpu> mKDTree;     // GPU-accelerated KD-tree (macOS)
#endif
```

Platform-specific KD-tree implementation for fast nearest-cell queries. Built from `cellCoord_vec`.

## Methods

### Initialization

#### `void initGrid(MPASOReader* reader)`
Load grid data from an MPAS netCDF file via `MPASOReader`.

**Parameters**:
- `reader`: Pointer to initialized `MPASOReader` with open MPAS file

**Behavior**:
- Reads mesh dimensions (`nCells`, `nVertices`, `nEdges`, `nVertLevels`, `maxEdges`)
- Loads coordinate arrays (`xCell`, `yCell`, `zCell`, `latVertex`, `lonVertex`, etc.)
- Loads connectivity arrays (`cellsOnCell`, `verticesOnCell`, etc.)
- Converts MPAS 1-based indices to 0-based
- Stores metadata (`mFolderPath`, `mMeshName`)

**Example**:
```cpp
MPASOReader reader("path/to/ocean.nc");
MPASOGrid grid;
grid.initGrid(&reader);
```

#### `void initGrid(ftk::ndarray_group* g, MPASOReader* reader = nullptr)`
Load grid from FTK `ndarray_group` (alternative loader for preprocessed data).

**Parameters**:
- `g`: Pointer to FTK ndarray group containing MPAS mesh variables
- `reader`: Optional reader for supplementary metadata

**Use Case**: When mesh data is already loaded into FTK's ndarray structure (e.g., from parallel I/O pipelines).

#### `void initGrid_FromBin(const char* prefix)`
Load grid from cached binary files (faster than netCDF for repeated use).

**Parameters**:
- `prefix`: Path prefix for binary cache files (e.g., `"/data/EC60to30"`)

**Expected Files**:
- `{prefix}_cellCoord.bin`
- `{prefix}_vertexCoord.bin`
- `{prefix}_verticesOnCell.bin`
- etc.

**Behavior**: Directly reads binary arrays via `readFromBlock_Vec3()` and `readFromBlock_Int()`.

#### `void initGrid_DemoLoading(const char* yaml_path)`
Load grid configuration from a YAML file (for testing/demos).

**Parameters**:
- `yaml_path`: Path to YAML configuration with mesh parameters

### Spatial Indexing

#### `void createKDTree(const char* kdTree_path, sycl::queue& SYCL_Q)`
Build or load a KD-tree for spatial queries.

**Parameters**:
- `kdTree_path`: Path to save/load serialized KD-tree (`.kdt` file)
- `SYCL_Q`: SYCL queue for GPU-accelerated construction (macOS only)

**Behavior**:
1. If `kdTree_path` exists, load pre-built tree from disk
2. Otherwise, construct tree from `cellCoord_vec` and save to `kdTree_path`
3. On Linux/Windows, uses nanoflann (CPU); on macOS, uses GPU-accelerated builder

**Performance**: Building a KD-tree for 1M cells takes ~5s (CPU) or ~1s (GPU). Loading from disk takes <1s.

**Example**:
```cpp
sycl::queue q;
grid.createKDTree("cache/kdtree.kdt", q);
```

#### `void searchKDT(const CartesianCoord& point, int& cell_id)`
Find the nearest cell to a 3D Cartesian point.

**Parameters**:
- `point`: 3D Cartesian coordinate (x, y, z) on the sphere
- `cell_id`: Output parameter receiving the nearest cell index

**Returns**: Modifies `cell_id` to the index of the closest cell center.

**Example**:
```cpp
vec3 queryPoint = {0.5, 0.5, 0.707};  // Normalize to unit sphere
int cellID;
grid.searchKDT(queryPoint, cellID);
// cellID now contains the index of the nearest cell
```

### Topology Navigation

#### `void getNeighborCells(const size_t cell_id, std::vector<size_t>& cell_on_cell, std::vector<size_t>& neighbor_id)`
Retrieve all cells sharing edges with a given cell.

**Parameters**:
- `cell_id`: Index of query cell
- `cell_on_cell`: Output vector of neighbor indices from `cellsOnCell_vec` (raw)
- `neighbor_id`: Output vector of valid neighbor cell indices (filtered)

**Example**:
```cpp
std::vector<size_t> raw, neighbors;
grid.getNeighborCells(42, raw, neighbors);
// neighbors contains [23, 45, 67, 89, ...] (adjacent cell IDs)
```

#### `void getVerticesOnCell(const size_t cell_id, std::vector<size_t>& vertex_on_cell, std::vector<size_t>& vertex_id)`
Get vertices forming a cell's boundary.

**Parameters**:
- `cell_id`: Index of query cell
- `vertex_on_cell`: Output vector of vertex indices (including padding)
- `vertex_id`: Output vector of valid vertex indices only

**Example**:
```cpp
std::vector<size_t> raw, vertices;
grid.getVerticesOnCell(100, raw, vertices);
// vertices contains [301, 302, 303, 304, 305] (5 vertices for a pentagon cell)
```

#### `void getCellsOnVertex(const size_t vertex_id, std::vector<size_t>& cell_on_vertex, std::vector<size_t>& cell_id)`
Get the three cells meeting at a vertex.

**Parameters**:
- `vertex_id`: Index of query vertex
- `cell_on_vertex`: Output vector of cell indices (raw, length 3)
- `cell_id`: Output vector of valid cell indices

**Example**:
```cpp
std::vector<size_t> raw, cells;
grid.getCellsOnVertex(200, raw, cells);
// cells contains [50, 51, 52] (3 cells around vertex 200)
```

#### `void getCellsOnEdge(const size_t edge_id, std::vector<size_t>& cell_on_edge, std::vector<size_t>& cell_id)`
Get the two cells sharing an edge.

**Parameters**:
- `edge_id`: Index of query edge
- `cell_on_edge`: Output vector (length 2)
- `cell_id`: Output vector of valid cell indices

#### `void getEdgesOnCell(const size_t cell_id, std::vector<size_t>& edge_on_cell, std::vector<size_t>& edge_id)`
Get edges forming a cell's boundary.

**Parameters**:
- `cell_id`: Index of query cell
- `edge_on_cell`: Output vector of edge indices (raw)
- `edge_id`: Output vector of valid edge indices

### Attribute Setters

These methods allow dynamic modification of grid properties (used by loaders and preprocessing tools):

#### `void setGridAttribute(GridAttributeType type, int val)`
Set a scalar grid attribute.

**Parameters**:
- `type`: Attribute identifier (e.g., `GridAttributeType::kCellSize`)
- `val`: Integer value

**Example**:
```cpp
grid.setGridAttribute(GridAttributeType::kVertLevels, 64);
```

#### `void setGridAttributesVec3(GridAttributeType type, const std::vector<vec3>& vec)`
Set a vec3 array attribute.

**Supported Types**:
- `kVertexCoord`, `kCellCoord`, `kEdgeCoord`

#### `void setGridAttributesVec2(GridAttributeType type, const std::vector<vec2>& vec)`
Set a vec2 array attribute.

**Supported Types**:
- `kVertexLatLon`

#### `void setGridAttributesInt(GridAttributeType type, const std::vector<size_t>& vec)`
Set an integer array attribute.

**Supported Types**:
- `kVerticesOnCell`, `kCellsOnCell`, `kCellsOnVertex`, `kEdgesOnCell`, `kCellsOnEdge`, `kNumberVertexOnCell`

#### `void setGridAttributesFloat(GridAttributeType type, const std::vector<float>& vec)`
Set a float array attribute.

**Supported Types**:
- `kCellWeight`

### Utilities

#### `std::string getFolderPath() const`
Returns the root path to the MPAS dataset.

#### `bool checkAttribute()`
Validates that all required grid attributes are properly initialized.

**Returns**: `true` if all critical arrays are non-empty and dimensions are consistent.

## Usage Examples

### Loading and Indexing a Mesh

```cpp
#include "Core/MPASOGrid.h"
#include "IO/MPASOReader.h"

// Initialize grid from MPAS netCDF file
MPASOReader reader("data/MPAS-Ocean.nc");
MOPS::MPASOGrid grid;
grid.initGrid(&reader);

// Build spatial index
sycl::queue q;
grid.createKDTree("cache/ocean_kdtree.kdt", q);

// Query nearest cell for a point at (45deg N, 180deg W, surface)
double lat = 45.0 * M_PI / 180.0;  // Convert to radians
double lon = -180.0 * M_PI / 180.0;
vec3 cartesian = {cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)};

int cellID;
grid.searchKDT(cartesian, cellID);
std::cout << "Point is in cell " << cellID << std::endl;
```

### Traversing Mesh Topology

```cpp
// Find all neighbors of a cell
size_t targetCell = 1000;
std::vector<size_t> rawNeighbors, validNeighbors;
grid.getNeighborCells(targetCell, rawNeighbors, validNeighbors);

std::cout << "Cell " << targetCell << " has " << validNeighbors.size() 
          << " neighbors: ";
for (auto neighbor : validNeighbors) {
    std::cout << neighbor << " ";
}
std::cout << std::endl;

// Get vertices of that cell
std::vector<size_t> rawVertices, vertices;
grid.getVerticesOnCell(targetCell, rawVertices, vertices);

// Print vertex coordinates
for (auto vIdx : vertices) {
    vec3 vCoord = grid.vertexCoord_vec[vIdx];
    std::cout << "Vertex " << vIdx << ": (" 
              << vCoord.x << ", " << vCoord.y << ", " << vCoord.z << ")" 
              << std::endl;
}
```

### Batch Spatial Queries

```cpp
// Query cells for multiple points (e.g., particle positions)
std::vector<vec3> particlePositions = {
    {0.5, 0.5, 0.707},
    {-0.3, 0.8, 0.5},
    // ... thousands of particles
};

std::vector<int> cellIDs(particlePositions.size());
for (size_t i = 0; i < particlePositions.size(); ++i) {
    grid.searchKDT(particlePositions[i], cellIDs[i]);
}
```

### Using Cached Binary Files

```cpp
// First run: load from netCDF and cache
MPASOReader reader("data/ocean.nc");
MOPS::MPASOGrid grid;
grid.initGrid(&reader);
// (Manually save binary files using private readFromBlock methods)

// Subsequent runs: fast binary loading
MOPS::MPASOGrid fastGrid;
fastGrid.initGrid_FromBin("cache/ocean_grid");
// Loads all connectivity/coordinate arrays from .bin files
```

### Coordinate Conversion

```cpp
// Convert lat/lon to 3D Cartesian (for KD-tree query)
auto latLonToCartesian = [](double lat, double lon) -> vec3 {
    return {
        cos(lat) * cos(lon),
        cos(lat) * sin(lon),
        sin(lat)
    };
};

// Query by geographic coordinates
double latitude = 30.0 * M_PI / 180.0;   // 30deg N
double longitude = -120.0 * M_PI / 180.0; // 120deg W
vec3 xyz = latLonToCartesian(latitude, longitude);

int cellID;
grid.searchKDT(xyz, cellID);
```

## Implementation Notes

### Memory Layout

Connectivity arrays use **strided indexing** to handle variable-length sublists:
- Cell `i` has `nVertices = numberVertexOnCell_vec[i]` vertices
- Vertex indices are at `verticesOnCell_vec[i * mMaxEdgesSize + j]` for `j = 0...nVertices-1`
- Unused slots (when `nVertices < mMaxEdgesSize`) contain sentinel values

This layout enables GPU-friendly access patterns (coalesced memory reads) compared to variable-length lists.

### Index Convention

MPAS netCDF files use **1-based indexing** (Fortran convention). MOPS converts to **0-based** during loading:
```cpp
size_t zeroBasedIndex = ndarrayValue - 1;
```

### Spherical Geometry

All spatial operations use 3D Cartesian coordinates to avoid:
- **Pole singularities** (lat/lon undefined at 90deg)
- **Date-line wrapping** (discontinuity at ±180deg)
- **Metric distortion** (distance calculations are uniform in Cartesian space)

### KD-Tree Performance

For a mesh with N cells:
- Construction: O(N log N)
- Query: O(log N)
- Typical performance: 1M cells → ~200 nanoseconds per query (CPU)

### Platform Differences

- **Linux/Windows**: Uses `nanoflann` library (header-only, CPU-based)
- **macOS**: Uses custom `kdtreegpu` (Metal/SYCL accelerated)

Both provide identical `searchKDT()` interface.

## Related Classes

- **MPASOSolution**: Stores time-varying fields (velocity, temperature) on the mesh
- **MPASOField**: Combines `MPASOGrid` + `MPASOSolution` for spatiotemporal queries
- **MPASOVisualizer**: Uses grid topology for field visualization and particle tracing
- **MPASOReader**: Reads MPAS netCDF files and populates `MPASOGrid`

## References

- [MPAS-Ocean User's Guide](https://mpas-dev.github.io/ocean/ocean.html)
- [Unstructured Mesh Specification](https://mpas-dev.github.io/files/documents/MPAS-MeshSpec.pdf)
- Grid files referenced: `/pscratch/sd/q/qiuyf/MOPS/src/Core/MPASOGrid.{h,cpp}`
