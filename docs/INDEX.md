# MOPS Documentation Index

**Last Updated**: 2026-08-09  
**Version**: 1.0

---

## Getting Started

- **[README](README.md)** - Project overview and quick start
- **[Getting Started Guide](GETTING_STARTED.md)** - Complete tutorial for new users
- **[Installation](GETTING_STARTED.md#installation)** - Build from source

---

## Core Components

### MPASGrid
**Purpose**: Manages MPAS-Ocean unstructured grid topology and spatial indexing

**Documentation**: [MPAS_GRID.md](MPAS_GRID.md)

**Key Features**:
- Unstructured spherical mesh (cells, vertices, edges)
- Connectivity arrays for mesh topology
- KD-tree spatial indexing
- File: `src/Core/MPASOGrid.{h,cpp}`

### MPASOSolution
**Purpose**: Stores time-varying ocean state

**Documentation**: [MPAS_SOLUTION.md](MPAS_SOLUTION.md)

**Key Features**:
- Cell-centered and vertex-centered velocity fields
- Vertical coordinate system (zTop)
- Extensible attributes (temperature, salinity, etc.)
- Multiple velocity representations (edge-normal, zonal/meridional, 3D Cartesian)
- Invalid value handling and attribute interpolation
- File: `src/Core/MPASOSolution.{h,cpp}`

### MPASOField
**Purpose**: Combines grid and solution(s)

**Documentation**: [MPAS_FIELD.md](MPAS_FIELD.md)

**Key Features**:
- Links grid with one or two solutions
- Supports temporal interpolation (for PathLine)
- Spatial queries (cell location, ocean/land detection)
- Shared pointer resource management
- File: `src/Core/MPASOField.{h,cpp}`

### MPASOVisualizer
**Purpose**: High-level API for visualization and particle tracing

**Methods**:
- `VisualizeFixedLayer()` - Velocity at constant vertical layer
- `VisualizeFixedDepth()` - Velocity at constant depth
- `VisualizeFixedLatitude()` - Latitude cross-section
- `StreamLine()` - Steady-state particle tracing
- `PathLine()` - Time-varying particle tracing
- File: `src/Core/MPASOVisualizer.{h,cpp}`

---

## GPU Implementations

### CUDA Backend

**Documentation**: [CUDA_IMPLEMENTATION.md](CUDA_IMPLEMENTATION.md)

**Location**: `src/GPU/CUDA/Kernel/`

**Unified Infrastructure** (2026-09):
- **CUDAConstants.cuh**: Unified constant definitions (MAX_VERTEX_NUM, MAX_VERTICAL_LEVELS, etc.)
- **CUDAMemoryUtils.cuh**: Unified memory management utilities (DeviceAlloc, CopyHostToDevice, etc.)
- **Eliminates 20+ duplicate constant definitions** with inconsistent values
- **Centralized error checking** and consistent memory patterns

**Detailed Step-by-Step Algorithms**:

1. **VisualizeFixedLayer**
   - Pixel → lat/lon → XYZ conversion
   - KD-tree cell lookup
   - Wachspress interpolation
   - XYZ → ENU velocity conversion

2. **VisualizeFixedDepth**
   - Horizontal interpolation
   - zTop profile computation
   - Layer finding (linear search)
   - Vertical interpolation

3. **VisualizeFixedLatitude**
   - Depth-longitude cross-section
   - CPU implementation

4. **StreamLine**
   - Per-particle trajectory
   - Cell tracking
   - Binary search for layer finding
   - Euler and RK4 integration
   - Spherical surface advection

5. **PathLine**
   - Time-varying velocity fields
   - Temporal interpolation
   - Boundary handling
   - Attribute tracking

### SYCL Backend

**Documentation**: [SYCL_IMPLEMENTATION.md](SYCL_IMPLEMENTATION.md)

**Location**: `src/GPU/SYCL/MPASOVisualizerSYCL.cpp`

**Key Features**:
- Buffer/accessor memory model
- Portable across Intel/NVIDIA/AMD
- Binary search optimization (StreamLine)
- Same algorithms as CUDA with SYCL syntax

**Comparison Table**:

| Aspect | CUDA | SYCL |
|--------|------|------|
| Memory | Explicit cudaMalloc | Buffer/accessor |
| Launch | <<<grid,block>>> | parallel_for |
| Portability | NVIDIA only | Multi-vendor |

---

## Algorithms

### Wachspress Interpolation
- Generalized barycentric coordinates
- Works on arbitrary convex polygons
- Used for horizontal interpolation within cells

### Vertical Interpolation
- Linear interpolation between layers
- Binary search for layer finding (SYCL StreamLine)
- Linear search (CUDA VisualizeFixedDepth)

### Particle Advection
- **Euler method**: Simple forward integration
- **RK4 method**: 4th-order Runge-Kutta
- Spherical surface advection using Rodrigues' rotation

### KD-Tree Spatial Search
- O(log N) nearest cell queries
- Pre-computed for efficiency
- Platform-specific implementations

---

## File Organization

```
MOPS/
├── src/
│   ├── Core/
│   │   ├── MPASOGrid.{h,cpp}
│   │   ├── MPASOSolution.{h,cpp}
│   │   ├── MPASOField.{h,cpp}
│   │   └── MPASOVisualizer.{h,cpp}
│   │
│   └── GPU/
│       ├── CUDA/
│       │   └── Kernel/
│       │       └── MPASOVisualizerKernels.cu
│       │
│       └── SYCL/
│           ├── MPASOVisualizerSYCL.cpp
│           └── Kernel/
│               └── SYCLKernel.cpp
│
├── tutorial/
│   ├── pathLine.cpp
│   ├── reGrid.cpp
│   └── pyMOPSAPI.py
│
└── docs/
    ├── README.md
    ├── INDEX.md
    ├── GETTING_STARTED.md
    ├── MPAS_GRID.md
    ├── CUDA_IMPLEMENTATION.md
    └── SYCL_IMPLEMENTATION.md
```

---

## Usage Examples

See [GETTING_STARTED.md](GETTING_STARTED.md) for complete working examples.

---

## Related Resources

- **MPAS-Ocean Documentation**: https://mpas-dev.github.io/
- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/
- **SYCL Specification**: https://www.khronos.org/sycl/
