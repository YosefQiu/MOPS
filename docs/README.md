# MOPS Documentation

**MOPS** - MPAS Ocean Particle Simulator  
**Version**: 1.0

👉 **[Start Here: Getting Started Guide](GETTING_STARTED.md)** - Complete tutorial for new users  
📚 **[Documentation Index](INDEX.md)** - Browse all documentation by topic

---

## Quick Links

### For New Users
- **[Getting Started](GETTING_STARTED.md)** - Quick tutorial with examples  
- **[Installation Guide](GETTING_STARTED.md#installation)** - Build from source
- **[Your First Visualization](GETTING_STARTED.md#your-first-visualization)** - Hello World

### Core Components  
- **[MPASOGrid](MPAS_GRID.md)** - Unstructured mesh structure and spatial indexing
- **[MPASOSolution](MPAS_SOLUTION.md)** - Velocity fields and ocean solution data
- **[MPASOField](MPAS_FIELD.md)** - Combined grid and solution interface
- **MPASOVisualizer** - Visualization and particle tracing API

### GPU Implementations
- **[CUDA Backend](CUDA_IMPLEMENTATION.md)** - NVIDIA GPU acceleration with detailed step-by-step algorithms
- **[SYCL Backend](SYCL_IMPLEMENTATION.md)** - Portable GPU acceleration with detailed step-by-step algorithms

---

## What is MOPS?

MOPS is a high-performance visualization and particle tracing framework for **MPAS-Ocean** data. It provides:

- **GPU-accelerated visualization** of ocean velocity fields
- **Fast particle trajectory computation** (streamlines and pathlines)
- **Support for unstructured spherical meshes**
- **Wachspress coordinate interpolation** for accurate results
- **Multiple integration methods**: Euler and RK4

---

## Documentation Created from Code Analysis

All documentation in this directory was created by thoroughly analyzing the actual MOPS source code:

### Core Data Structures
✅ **[MPAS_GRID.md](MPAS_GRID.md)** - Complete MPASGrid class documentation  
✅ **[MPAS_SOLUTION.md](MPAS_SOLUTION.md)** - MPASOSolution with velocity representations and zTop  
✅ **[MPAS_FIELD.md](MPAS_FIELD.md)** - MPASOField spatial queries and temporal interpolation

### GPU Implementations
✅ **[CUDA_IMPLEMENTATION.md](CUDA_IMPLEMENTATION.md)** - Detailed CUDA algorithms with:
- Unified infrastructure (CUDAConstants.cuh, CUDAMemoryUtils.cuh)
- VisualizeFixedLayer/Depth/Latitude algorithms
- StreamLine and PathLine particle tracing
- maxLevelCell clamping and configuration parameters
- Performance optimization and debugging

✅ **[SYCL_IMPLEMENTATION.md](SYCL_IMPLEMENTATION.md)** - SYCL portable GPU algorithms

### Tutorials
✅ **[GETTING_STARTED.md](GETTING_STARTED.md)** - Complete tutorial with working examples

---

## Quick Example

```cpp
#include "Core/MPASOVisualizer.h"

MPASOField field;
field.initField(grid, solution);

VisualizationSettings config;
config.imageSize = vec2(1024, 512);
config.FixedDepth = 100.0;

ImageBuffer<double> img;
MPASOVisualizer::VisualizeFixedDepth(&field, &config, &img, sycl_queue);
```

