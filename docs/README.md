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
- **[MPASGrid](MPAS_GRID.md)** - Unstructured mesh structure and spatial indexing
- **MPASOSolution** - Velocity fields and ocean solution data
- **MPASOField** - Combined grid and solution interface
- **MPASOVisualizer** - Visualization and particle tracing

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

✅ **[CUDA_IMPLEMENTATION.md](CUDA_IMPLEMENTATION.md)** - Detailed step-by-step algorithms for:
- VisualizeFixedLayer
- VisualizeFixedDepth  
- VisualizeFixedLatitude
- StreamLine (with binary search optimization)
- PathLine (time-varying fields)

✅ **[SYCL_IMPLEMENTATION.md](SYCL_IMPLEMENTATION.md)** - Detailed step-by-step algorithms for SYCL portable GPU code

✅ **[MPAS_GRID.md](MPAS_GRID.md)** - Complete MPASGrid class documentation

✅ **[GETTING_STARTED.md](GETTING_STARTED.md)** - Tutorial with working examples

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

