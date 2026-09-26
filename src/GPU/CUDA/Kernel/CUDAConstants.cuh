#pragma once

// ============================================================================
// CUDA Constants - Unified Definition
//
// All CUDA kernels should include this file and use these constants.
// DO NOT define these constants locally in kernel functions.
// ============================================================================

namespace MOPS {
namespace GPU {
namespace CUDABackend {
namespace Constants {

// ============================================================================
// Mesh Topology Limits
// ============================================================================

/// Maximum number of vertices per cell in MPAS mesh
/// This is the maximum polygon size we support
constexpr int MAX_VERTEX_NUM = 10;

/// Maximum number of cell neighbors (including self)
/// MPAS cells can have up to 10 neighbors + self = 11
constexpr int MAX_CELL_NEIGHBOR_NUM = 11;

// ============================================================================
// Vertical Structure Limits
// ============================================================================

/// Maximum number of vertical levels
/// MPAS-Ocean typically uses 60-100 levels
constexpr int MAX_VERTICAL_LEVELS = 100;

/// Maximum number of z-top layers (levels + 1 for boundaries)
constexpr int MAX_ZTOP_LAYERS = MAX_VERTICAL_LEVELS + 1;

// ============================================================================
// Performance Tuning Parameters
// ============================================================================

/// Default CUDA block size for 1D kernels
constexpr int DEFAULT_BLOCK_SIZE_1D = 128;

/// Default CUDA block size for 2D kernels (16x16)
constexpr int DEFAULT_BLOCK_SIZE_2D = 16;

/// Maximum grid dimension (CUDA architecture limit)
constexpr int MAX_GRID_DIM = 65535;

} // namespace Constants
} // namespace CUDABackend
} // namespace GPU
} // namespace MOPS
