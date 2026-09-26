#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "Utils/Log.hpp"

namespace MOPS {
namespace GPU {
namespace CUDABackend {
namespace MemoryUtils {

// ============================================================================
// CUDA Memory Management Utilities
//
// Provides type-safe, error-checked wrappers for common CUDA memory operations.
// All functions use CheckCuda for error handling and provide descriptive messages.
// ============================================================================

inline void CheckCuda(cudaError_t code, const char* what)
{
    if (code != cudaSuccess) {
        Error("[CUDA]::%s failed: %s", what, cudaGetErrorString(code));
    }
}

// ============================================================================
// Basic Operations (Single Responsibility)
// ============================================================================

/// Allocate device memory for 'count' elements of type T
template <typename T>
T* DeviceAlloc(size_t count, const char* what = "cudaMalloc")
{
    if (count == 0) {
        return nullptr;
    }

    T* ptr = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)), what);
    return ptr;
}

/// Copy data from host vector to device pointer
template <typename T>
void CopyHostToDevice(T* dst, const std::vector<T>& src, const char* what = "cudaMemcpy H2D")
{
    if (src.empty() || dst == nullptr) {
        return;
    }
    CheckCuda(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), what);
}

/// Copy data from device pointer to host vector
template <typename T>
void CopyDeviceToHost(std::vector<T>& dst, const T* src, const char* what = "cudaMemcpy D2H")
{
    if (dst.empty() || src == nullptr) {
        return;
    }
    CheckCuda(cudaMemcpy(dst.data(), src, dst.size() * sizeof(T), cudaMemcpyDeviceToHost), what);
}

/// Free device memory
template <typename T>
void FreeDev(T* dev_ptr, const char* what = "cudaFree")
{
    if (dev_ptr != nullptr) {
        CheckCuda(cudaFree(dev_ptr), what);
    }
}

// ============================================================================
// Convenience Combinations (Common Patterns)
// ============================================================================

/// Allocate device memory and copy data from host vector (Alloc + H2D Copy)
template <typename T>
T* DeviceAllocAndCopy(const std::vector<T>& src, const char* what = "cudaMalloc/cudaMemcpy H2D")
{
    if (src.empty()) {
        return nullptr;
    }

    T* dst = DeviceAlloc<T>(src.size(), what);
    CopyHostToDevice(dst, src, what);
    return dst;
}

/// Copy data from device to host and free device memory (D2H Copy + Free)
template <typename T>
void CopyBackAndFree(std::vector<T>& dst, T* dev_ptr, const char* what = "cudaMemcpy D2H/cudaFree")
{
    if (dev_ptr == nullptr) {
        return;
    }

    if (!dst.empty()) {
        CopyDeviceToHost(dst, dev_ptr, what);
    }
    FreeDev(dev_ptr, what);
}

} // namespace MemoryUtils
} // namespace CUDABackend
} // namespace GPU
} // namespace MOPS
