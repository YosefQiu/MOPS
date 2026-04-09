#pragma once

#include "ggl.h"

namespace MOPS {

enum class GPUBackend : int {
    kUnknown = 0,
    kSYCL,
    kCUDA,
    kHIP
};

struct GPUContext {
    GPUBackend backend = GPUBackend::kUnknown;
    void* nativeQueue = nullptr;

    static GPUContext FromSYCL(sycl::queue& q) {
        GPUContext ctx;
        ctx.backend = GPUBackend::kSYCL;
        ctx.nativeQueue = static_cast<void*>(&q);
        return ctx;
    }

    static GPUContext FromCUDA(void* stream = nullptr) {
        GPUContext ctx;
        ctx.backend = GPUBackend::kCUDA;
        ctx.nativeQueue = stream;
        return ctx;
    }

    static GPUContext FromHIP(void* stream = nullptr) {
        GPUContext ctx;
        ctx.backend = GPUBackend::kHIP;
        ctx.nativeQueue = stream;
        return ctx;
    }

    sycl::queue* syclQueue() const {
        if (backend != GPUBackend::kSYCL || nativeQueue == nullptr) {
            return nullptr;
        }
        return static_cast<sycl::queue*>(nativeQueue);
    }
};

} // namespace MOPS
