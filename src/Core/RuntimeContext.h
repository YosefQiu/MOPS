#pragma once

#include "Core/CPUContext.h"
#include "Core/GPUContext.h"

namespace MOPS {

enum class RuntimeKind : int {
    kUnknown = 0,
    kGPU,
    kCPU
};

struct RuntimeContext {
    RuntimeKind kind = RuntimeKind::kUnknown;
    GPUContext gpu;
    CPUContext cpu;

    static RuntimeContext FromGPU(const GPUContext& gpuCtx)
    {
        RuntimeContext ctx;
        ctx.kind = RuntimeKind::kGPU;
        ctx.gpu = gpuCtx;
        return ctx;
    }

    static RuntimeContext FromCPU(const CPUContext& cpuCtx)
    {
        RuntimeContext ctx;
        ctx.kind = RuntimeKind::kCPU;
        ctx.cpu = cpuCtx;
        return ctx;
    }
};

} // namespace MOPS
