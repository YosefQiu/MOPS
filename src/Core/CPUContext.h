#pragma once

namespace MOPS {

enum class CPUBackend : int {
    kUnknown = 0,
    kTBB,
    kOpenMP
};

struct CPUContext {
    CPUBackend backend = CPUBackend::kUnknown;
    int numThreads = 0;
};

} // namespace MOPS
