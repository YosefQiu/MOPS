#include "GPU/Common/GPUFactory.h"

#if defined(MOPS_USE_SYCL) && MOPS_USE_SYCL
#include "GPU/SYCL/MPASOSolutionSYCL.h"
#include "GPU/SYCL/MPASOVisualizerSYCL.h"
#endif

#if defined(MOPS_USE_CUDA) && MOPS_USE_CUDA
#include "GPU/CUDA/MPASOSolutionCUDA.h"
#include "GPU/CUDA/MPASOVisualizerCUDA.h"
#endif

#if defined(MOPS_USE_HIP) && MOPS_USE_HIP
#include "GPU/HIP/MPASOSolutionHIP.h"
#include "GPU/HIP/MPASOVisualizerHIP.h"
#endif

namespace {
#if defined(MOPS_USE_SYCL) && MOPS_USE_SYCL
sycl::queue* ResolveSyclQueue(const MOPS::GPUContext& ctx, const char* fn_name) {
    sycl::queue* q = ctx.syclQueue();
    if (q == nullptr) {
        Error("[GPUFactory]::%s requires a valid SYCL queue in GPUContext", fn_name);
    }
    return q;
}
#endif
} // namespace

namespace MOPS::GPU::Factory {

void VisualizeFixedLayer(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, const GPUContext& ctx)
{
    if (ctx.backend == GPUBackend::kSYCL) {
#if defined(MOPS_USE_SYCL) && MOPS_USE_SYCL
    sycl::queue* q = ResolveSyclQueue(ctx, "VisualizeFixedLayer");
    if (q == nullptr) return;
    SYCLBackend::VisualizeFixedLayer(mpasoF, config, img, *q);
#else
    Error("[GPUFactory]::VisualizeFixedLayer SYCL backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kHIP) {
#if defined(MOPS_USE_HIP) && MOPS_USE_HIP
    HIPBackend::VisualizeFixedLayer(mpasoF, config, img);
#else
    Error("[GPUFactory]::VisualizeFixedLayer HIP backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kCUDA) {
#if defined(MOPS_USE_CUDA) && MOPS_USE_CUDA
    CUDABackend::VisualizeFixedLayer(mpasoF, config, img);
#else
    Error("[GPUFactory]::VisualizeFixedLayer CUDA backend is not enabled at build time");
#endif
    return;
    }
    Error("[GPUFactory]::VisualizeFixedLayer backend is unsupported");
}

void VisualizeFixedDepth(MPASOField* mpasoF, VisualizationSettings* config, std::vector<ImageBuffer<double>>& img_vec, const GPUContext& ctx)
{
    if (ctx.backend == GPUBackend::kSYCL) {
#if defined(MOPS_USE_SYCL) && MOPS_USE_SYCL
    sycl::queue* q = ResolveSyclQueue(ctx, "VisualizeFixedDepth");
    if (q == nullptr) return;
    SYCLBackend::VisualizeFixedDepth(mpasoF, config, img_vec, *q);
#else
    Error("[GPUFactory]::VisualizeFixedDepth SYCL backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kHIP) {
#if defined(MOPS_USE_HIP) && MOPS_USE_HIP
    HIPBackend::VisualizeFixedDepth(mpasoF, config, img_vec);
#else
    Error("[GPUFactory]::VisualizeFixedDepth HIP backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kCUDA) {
#if defined(MOPS_USE_CUDA) && MOPS_USE_CUDA
    CUDABackend::VisualizeFixedDepth(mpasoF, config, img_vec);
#else
    Error("[GPUFactory]::VisualizeFixedDepth CUDA backend is not enabled at build time");
#endif
    return;
    }
    Error("[GPUFactory]::VisualizeFixedDepth backend is unsupported");
}

void VisualizeFixedLatitude(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, const GPUContext& ctx)
{
    if (ctx.backend == GPUBackend::kSYCL) {
#if defined(MOPS_USE_SYCL) && MOPS_USE_SYCL
    sycl::queue* q = ResolveSyclQueue(ctx, "VisualizeFixedLatitude");
    if (q == nullptr) return;
    SYCLBackend::VisualizeFixedLatitude(mpasoF, config, img, *q);
#else
    Error("[GPUFactory]::VisualizeFixedLatitude SYCL backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kHIP) {
#if defined(MOPS_USE_HIP) && MOPS_USE_HIP
    HIPBackend::VisualizeFixedLatitude(mpasoF, config, img);
#else
    Error("[GPUFactory]::VisualizeFixedLatitude HIP backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kCUDA) {
#if defined(MOPS_USE_CUDA) && MOPS_USE_CUDA
    CUDABackend::VisualizeFixedLatitude(mpasoF, config, img);
#else
    Error("[GPUFactory]::VisualizeFixedLatitude CUDA backend is not enabled at build time");
#endif
    return;
    }
    Error("[GPUFactory]::VisualizeFixedLatitude backend is unsupported");
}

std::vector<TrajectoryLine> StreamLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, const GPUContext& ctx)
{
    if (ctx.backend == GPUBackend::kSYCL) {
#if defined(MOPS_USE_SYCL) && MOPS_USE_SYCL
    sycl::queue* q = ResolveSyclQueue(ctx, "StreamLine");
    if (q == nullptr) return {};
    return SYCLBackend::StreamLine(mpasoF, points, config, default_cell_id, *q);
#else
    Error("[GPUFactory]::StreamLine SYCL backend is not enabled at build time");
#endif
    return {};
    }
    if (ctx.backend == GPUBackend::kHIP) {
#if defined(MOPS_USE_HIP) && MOPS_USE_HIP
    return HIPBackend::StreamLine(mpasoF, points, config, default_cell_id);
#else
    Error("[GPUFactory]::StreamLine HIP backend is not enabled at build time");
#endif
    return {};
    }
    if (ctx.backend == GPUBackend::kCUDA) {
#if defined(MOPS_USE_CUDA) && MOPS_USE_CUDA
    return CUDABackend::StreamLine(mpasoF, points, config, default_cell_id);
#else
    Error("[GPUFactory]::StreamLine CUDA backend is not enabled at build time");
#endif
    return {};
    }
    Error("[GPUFactory]::StreamLine backend is unsupported");
    return {};
}

std::vector<TrajectoryLine> PathLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, const GPUContext& ctx)
{
    if (ctx.backend == GPUBackend::kSYCL) {
#if defined(MOPS_USE_SYCL) && MOPS_USE_SYCL
    sycl::queue* q = ResolveSyclQueue(ctx, "PathLine");
    if (q == nullptr) return {};
    return SYCLBackend::PathLine(mpasoF, points, config, default_cell_id, *q);
#else
    Error("[GPUFactory]::PathLine SYCL backend is not enabled at build time");
#endif
    return {};
    }
    if (ctx.backend == GPUBackend::kHIP) {
#if defined(MOPS_USE_HIP) && MOPS_USE_HIP
    return HIPBackend::PathLine(mpasoF, points, config, default_cell_id);
#else
    Error("[GPUFactory]::PathLine HIP backend is not enabled at build time");
#endif
    return {};
    }
    if (ctx.backend == GPUBackend::kCUDA) {
#if defined(MOPS_USE_CUDA) && MOPS_USE_CUDA
    return CUDABackend::PathLine(mpasoF, points, config, default_cell_id);
#else
    Error("[GPUFactory]::PathLine CUDA backend is not enabled at build time");
#endif
    return {};
    }
    Error("[GPUFactory]::PathLine backend is unsupported");
    return {};
}

void CalcCellVertexZtop(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_center_ztop,
    std::vector<double>& cell_vertex_ztop,
    const std::vector<size_t>& grid_info,
    const GPUContext& ctx)
{
    if (ctx.backend == GPUBackend::kSYCL) {
#if defined(MOPS_USE_SYCL) && MOPS_USE_SYCL
    sycl::queue* q = ResolveSyclQueue(ctx, "CalcCellVertexZtop");
    if (q == nullptr) return;
    SYCLBackend::CalcCellVertexZtop(grid, cells_size, total_ztop_layer, cell_center_ztop, cell_vertex_ztop, grid_info, *q);
#else
    Error("[GPUFactory]::CalcCellVertexZtop SYCL backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kHIP) {
#if defined(MOPS_USE_HIP) && MOPS_USE_HIP
    HIPBackend::CalcCellVertexZtop(grid, cells_size, total_ztop_layer, cell_center_ztop, cell_vertex_ztop, grid_info);
#else
    Error("[GPUFactory]::CalcCellVertexZtop HIP backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kCUDA) {
#if defined(MOPS_USE_CUDA) && MOPS_USE_CUDA
    CUDABackend::CalcCellVertexZtop(grid, cells_size, total_ztop_layer, cell_center_ztop, cell_vertex_ztop, grid_info);
#else
    Error("[GPUFactory]::CalcCellVertexZtop CUDA backend is not enabled at build time");
#endif
    return;
    }
    Error("[GPUFactory]::CalcCellVertexZtop backend is unsupported");
}

void CalcCellCenterToVertex(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_center_attr,
    std::vector<double>& cell_vertex_attr,
    const std::vector<size_t>& grid_info,
    const GPUContext& ctx)
{
    if (ctx.backend == GPUBackend::kSYCL) {
#if defined(MOPS_USE_SYCL) && MOPS_USE_SYCL
    sycl::queue* q = ResolveSyclQueue(ctx, "CalcCellCenterToVertex");
    if (q == nullptr) return;
    SYCLBackend::CalcCellCenterToVertex(grid, cells_size, total_ztop_layer, cell_center_attr, cell_vertex_attr, grid_info, *q);
#else
    Error("[GPUFactory]::CalcCellCenterToVertex SYCL backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kHIP) {
#if defined(MOPS_USE_HIP) && MOPS_USE_HIP
    HIPBackend::CalcCellCenterToVertex(grid, cells_size, total_ztop_layer, cell_center_attr, cell_vertex_attr, grid_info);
#else
    Error("[GPUFactory]::CalcCellCenterToVertex HIP backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kCUDA) {
#if defined(MOPS_USE_CUDA) && MOPS_USE_CUDA
    CUDABackend::CalcCellCenterToVertex(grid, cells_size, total_ztop_layer, cell_center_attr, cell_vertex_attr, grid_info);
#else
    Error("[GPUFactory]::CalcCellCenterToVertex CUDA backend is not enabled at build time");
#endif
    return;
    }
    Error("[GPUFactory]::CalcCellCenterToVertex backend is unsupported");
}

void CalcCellCenterVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_normal_velocity,
    std::vector<vec3>& cell_center_velocity,
    const std::vector<size_t>& grid_info,
    const GPUContext& ctx)
{
    if (ctx.backend == GPUBackend::kSYCL) {
#if defined(MOPS_USE_SYCL) && MOPS_USE_SYCL
    sycl::queue* q = ResolveSyclQueue(ctx, "CalcCellCenterVelocity");
    if (q == nullptr) return;
    SYCLBackend::CalcCellCenterVelocity(grid, cells_size, total_ztop_layer, cell_normal_velocity, cell_center_velocity, grid_info, *q);
#else
    Error("[GPUFactory]::CalcCellCenterVelocity SYCL backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kHIP) {
#if defined(MOPS_USE_HIP) && MOPS_USE_HIP
    HIPBackend::CalcCellCenterVelocity(grid, cells_size, total_ztop_layer, cell_normal_velocity, cell_center_velocity, grid_info);
#else
    Error("[GPUFactory]::CalcCellCenterVelocity HIP backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kCUDA) {
#if defined(MOPS_USE_CUDA) && MOPS_USE_CUDA
    CUDABackend::CalcCellCenterVelocity(grid, cells_size, total_ztop_layer, cell_normal_velocity, cell_center_velocity, grid_info);
#else
    Error("[GPUFactory]::CalcCellCenterVelocity CUDA backend is not enabled at build time");
#endif
    return;
    }
    Error("[GPUFactory]::CalcCellCenterVelocity backend is unsupported");
}

void CalcCellCenterVelocityByZM(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_zonal_velocity,
    const std::vector<double>& cell_meridional_velocity,
    std::vector<vec3>& cell_center_velocity,
    const std::vector<size_t>& grid_info,
    const GPUContext& ctx)
{
    if (ctx.backend == GPUBackend::kSYCL) {
#if defined(MOPS_USE_SYCL) && MOPS_USE_SYCL
    sycl::queue* q = ResolveSyclQueue(ctx, "CalcCellCenterVelocityByZM");
    if (q == nullptr) return;
    SYCLBackend::CalcCellCenterVelocityByZM(grid, cells_size, total_ztop_layer, cell_zonal_velocity, cell_meridional_velocity, cell_center_velocity, grid_info, *q);
#else
    Error("[GPUFactory]::CalcCellCenterVelocityByZM SYCL backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kHIP) {
#if defined(MOPS_USE_HIP) && MOPS_USE_HIP
    HIPBackend::CalcCellCenterVelocityByZM(grid, cells_size, total_ztop_layer, cell_zonal_velocity, cell_meridional_velocity, cell_center_velocity, grid_info);
#else
    Error("[GPUFactory]::CalcCellCenterVelocityByZM HIP backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kCUDA) {
#if defined(MOPS_USE_CUDA) && MOPS_USE_CUDA
    CUDABackend::CalcCellCenterVelocityByZM(grid, cells_size, total_ztop_layer, cell_zonal_velocity, cell_meridional_velocity, cell_center_velocity, grid_info);
#else
    Error("[GPUFactory]::CalcCellCenterVelocityByZM CUDA backend is not enabled at build time");
#endif
    return;
    }
    Error("[GPUFactory]::CalcCellCenterVelocityByZM backend is unsupported");
}

void CalcCellVertexVelocityByZM(
    MPASOGrid* grid,
    int vertex_size,
    int total_ztop_layer,
    const std::vector<double>& cell_vertex_zonal_velocity,
    const std::vector<double>& cell_vertex_meridional_velocity,
    std::vector<vec3>& cell_vertex_velocity,
    const std::vector<size_t>& grid_info,
    const GPUContext& ctx)
{
    if (ctx.backend == GPUBackend::kSYCL) {
#if defined(MOPS_USE_SYCL) && MOPS_USE_SYCL
    sycl::queue* q = ResolveSyclQueue(ctx, "CalcCellVertexVelocityByZM");
    if (q == nullptr) return;
    SYCLBackend::CalcCellVertexVelocityByZM(grid, vertex_size, total_ztop_layer, cell_vertex_zonal_velocity, cell_vertex_meridional_velocity, cell_vertex_velocity, grid_info, *q);
#else
    Error("[GPUFactory]::CalcCellVertexVelocityByZM SYCL backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kHIP) {
#if defined(MOPS_USE_HIP) && MOPS_USE_HIP
    HIPBackend::CalcCellVertexVelocityByZM(grid, vertex_size, total_ztop_layer, cell_vertex_zonal_velocity, cell_vertex_meridional_velocity, cell_vertex_velocity, grid_info);
#else
    Error("[GPUFactory]::CalcCellVertexVelocityByZM HIP backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kCUDA) {
#if defined(MOPS_USE_CUDA) && MOPS_USE_CUDA
    CUDABackend::CalcCellVertexVelocityByZM(grid, vertex_size, total_ztop_layer, cell_vertex_zonal_velocity, cell_vertex_meridional_velocity, cell_vertex_velocity, grid_info);
#else
    Error("[GPUFactory]::CalcCellVertexVelocityByZM CUDA backend is not enabled at build time");
#endif
    return;
    }
    Error("[GPUFactory]::CalcCellVertexVelocityByZM backend is unsupported");
}

void CalcCellVertexVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<vec3>& cell_center_velocity,
    std::vector<vec3>& cell_vertex_velocity,
    const std::vector<size_t>& grid_info,
    const GPUContext& ctx)
{
    if (ctx.backend == GPUBackend::kSYCL) {
#if defined(MOPS_USE_SYCL) && MOPS_USE_SYCL
    sycl::queue* q = ResolveSyclQueue(ctx, "CalcCellVertexVelocity");
    if (q == nullptr) return;
    SYCLBackend::CalcCellVertexVelocity(grid, cells_size, total_ztop_layer, cell_center_velocity, cell_vertex_velocity, grid_info, *q);
#else
    Error("[GPUFactory]::CalcCellVertexVelocity SYCL backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kHIP) {
#if defined(MOPS_USE_HIP) && MOPS_USE_HIP
    HIPBackend::CalcCellVertexVelocity(grid, cells_size, total_ztop_layer, cell_center_velocity, cell_vertex_velocity, grid_info);
#else
    Error("[GPUFactory]::CalcCellVertexVelocity HIP backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kCUDA) {
#if defined(MOPS_USE_CUDA) && MOPS_USE_CUDA
    CUDABackend::CalcCellVertexVelocity(grid, cells_size, total_ztop_layer, cell_center_velocity, cell_vertex_velocity, grid_info);
#else
    Error("[GPUFactory]::CalcCellVertexVelocity CUDA backend is not enabled at build time");
#endif
    return;
    }
    Error("[GPUFactory]::CalcCellVertexVelocity backend is unsupported");
}

void CalcCellVertexVertVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer_p1,
    const std::vector<double>& cell_center_vert_velocity,
    std::vector<double>& cell_vertex_vert_velocity,
    const std::vector<size_t>& grid_info,
    const GPUContext& ctx)
{
    if (ctx.backend == GPUBackend::kSYCL) {
#if defined(MOPS_USE_SYCL) && MOPS_USE_SYCL
    sycl::queue* q = ResolveSyclQueue(ctx, "CalcCellVertexVertVelocity");
    if (q == nullptr) return;
    SYCLBackend::CalcCellVertexVertVelocity(grid, cells_size, total_ztop_layer_p1, cell_center_vert_velocity, cell_vertex_vert_velocity, grid_info, *q);
#else
    Error("[GPUFactory]::CalcCellVertexVertVelocity SYCL backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kHIP) {
#if defined(MOPS_USE_HIP) && MOPS_USE_HIP
    HIPBackend::CalcCellVertexVertVelocity(grid, cells_size, total_ztop_layer_p1, cell_center_vert_velocity, cell_vertex_vert_velocity, grid_info);
#else
    Error("[GPUFactory]::CalcCellVertexVertVelocity HIP backend is not enabled at build time");
#endif
    return;
    }
    if (ctx.backend == GPUBackend::kCUDA) {
#if defined(MOPS_USE_CUDA) && MOPS_USE_CUDA
    CUDABackend::CalcCellVertexVertVelocity(grid, cells_size, total_ztop_layer_p1, cell_center_vert_velocity, cell_vertex_vert_velocity, grid_info);
#else
    Error("[GPUFactory]::CalcCellVertexVertVelocity CUDA backend is not enabled at build time");
#endif
    return;
    }
    Error("[GPUFactory]::CalcCellVertexVertVelocity backend is unsupported");
}

} // namespace MOPS::GPU::Factory
