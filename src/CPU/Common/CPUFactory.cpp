#include "CPU/Common/CPUFactory.h"

#if defined(MOPS_USE_TBB) && MOPS_USE_TBB
#include "CPU/TBB/MPASOSolutionTBB.h"
#include "CPU/TBB/MPASOVisualizerTBB.h"
#endif

namespace MOPS::CPU::Factory {

void VisualizeFixedLayer(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, const CPUContext& ctx)
{
    if (ctx.backend == CPUBackend::kTBB) {
#if defined(MOPS_USE_TBB) && MOPS_USE_TBB
        MOPS::CPU::TBBBackend::VisualizeFixedLayer(mpasoF, config, img);
#else
        Error("[CPUFactory]::VisualizeFixedLayer TBB backend is not enabled at build time");
#endif
        return;
    }
    Error("[CPUFactory]::VisualizeFixedLayer backend is unsupported");
}

void VisualizeFixedDepth(MPASOField* mpasoF, VisualizationSettings* config, std::vector<ImageBuffer<double>>& img_vec, const CPUContext& ctx)
{
    if (ctx.backend == CPUBackend::kTBB) {
#if defined(MOPS_USE_TBB) && MOPS_USE_TBB
        MOPS::CPU::TBBBackend::VisualizeFixedDepth(mpasoF, config, img_vec);
#else
        Error("[CPUFactory]::VisualizeFixedDepth TBB backend is not enabled at build time");
#endif
        return;
    }
    Error("[CPUFactory]::VisualizeFixedDepth backend is unsupported");
}

void VisualizeFixedLatitude(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, const CPUContext& ctx)
{
    if (ctx.backend == CPUBackend::kTBB) {
#if defined(MOPS_USE_TBB) && MOPS_USE_TBB
        MOPS::CPU::TBBBackend::VisualizeFixedLatitude(mpasoF, config, img);
#else
        Error("[CPUFactory]::VisualizeFixedLatitude TBB backend is not enabled at build time");
#endif
        return;
    }
    Error("[CPUFactory]::VisualizeFixedLatitude backend is unsupported");
}

std::vector<TrajectoryLine> StreamLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, const CPUContext& ctx)
{
    if (ctx.backend == CPUBackend::kTBB) {
#if defined(MOPS_USE_TBB) && MOPS_USE_TBB
        return MOPS::CPU::TBBBackend::StreamLine(mpasoF, points, config, default_cell_id);
#else
        Error("[CPUFactory]::StreamLine TBB backend is not enabled at build time");
#endif
        return {};
    }
    Error("[CPUFactory]::StreamLine backend is unsupported");
    return {};
}

std::vector<TrajectoryLine> PathLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, const CPUContext& ctx)
{
    if (ctx.backend == CPUBackend::kTBB) {
#if defined(MOPS_USE_TBB) && MOPS_USE_TBB
        return MOPS::CPU::TBBBackend::PathLine(mpasoF, points, config, default_cell_id);
#else
        Error("[CPUFactory]::PathLine TBB backend is not enabled at build time");
#endif
        return {};
    }
    Error("[CPUFactory]::PathLine backend is unsupported");
    return {};
}

void CalcCellVertexZtop(MPASOGrid* grid, int cells_size, int total_ztop_layer, const std::vector<double>& cell_center_ztop, std::vector<double>& cell_vertex_ztop, const std::vector<size_t>& grid_info, const CPUContext& ctx)
{
    if (ctx.backend == CPUBackend::kTBB) {
#if defined(MOPS_USE_TBB) && MOPS_USE_TBB
        MOPS::CPU::TBBBackend::CalcCellVertexZtop(grid, cells_size, total_ztop_layer, cell_center_ztop, cell_vertex_ztop, grid_info);
#else
        Error("[CPUFactory]::CalcCellVertexZtop TBB backend is not enabled at build time");
#endif
        return;
    }
    Error("[CPUFactory]::CalcCellVertexZtop backend is unsupported");
}

void CalcCellCenterToVertex(MPASOGrid* grid, int cells_size, int total_ztop_layer, const std::vector<double>& cell_center_attr, std::vector<double>& cell_vertex_attr, const std::vector<size_t>& grid_info, const CPUContext& ctx)
{
    if (ctx.backend == CPUBackend::kTBB) {
#if defined(MOPS_USE_TBB) && MOPS_USE_TBB
        MOPS::CPU::TBBBackend::CalcCellCenterToVertex(grid, cells_size, total_ztop_layer, cell_center_attr, cell_vertex_attr, grid_info);
#else
        Error("[CPUFactory]::CalcCellCenterToVertex TBB backend is not enabled at build time");
#endif
        return;
    }
    Error("[CPUFactory]::CalcCellCenterToVertex backend is unsupported");
}

void CalcCellCenterVelocity(MPASOGrid* grid, int cells_size, int total_ztop_layer, const std::vector<double>& cell_normal_velocity, std::vector<vec3>& cell_center_velocity, const std::vector<size_t>& grid_info, const CPUContext& ctx)
{
    if (ctx.backend == CPUBackend::kTBB) {
#if defined(MOPS_USE_TBB) && MOPS_USE_TBB
        MOPS::CPU::TBBBackend::CalcCellCenterVelocity(grid, cells_size, total_ztop_layer, cell_normal_velocity, cell_center_velocity, grid_info);
#else
        Error("[CPUFactory]::CalcCellCenterVelocity TBB backend is not enabled at build time");
#endif
        return;
    }
    Error("[CPUFactory]::CalcCellCenterVelocity backend is unsupported");
}

void CalcCellCenterVelocityByZM(MPASOGrid* grid, int cells_size, int total_ztop_layer, const std::vector<double>& cell_zonal_velocity, const std::vector<double>& cell_meridional_velocity, std::vector<vec3>& cell_center_velocity, const std::vector<size_t>& grid_info, const CPUContext& ctx)
{
    if (ctx.backend == CPUBackend::kTBB) {
#if defined(MOPS_USE_TBB) && MOPS_USE_TBB
        MOPS::CPU::TBBBackend::CalcCellCenterVelocityByZM(grid, cells_size, total_ztop_layer, cell_zonal_velocity, cell_meridional_velocity, cell_center_velocity, grid_info);
#else
        Error("[CPUFactory]::CalcCellCenterVelocityByZM TBB backend is not enabled at build time");
#endif
        return;
    }
    Error("[CPUFactory]::CalcCellCenterVelocityByZM backend is unsupported");
}

void CalcCellVertexVelocityByZM(MPASOGrid* grid, int vertex_size, int total_ztop_layer, const std::vector<double>& cell_vertex_zonal_velocity, const std::vector<double>& cell_vertex_meridional_velocity, std::vector<vec3>& cell_vertex_velocity, const std::vector<size_t>& grid_info, const CPUContext& ctx)
{
    if (ctx.backend == CPUBackend::kTBB) {
#if defined(MOPS_USE_TBB) && MOPS_USE_TBB
        MOPS::CPU::TBBBackend::CalcCellVertexVelocityByZM(grid, vertex_size, total_ztop_layer, cell_vertex_zonal_velocity, cell_vertex_meridional_velocity, cell_vertex_velocity, grid_info);
#else
        Error("[CPUFactory]::CalcCellVertexVelocityByZM TBB backend is not enabled at build time");
#endif
        return;
    }
    Error("[CPUFactory]::CalcCellVertexVelocityByZM backend is unsupported");
}

void CalcCellVertexVelocity(MPASOGrid* grid, int cells_size, int total_ztop_layer, const std::vector<vec3>& cell_center_velocity, std::vector<vec3>& cell_vertex_velocity, const std::vector<size_t>& grid_info, const CPUContext& ctx)
{
    if (ctx.backend == CPUBackend::kTBB) {
#if defined(MOPS_USE_TBB) && MOPS_USE_TBB
        MOPS::CPU::TBBBackend::CalcCellVertexVelocity(grid, cells_size, total_ztop_layer, cell_center_velocity, cell_vertex_velocity, grid_info);
#else
        Error("[CPUFactory]::CalcCellVertexVelocity TBB backend is not enabled at build time");
#endif
        return;
    }
    Error("[CPUFactory]::CalcCellVertexVelocity backend is unsupported");
}

void CalcCellVertexVertVelocity(MPASOGrid* grid, int cells_size, int total_ztop_layer_p1, const std::vector<double>& cell_center_vert_velocity, std::vector<double>& cell_vertex_vert_velocity, const std::vector<size_t>& grid_info, const CPUContext& ctx)
{
    if (ctx.backend == CPUBackend::kTBB) {
#if defined(MOPS_USE_TBB) && MOPS_USE_TBB
        MOPS::CPU::TBBBackend::CalcCellVertexVertVelocity(grid, cells_size, total_ztop_layer_p1, cell_center_vert_velocity, cell_vertex_vert_velocity, grid_info);
#else
        Error("[CPUFactory]::CalcCellVertexVertVelocity TBB backend is not enabled at build time");
#endif
        return;
    }
    Error("[CPUFactory]::CalcCellVertexVertVelocity backend is unsupported");
}

} // namespace MOPS::CPU::Factory
