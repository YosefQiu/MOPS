#include "Common/MOPSFactory.h"

#include "CPU/Common/CPUFactory.h"
#include "GPU/Common/GPUFactory.h"

namespace MOPS::Factory {

void VisualizeFixedLayer(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, const RuntimeContext& ctx)
{
    if (ctx.kind == RuntimeKind::kGPU) {
        MOPS::GPU::Factory::VisualizeFixedLayer(mpasoF, config, img, ctx.gpu);
        return;
    }
    if (ctx.kind == RuntimeKind::kCPU) {
        MOPS::CPU::Factory::VisualizeFixedLayer(mpasoF, config, img, ctx.cpu);
        return;
    }
    Error("[MOPSFactory]::VisualizeFixedLayer runtime kind is unsupported");
}

void VisualizeFixedDepth(MPASOField* mpasoF, VisualizationSettings* config, std::vector<ImageBuffer<double>>& img_vec, const RuntimeContext& ctx)
{
    if (ctx.kind == RuntimeKind::kGPU) {
        MOPS::GPU::Factory::VisualizeFixedDepth(mpasoF, config, img_vec, ctx.gpu);
        return;
    }
    if (ctx.kind == RuntimeKind::kCPU) {
        MOPS::CPU::Factory::VisualizeFixedDepth(mpasoF, config, img_vec, ctx.cpu);
        return;
    }
    Error("[MOPSFactory]::VisualizeFixedDepth runtime kind is unsupported");
}

void VisualizeFixedLatitude(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, const RuntimeContext& ctx)
{
    if (ctx.kind == RuntimeKind::kGPU) {
        MOPS::GPU::Factory::VisualizeFixedLatitude(mpasoF, config, img, ctx.gpu);
        return;
    }
    if (ctx.kind == RuntimeKind::kCPU) {
        MOPS::CPU::Factory::VisualizeFixedLatitude(mpasoF, config, img, ctx.cpu);
        return;
    }
    Error("[MOPSFactory]::VisualizeFixedLatitude runtime kind is unsupported");
}

std::vector<TrajectoryLine> StreamLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, const RuntimeContext& ctx)
{
    if (ctx.kind == RuntimeKind::kGPU) {
        return MOPS::GPU::Factory::StreamLine(mpasoF, points, config, default_cell_id, ctx.gpu);
    }
    if (ctx.kind == RuntimeKind::kCPU) {
        return MOPS::CPU::Factory::StreamLine(mpasoF, points, config, default_cell_id, ctx.cpu);
    }
    Error("[MOPSFactory]::StreamLine runtime kind is unsupported");
    return {};
}

std::vector<TrajectoryLine> PathLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, const RuntimeContext& ctx)
{
    if (ctx.kind == RuntimeKind::kGPU) {
        return MOPS::GPU::Factory::PathLine(mpasoF, points, config, default_cell_id, ctx.gpu);
    }
    if (ctx.kind == RuntimeKind::kCPU) {
        return MOPS::CPU::Factory::PathLine(mpasoF, points, config, default_cell_id, ctx.cpu);
    }
    Error("[MOPSFactory]::PathLine runtime kind is unsupported");
    return {};
}

void CalcCellVertexZtop(MPASOGrid* grid, int cells_size, int total_ztop_layer, const std::vector<double>& cell_center_ztop, std::vector<double>& cell_vertex_ztop, const std::vector<size_t>& grid_info, const RuntimeContext& ctx)
{
    if (ctx.kind == RuntimeKind::kGPU) {
        MOPS::GPU::Factory::CalcCellVertexZtop(grid, cells_size, total_ztop_layer, cell_center_ztop, cell_vertex_ztop, grid_info, ctx.gpu);
        return;
    }
    if (ctx.kind == RuntimeKind::kCPU) {
        MOPS::CPU::Factory::CalcCellVertexZtop(grid, cells_size, total_ztop_layer, cell_center_ztop, cell_vertex_ztop, grid_info, ctx.cpu);
        return;
    }
    Error("[MOPSFactory]::CalcCellVertexZtop runtime kind is unsupported");
}

void CalcCellCenterToVertex(MPASOGrid* grid, int cells_size, int total_ztop_layer, const std::vector<double>& cell_center_attr, std::vector<double>& cell_vertex_attr, const std::vector<size_t>& grid_info, const RuntimeContext& ctx)
{
    if (ctx.kind == RuntimeKind::kGPU) {
        MOPS::GPU::Factory::CalcCellCenterToVertex(grid, cells_size, total_ztop_layer, cell_center_attr, cell_vertex_attr, grid_info, ctx.gpu);
        return;
    }
    if (ctx.kind == RuntimeKind::kCPU) {
        MOPS::CPU::Factory::CalcCellCenterToVertex(grid, cells_size, total_ztop_layer, cell_center_attr, cell_vertex_attr, grid_info, ctx.cpu);
        return;
    }
    Error("[MOPSFactory]::CalcCellCenterToVertex runtime kind is unsupported");
}

void CalcCellCenterVelocity(MPASOGrid* grid, int cells_size, int total_ztop_layer, const std::vector<double>& cell_normal_velocity, std::vector<vec3>& cell_center_velocity, const std::vector<size_t>& grid_info, const RuntimeContext& ctx)
{
    if (ctx.kind == RuntimeKind::kGPU) {
        MOPS::GPU::Factory::CalcCellCenterVelocity(grid, cells_size, total_ztop_layer, cell_normal_velocity, cell_center_velocity, grid_info, ctx.gpu);
        return;
    }
    if (ctx.kind == RuntimeKind::kCPU) {
        MOPS::CPU::Factory::CalcCellCenterVelocity(grid, cells_size, total_ztop_layer, cell_normal_velocity, cell_center_velocity, grid_info, ctx.cpu);
        return;
    }
    Error("[MOPSFactory]::CalcCellCenterVelocity runtime kind is unsupported");
}

void CalcCellCenterVelocityByZM(MPASOGrid* grid, int cells_size, int total_ztop_layer, const std::vector<double>& cell_zonal_velocity, const std::vector<double>& cell_meridional_velocity, std::vector<vec3>& cell_center_velocity, const std::vector<size_t>& grid_info, const RuntimeContext& ctx)
{
    if (ctx.kind == RuntimeKind::kGPU) {
        MOPS::GPU::Factory::CalcCellCenterVelocityByZM(grid, cells_size, total_ztop_layer, cell_zonal_velocity, cell_meridional_velocity, cell_center_velocity, grid_info, ctx.gpu);
        return;
    }
    if (ctx.kind == RuntimeKind::kCPU) {
        MOPS::CPU::Factory::CalcCellCenterVelocityByZM(grid, cells_size, total_ztop_layer, cell_zonal_velocity, cell_meridional_velocity, cell_center_velocity, grid_info, ctx.cpu);
        return;
    }
    Error("[MOPSFactory]::CalcCellCenterVelocityByZM runtime kind is unsupported");
}

void CalcCellVertexVelocityByZM(MPASOGrid* grid, int vertex_size, int total_ztop_layer, const std::vector<double>& cell_vertex_zonal_velocity, const std::vector<double>& cell_vertex_meridional_velocity, std::vector<vec3>& cell_vertex_velocity, const std::vector<size_t>& grid_info, const RuntimeContext& ctx)
{
    if (ctx.kind == RuntimeKind::kGPU) {
        MOPS::GPU::Factory::CalcCellVertexVelocityByZM(grid, vertex_size, total_ztop_layer, cell_vertex_zonal_velocity, cell_vertex_meridional_velocity, cell_vertex_velocity, grid_info, ctx.gpu);
        return;
    }
    if (ctx.kind == RuntimeKind::kCPU) {
        MOPS::CPU::Factory::CalcCellVertexVelocityByZM(grid, vertex_size, total_ztop_layer, cell_vertex_zonal_velocity, cell_vertex_meridional_velocity, cell_vertex_velocity, grid_info, ctx.cpu);
        return;
    }
    Error("[MOPSFactory]::CalcCellVertexVelocityByZM runtime kind is unsupported");
}

void CalcCellVertexVelocity(MPASOGrid* grid, int cells_size, int total_ztop_layer, const std::vector<vec3>& cell_center_velocity, std::vector<vec3>& cell_vertex_velocity, const std::vector<size_t>& grid_info, const RuntimeContext& ctx)
{
    if (ctx.kind == RuntimeKind::kGPU) {
        MOPS::GPU::Factory::CalcCellVertexVelocity(grid, cells_size, total_ztop_layer, cell_center_velocity, cell_vertex_velocity, grid_info, ctx.gpu);
        return;
    }
    if (ctx.kind == RuntimeKind::kCPU) {
        MOPS::CPU::Factory::CalcCellVertexVelocity(grid, cells_size, total_ztop_layer, cell_center_velocity, cell_vertex_velocity, grid_info, ctx.cpu);
        return;
    }
    Error("[MOPSFactory]::CalcCellVertexVelocity runtime kind is unsupported");
}

void CalcCellVertexVertVelocity(MPASOGrid* grid, int cells_size, int total_ztop_layer_p1, const std::vector<double>& cell_center_vert_velocity, std::vector<double>& cell_vertex_vert_velocity, const std::vector<size_t>& grid_info, const RuntimeContext& ctx)
{
    if (ctx.kind == RuntimeKind::kGPU) {
        MOPS::GPU::Factory::CalcCellVertexVertVelocity(grid, cells_size, total_ztop_layer_p1, cell_center_vert_velocity, cell_vertex_vert_velocity, grid_info, ctx.gpu);
        return;
    }
    if (ctx.kind == RuntimeKind::kCPU) {
        MOPS::CPU::Factory::CalcCellVertexVertVelocity(grid, cells_size, total_ztop_layer_p1, cell_center_vert_velocity, cell_vertex_vert_velocity, grid_info, ctx.cpu);
        return;
    }
    Error("[MOPSFactory]::CalcCellVertexVertVelocity runtime kind is unsupported");
}

} // namespace MOPS::Factory
