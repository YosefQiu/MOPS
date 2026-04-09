#include "GPU/HIP/MPASOSolutionHIP.h"
#include "GPU/HIP/Kernel/MPASOSolutionKernels.h"

namespace MOPS::GPU::HIPBackend {

void CalcCellVertexZtop(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_center_ztop,
    std::vector<double>& cell_vertex_ztop,
    const std::vector<size_t>& grid_info)
{
    Kernel::CalcCellVertexZtop(
        grid,
        cells_size,
        total_ztop_layer,
        cell_center_ztop,
        cell_vertex_ztop,
        grid_info);
}

void CalcCellCenterToVertex(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_center_attr,
    std::vector<double>& cell_vertex_attr,
    const std::vector<size_t>& grid_info)
{
    Kernel::CalcCellCenterToVertex(
        grid,
        cells_size,
        total_ztop_layer,
        cell_center_attr,
        cell_vertex_attr,
        grid_info);
}

void CalcCellCenterVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_normal_velocity,
    std::vector<vec3>& cell_center_velocity,
    const std::vector<size_t>& grid_info)
{
    Kernel::CalcCellCenterVelocity(
        grid,
        cells_size,
        total_ztop_layer,
        cell_normal_velocity,
        cell_center_velocity,
        grid_info);
}

void CalcCellCenterVelocityByZM(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_zonal_velocity,
    const std::vector<double>& cell_meridional_velocity,
    std::vector<vec3>& cell_center_velocity,
    const std::vector<size_t>& grid_info)
{
    Kernel::CalcCellCenterVelocityByZM(
        grid,
        cells_size,
        total_ztop_layer,
        cell_zonal_velocity,
        cell_meridional_velocity,
        cell_center_velocity,
        grid_info);
}

void CalcCellVertexVelocityByZM(
    MPASOGrid* grid,
    int vertex_size,
    int total_ztop_layer,
    const std::vector<double>& cell_vertex_zonal_velocity,
    const std::vector<double>& cell_vertex_meridional_velocity,
    std::vector<vec3>& cell_vertex_velocity,
    const std::vector<size_t>& grid_info)
{
    Kernel::CalcCellVertexVelocityByZM(
        grid,
        vertex_size,
        total_ztop_layer,
        cell_vertex_zonal_velocity,
        cell_vertex_meridional_velocity,
        cell_vertex_velocity,
        grid_info);
}

void CalcCellVertexVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<vec3>& cell_center_velocity,
    std::vector<vec3>& cell_vertex_velocity,
    const std::vector<size_t>& grid_info)
{
    Kernel::CalcCellVertexVelocity(
        grid,
        cells_size,
        total_ztop_layer,
        cell_center_velocity,
        cell_vertex_velocity,
        grid_info);
}

void CalcCellVertexVertVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer_p1,
    const std::vector<double>& cell_center_vert_velocity,
    std::vector<double>& cell_vertex_vert_velocity,
    const std::vector<size_t>& grid_info)
{
    Kernel::CalcCellVertexVertVelocity(
        grid,
        cells_size,
        total_ztop_layer_p1,
        cell_center_vert_velocity,
        cell_vertex_vert_velocity,
        grid_info);
}

} // namespace MOPS::GPU::HIPBackend
