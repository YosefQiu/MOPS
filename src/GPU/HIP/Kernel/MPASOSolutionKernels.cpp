#include "GPU/HIP/Kernel/MPASOSolutionKernels.h"

namespace {

inline bool IsGridInfoValid(const std::vector<size_t>& grid_info)
{
    return grid_info.size() >= 6;
}

} // namespace

namespace MOPS::GPU::HIPBackend::Kernel::HIPImpl {

void LaunchCalcCellVertexZtop(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_center_ztop,
    std::vector<double>& cell_vertex_ztop,
    const std::vector<size_t>& grid_info);

void LaunchCalcCellCenterToVertex(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_center_attr,
    std::vector<double>& cell_vertex_attr,
    const std::vector<size_t>& grid_info);

void LaunchCalcCellCenterVelocityByZM(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_zonal_velocity,
    const std::vector<double>& cell_meridional_velocity,
    std::vector<vec3>& cell_center_velocity,
    const std::vector<size_t>& grid_info);

void LaunchCalcCellVertexVelocityByZM(
    MPASOGrid* grid,
    int vertex_size,
    int total_ztop_layer,
    const std::vector<double>& cell_vertex_zonal_velocity,
    const std::vector<double>& cell_vertex_meridional_velocity,
    std::vector<vec3>& cell_vertex_velocity,
    const std::vector<size_t>& grid_info);

void LaunchCalcCellCenterVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_normal_velocity,
    std::vector<vec3>& cell_center_velocity,
    const std::vector<size_t>& grid_info);

void LaunchCalcCellVertexVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<vec3>& cell_center_velocity,
    std::vector<vec3>& cell_vertex_velocity,
    const std::vector<size_t>& grid_info);

void LaunchCalcCellVertexVertVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer_p1,
    const std::vector<double>& cell_center_vert_velocity,
    std::vector<double>& cell_vertex_vert_velocity,
    const std::vector<size_t>& grid_info);

} // namespace MOPS::GPU::HIPBackend::Kernel::HIPImpl

namespace MOPS::GPU::HIPBackend::Kernel {

void CalcCellVertexZtop(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_center_ztop,
    std::vector<double>& cell_vertex_ztop,
    const std::vector<size_t>& grid_info)
{
    if (grid == nullptr) {
        Error("[HIPBackend::Kernel]::CalcCellVertexZtop grid is null");
    }
    if (!IsGridInfoValid(grid_info)) {
        Error("[HIPBackend::Kernel]::CalcCellVertexZtop grid_info is invalid");
    }
    HIPImpl::LaunchCalcCellVertexZtop(grid, cells_size, total_ztop_layer, cell_center_ztop, cell_vertex_ztop, grid_info);
}

void CalcCellCenterToVertex(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_center_attr,
    std::vector<double>& cell_vertex_attr,
    const std::vector<size_t>& grid_info)
{
    if (grid == nullptr) {
        Error("[HIPBackend::Kernel]::CalcCellCenterToVertex grid is null");
    }
    if (!IsGridInfoValid(grid_info)) {
        Error("[HIPBackend::Kernel]::CalcCellCenterToVertex grid_info is invalid");
    }
    HIPImpl::LaunchCalcCellCenterToVertex(grid, cells_size, total_ztop_layer, cell_center_attr, cell_vertex_attr, grid_info);
}

void CalcCellCenterVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_normal_velocity,
    std::vector<vec3>& cell_center_velocity,
    const std::vector<size_t>& grid_info)
{
    if (grid == nullptr) {
        Error("[HIPBackend::Kernel]::CalcCellCenterVelocity grid is null");
    }
    if (!IsGridInfoValid(grid_info)) {
        Error("[HIPBackend::Kernel]::CalcCellCenterVelocity grid_info is invalid");
    }
    HIPImpl::LaunchCalcCellCenterVelocity(
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
    if (grid == nullptr) {
        Error("[HIPBackend::Kernel]::CalcCellCenterVelocityByZM grid is null");
    }
    if (!IsGridInfoValid(grid_info)) {
        Error("[HIPBackend::Kernel]::CalcCellCenterVelocityByZM grid_info is invalid");
    }
    HIPImpl::LaunchCalcCellCenterVelocityByZM(
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
    if (grid == nullptr) {
        Error("[HIPBackend::Kernel]::CalcCellVertexVelocityByZM grid is null");
    }
    if (!IsGridInfoValid(grid_info)) {
        Error("[HIPBackend::Kernel]::CalcCellVertexVelocityByZM grid_info is invalid");
    }
    HIPImpl::LaunchCalcCellVertexVelocityByZM(
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
    if (grid == nullptr) {
        Error("[HIPBackend::Kernel]::CalcCellVertexVelocity grid is null");
    }
    if (!IsGridInfoValid(grid_info)) {
        Error("[HIPBackend::Kernel]::CalcCellVertexVelocity grid_info is invalid");
    }
    HIPImpl::LaunchCalcCellVertexVelocity(
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
    if (grid == nullptr) {
        Error("[HIPBackend::Kernel]::CalcCellVertexVertVelocity grid is null");
    }
    if (!IsGridInfoValid(grid_info)) {
        Error("[HIPBackend::Kernel]::CalcCellVertexVertVelocity grid_info is invalid");
    }
    HIPImpl::LaunchCalcCellVertexVertVelocity(
        grid,
        cells_size,
        total_ztop_layer_p1,
        cell_center_vert_velocity,
        cell_vertex_vert_velocity,
        grid_info);
}

} // namespace MOPS::GPU::HIPBackend::Kernel
