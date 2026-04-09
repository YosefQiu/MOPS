#include "GPU/HIP/Kernel/MPASOSolutionKernels.h"

#include <hip/hip_runtime.h>

#include "GPU/HIP/Kernel/HIPKernel.h"
#include "Utils/GeoConverter.hpp"
#include "Utils/Interpolation.hpp"

namespace {

inline void CheckHip(hipError_t code, const char* what)
{
    if (code != hipSuccess) {
        Error("[CUDA]::%s failed: %s", what, hipGetErrorString(code));
    }
}

__global__ void KernelCalcCellVertexZtop(
    const vec3* vertex_coord,
    const vec3* cell_coord,
    const size_t* cells_on_vertex,
    size_t cells_size,
    size_t vertex_size,
    size_t total_layer,
    const double* cell_center_ztop,
    double* cell_vertex_ztop)
{
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = vertex_size * total_layer;
    if (tid >= total) {
        return;
    }

    const size_t vertex_idx = tid / total_layer;
    const size_t layer = tid % total_layer;

    constexpr int kNeighborNum = 3;
    size_t neighbor_cells[kNeighborNum] = {0, 0, 0};
    bool boundary = false;

    for (int i = 0; i < kNeighborNum; ++i) {
        const size_t raw_id = cells_on_vertex[vertex_idx * kNeighborNum + i];
        if (raw_id == 0) {
            boundary = true;
            continue;
        }
        const size_t cid = raw_id - 1;
        if (cid >= cells_size) {
            boundary = true;
            continue;
        }
        neighbor_cells[i] = cid;
    }

    if (boundary) {
        cell_vertex_ztop[tid] = 0.0;
        return;
    }

    const vec3 p = vertex_coord[vertex_idx];
    const vec3 p1 = cell_coord[neighbor_cells[0]];
    const vec3 p2 = cell_coord[neighbor_cells[1]];
    const vec3 p3 = cell_coord[neighbor_cells[2]];

    MOPS::Interpolator::TRIANGLE tri(p1, p2, p3);
    double u = 0.0;
    double v = 0.0;
    double w = 0.0;
    MOPS::Interpolator::calcTriangleBarycentric(p, &tri, u, v, w);

    const double s1 = cell_center_ztop[neighbor_cells[0] * total_layer + layer];
    const double s2 = cell_center_ztop[neighbor_cells[1] * total_layer + layer];
    const double s3 = cell_center_ztop[neighbor_cells[2] * total_layer + layer];
    cell_vertex_ztop[tid] = u * s1 + v * s2 + w * s3;
}

__global__ void KernelCalcCellCenterToVertex(
    const vec3* vertex_coord,
    const vec3* cell_coord,
    const size_t* cells_on_vertex,
    size_t cells_size,
    size_t vertex_size,
    size_t total_layer,
    const double* cell_center_attr,
    double* cell_vertex_attr)
{
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = vertex_size * total_layer;
    if (tid >= total) {
        return;
    }

    const size_t vertex_idx = tid / total_layer;
    const size_t layer = tid % total_layer;

    constexpr int kNeighborNum = 3;
    size_t neighbor_cells[kNeighborNum] = {0, 0, 0};
    bool boundary = false;

    for (int i = 0; i < kNeighborNum; ++i) {
        const size_t raw_id = cells_on_vertex[vertex_idx * kNeighborNum + i];
        if (raw_id == 0) {
            boundary = true;
            continue;
        }
        const size_t cid = raw_id - 1;
        if (cid >= cells_size) {
            boundary = true;
            continue;
        }
        neighbor_cells[i] = cid;
    }

    if (boundary) {
        cell_vertex_attr[tid] = 0.0;
        return;
    }

    const vec3 p = vertex_coord[vertex_idx];
    const vec3 p1 = cell_coord[neighbor_cells[0]];
    const vec3 p2 = cell_coord[neighbor_cells[1]];
    const vec3 p3 = cell_coord[neighbor_cells[2]];

    MOPS::Interpolator::TRIANGLE tri(p1, p2, p3);
    double u = 0.0;
    double v = 0.0;
    double w = 0.0;
    MOPS::Interpolator::calcTriangleBarycentric(p, &tri, u, v, w);

    const double s1 = cell_center_attr[neighbor_cells[0] * total_layer + layer];
    const double s2 = cell_center_attr[neighbor_cells[1] * total_layer + layer];
    const double s3 = cell_center_attr[neighbor_cells[2] * total_layer + layer];

    double value = u * s1 + v * s2 + w * s3;
    if (value < 0.0) {
        value = 0.0;
    }
    cell_vertex_attr[tid] = value;
}

__global__ void KernelCalcCellCenterVelocityByZM(
    const vec3* cell_coord,
    size_t cells_size,
    size_t total_layer,
    const double* cell_zonal_velocity,
    const double* cell_meridional_velocity,
    vec3* cell_center_velocity)
{
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = cells_size * total_layer;
    if (tid >= total) {
        return;
    }

    const size_t cell_id = tid / total_layer;
    const vec3 cell_position = cell_coord[cell_id];

    vec3 velocity = {0.0, 0.0, 0.0};
    MOPS::GeoConverter::convertENUVelocityToXYZ(
        cell_position,
        cell_zonal_velocity[tid],
        cell_meridional_velocity[tid],
        0.0,
        velocity);
    cell_center_velocity[tid] = velocity;
}

__global__ void KernelCalcCellVertexVelocityByZM(
    const vec3* vertex_coord,
    size_t vertex_size,
    size_t total_layer,
    const double* vertex_zonal_velocity,
    const double* vertex_meridional_velocity,
    vec3* vertex_velocity)
{
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = vertex_size * total_layer;
    if (tid >= total) {
        return;
    }

    const size_t vertex_id = tid / total_layer;
    const vec3 vertex_position = vertex_coord[vertex_id];

    vec3 velocity = {0.0, 0.0, 0.0};
    MOPS::GeoConverter::convertENUVelocityToXYZ(
        vertex_position,
        vertex_zonal_velocity[tid],
        vertex_meridional_velocity[tid],
        0.0,
        velocity);
    vertex_velocity[tid] = velocity;
}

__global__ void KernelCalcCellCenterVelocity(
    const vec3* cell_coord,
    const vec3* edge_coord,
    const size_t* number_vertex_on_cell,
    const size_t* edges_on_cell,
    const size_t* cells_on_edge,
    size_t cells_size,
    size_t total_layer,
    const double* cell_normal_velocity,
    vec3* cell_center_velocity)
{
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = cells_size * total_layer;
    if (tid >= total) {
        return;
    }

    constexpr int kMaxVertexNum = 7;
    const size_t cell_id = tid / total_layer;
    const size_t layer = tid % total_layer;

    const vec3 cell_position = cell_coord[cell_id];
    size_t current_cell_edges_id[kMaxVertexNum];
    const size_t invalid = static_cast<size_t>(-1);

    const size_t current_cell_vertices_number = number_vertex_on_cell[cell_id];
    for (int k = 0; k < kMaxVertexNum; ++k) {
        current_cell_edges_id[k] = invalid;
    }
    for (size_t k = 0; k < current_cell_vertices_number && k < static_cast<size_t>(kMaxVertexNum); ++k) {
        const size_t raw = edges_on_cell[cell_id * kMaxVertexNum + k];
        current_cell_edges_id[k] = (raw == 0) ? invalid : (raw - 1);
    }

    vec3 up = cell_position / MOPS_LENGTH(cell_position);
    vec3 east = MOPS_CROSS(make_double3(0.0, 0.0, 1.0), up);
    if (MOPS_LENGTH(east) < 1e-6) {
        east = MOPS_CROSS(make_double3(0.0, 1.0, 0.0), up);
    }
    east = east / MOPS_LENGTH(east);
    const vec3 north = MOPS_CROSS(up, east);

    double plane_basis_vector[2][3] = {
        {east.x(), east.y(), east.z()},
        {north.x(), north.y(), north.z()}
    };
    double cell_center[3] = {cell_position.x(), cell_position.y(), cell_position.z()};
    double edge_center[kMaxVertexNum][3] = {{0.0}};
    double unit_vector[kMaxVertexNum][3] = {{0.0}};
    double normal_vel[kMaxVertexNum][1] = {{0.0}};
    double coeffs[kMaxVertexNum][3] = {{0.0}};

    for (int kidx = 0; kidx < kMaxVertexNum; ++kidx) {
        const size_t edge_id = current_cell_edges_id[kidx];
        if (edge_id == invalid) {
            continue;
        }

        const vec3 edge_position = edge_coord[edge_id];
        edge_center[kidx][0] = edge_position.x();
        edge_center[kidx][1] = edge_position.y();
        edge_center[kidx][2] = edge_position.z();

        const size_t cell0_raw = cells_on_edge[edge_id * 2 + 0];
        const size_t cell1_raw = cells_on_edge[edge_id * 2 + 1];
        if (cell0_raw == 0 || cell1_raw == 0) {
            continue;
        }

        const size_t cell0 = cell0_raw - 1;
        const size_t cell1 = cell1_raw - 1;
        const size_t min_cell_id = (cell0 < cell1) ? cell0 : cell1;
        const size_t max_cell_id = (cell0 > cell1) ? cell0 : cell1;

        vec3 normal_vector;
        if (max_cell_id > cells_size) {
            const vec3 min_cell_position = cell_coord[min_cell_id];
            normal_vector = edge_position - min_cell_position;
        } else {
            const vec3 min_cell_position = cell_coord[min_cell_id];
            const vec3 max_cell_position = cell_coord[max_cell_id];
            normal_vector = max_cell_position - min_cell_position;
        }

        const double length = MOPS_LENGTH(normal_vector);
        if (length == 0.0) {
            continue;
        }
        normal_vector = normal_vector / length;

        normal_vel[kidx][0] = cell_normal_velocity[edge_id * total_layer + layer];
        unit_vector[kidx][0] = normal_vector.x();
        unit_vector[kidx][1] = normal_vector.y();
        unit_vector[kidx][2] = normal_vector.z();
    }

    double alpha = MOPS::Interpolator::compute_alpha(edge_center, kMaxVertexNum, cell_center);
    (void)alpha;
    alpha = 1.0;
    MOPS::Interpolator::mpas_rbf_interp_func_3D_plane_vec_const_dir_comp_coeffs(
        kMaxVertexNum,
        edge_center,
        unit_vector,
        cell_center,
        alpha,
        plane_basis_vector,
        coeffs);

    double x_vel = 0.0;
    double y_vel = 0.0;
    double z_vel = 0.0;
    for (int kidx = 0; kidx < kMaxVertexNum; ++kidx) {
        x_vel += coeffs[kidx][0] * normal_vel[kidx][0];
        y_vel += coeffs[kidx][1] * normal_vel[kidx][0];
        z_vel += coeffs[kidx][2] * normal_vel[kidx][0];
    }

    cell_center_velocity[tid] = make_double3(x_vel, y_vel, z_vel);
}

__global__ void KernelCalcCellVertexVelocity(
    const vec3* vertex_coord,
    const vec3* cell_coord,
    const size_t* cells_on_vertex,
    size_t cells_size,
    size_t vertex_size,
    size_t total_layer,
    const vec3* cell_center_velocity,
    vec3* cell_vertex_velocity)
{
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = vertex_size * total_layer;
    if (tid >= total) {
        return;
    }

    const size_t vertex_idx = tid / total_layer;
    const size_t layer = tid % total_layer;

    constexpr int kNeighborNum = 3;
    size_t neighbor_cells[kNeighborNum] = {0, 0, 0};
    bool boundary = false;

    for (int i = 0; i < kNeighborNum; ++i) {
        const size_t raw_id = cells_on_vertex[vertex_idx * kNeighborNum + i];
        if (raw_id == 0) {
            boundary = true;
            continue;
        }
        const size_t cid = raw_id - 1;
        if (cid >= cells_size) {
            boundary = true;
            continue;
        }
        neighbor_cells[i] = cid;
    }

    if (boundary) {
        cell_vertex_velocity[tid] = make_double3(0.0, 0.0, 0.0);
        return;
    }

    const vec3 p = vertex_coord[vertex_idx];
    const vec3 p1 = cell_coord[neighbor_cells[0]];
    const vec3 p2 = cell_coord[neighbor_cells[1]];
    const vec3 p3 = cell_coord[neighbor_cells[2]];

    MOPS::Interpolator::TRIANGLE tri(p1, p2, p3);
    double u = 0.0;
    double v = 0.0;
    double w = 0.0;
    MOPS::Interpolator::calcTriangleBarycentric(p, &tri, u, v, w);

    const vec3 s1 = cell_center_velocity[neighbor_cells[0] * total_layer + layer];
    const vec3 s2 = cell_center_velocity[neighbor_cells[1] * total_layer + layer];
    const vec3 s3 = cell_center_velocity[neighbor_cells[2] * total_layer + layer];
    cell_vertex_velocity[tid] = u * s1 + v * s2 + w * s3;
}

__global__ void KernelCalcCellVertexVertVelocity(
    const vec3* vertex_coord,
    const vec3* cell_coord,
    const size_t* cells_on_vertex,
    size_t cells_size,
    size_t vertex_size,
    size_t total_layer,
    const double* cell_center_vert_velocity,
    double* cell_vertex_vert_velocity)
{
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = vertex_size * total_layer;
    if (tid >= total) {
        return;
    }

    const size_t vertex_idx = tid / total_layer;
    const size_t layer = tid % total_layer;

    constexpr int kNeighborNum = 3;
    size_t neighbor_cells[kNeighborNum] = {0, 0, 0};
    bool boundary = false;

    for (int i = 0; i < kNeighborNum; ++i) {
        const size_t raw_id = cells_on_vertex[vertex_idx * kNeighborNum + i];
        if (raw_id == 0) {
            boundary = true;
            continue;
        }
        const size_t cid = raw_id - 1;
        if (cid >= cells_size) {
            boundary = true;
            continue;
        }
        neighbor_cells[i] = cid;
    }

    if (boundary) {
        cell_vertex_vert_velocity[tid] = 0.0;
        return;
    }

    const vec3 p = vertex_coord[vertex_idx];
    const vec3 p1 = cell_coord[neighbor_cells[0]];
    const vec3 p2 = cell_coord[neighbor_cells[1]];
    const vec3 p3 = cell_coord[neighbor_cells[2]];

    MOPS::Interpolator::TRIANGLE tri(p1, p2, p3);
    double u = 0.0;
    double v = 0.0;
    double w = 0.0;
    MOPS::Interpolator::calcTriangleBarycentric(p, &tri, u, v, w);

    const double s1 = cell_center_vert_velocity[neighbor_cells[0] * total_layer + layer];
    const double s2 = cell_center_vert_velocity[neighbor_cells[1] * total_layer + layer];
    const double s3 = cell_center_vert_velocity[neighbor_cells[2] * total_layer + layer];
    cell_vertex_vert_velocity[tid] = u * s1 + v * s2 + w * s3;
}

template <typename T>
T* DeviceAllocAndCopy(const std::vector<T>& src, const char* what)
{
    if (src.empty()) {
        return nullptr;
    }

    T* dst = nullptr;
    CheckHip(hipMalloc(reinterpret_cast<void**>(&dst), src.size() * sizeof(T)), what);
    CheckHip(hipMemcpy(dst, src.data(), src.size() * sizeof(T), hipMemcpyHostToDevice), what);
    return dst;
}

template <typename T>
void CopyBackAndFree(std::vector<T>& dst, T* dev_ptr, const char* what)
{
    if (dev_ptr == nullptr) {
        return;
    }
    if (!dst.empty()) {
        CheckHip(hipMemcpy(dst.data(), dev_ptr, dst.size() * sizeof(T), hipMemcpyDeviceToHost), what);
    }
    CheckHip(hipFree(dev_ptr), what);
}

template <typename T>
void FreeDev(T* dev_ptr, const char* what)
{
    if (dev_ptr != nullptr) {
        CheckHip(hipFree(dev_ptr), what);
    }
}

} // namespace

namespace MOPS::GPU::HIPBackend::Kernel::HIPImpl {

void LaunchCalcCellVertexZtop(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_center_ztop,
    std::vector<double>& cell_vertex_ztop,
    const std::vector<size_t>& grid_info)
{
    (void)grid_info;
    if (cells_size <= 0 || total_ztop_layer <= 0) {
        cell_vertex_ztop.clear();
        return;
    }

    const size_t total_cells = static_cast<size_t>(cells_size);
    const size_t total_layer = static_cast<size_t>(total_ztop_layer);
    const size_t vertex_size = grid->vertexCoord_vec.size();

    if (cell_center_ztop.size() < total_cells * total_layer) {
        Error("[HIPBackend::Kernel]::LaunchCalcCellVertexZtop input size mismatch");
    }

    cell_vertex_ztop.assign(vertex_size * total_layer, 0.0);

    vec3* d_vertex_coord = DeviceAllocAndCopy(grid->vertexCoord_vec, "hipMalloc/hipMemcpy vertexCoord");
    vec3* d_cell_coord = DeviceAllocAndCopy(grid->cellCoord_vec, "hipMalloc/hipMemcpy cellCoord");
    size_t* d_cells_on_vertex = DeviceAllocAndCopy(grid->cellsOnVertex_vec, "hipMalloc/hipMemcpy cellsOnVertex");
    double* d_cell_center_ztop = DeviceAllocAndCopy(cell_center_ztop, "hipMalloc/hipMemcpy cellCenterZTop");
    double* d_cell_vertex_ztop = nullptr;

    CheckHip(hipMalloc(reinterpret_cast<void**>(&d_cell_vertex_ztop), cell_vertex_ztop.size() * sizeof(double)), "hipMalloc cellVertexZTop");

    const size_t total = vertex_size * total_layer;
    constexpr int kBlock = 256;
    const int grid_dim = static_cast<int>((total + kBlock - 1) / kBlock);
    KernelCalcCellVertexZtop<<<grid_dim, kBlock>>>(
        d_vertex_coord,
        d_cell_coord,
        d_cells_on_vertex,
        total_cells,
        vertex_size,
        total_layer,
        d_cell_center_ztop,
        d_cell_vertex_ztop);

    CheckHip(hipGetLastError(), "KernelCalcCellVertexZtop launch");
    CheckHip(hipDeviceSynchronize(), "KernelCalcCellVertexZtop sync");

    CopyBackAndFree(cell_vertex_ztop, d_cell_vertex_ztop, "hipMemcpy/hipFree cellVertexZTop");
    FreeDev(d_cell_center_ztop, "hipFree cellCenterZTop");
    FreeDev(d_cells_on_vertex, "hipFree cellsOnVertex");
    FreeDev(d_cell_coord, "hipFree cellCoord");
    FreeDev(d_vertex_coord, "hipFree vertexCoord");
}

void LaunchCalcCellCenterToVertex(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_center_attr,
    std::vector<double>& cell_vertex_attr,
    const std::vector<size_t>& grid_info)
{
    (void)grid_info;
    if (cells_size <= 0 || total_ztop_layer <= 0) {
        cell_vertex_attr.clear();
        return;
    }

    const size_t total_cells = static_cast<size_t>(cells_size);
    const size_t total_layer = static_cast<size_t>(total_ztop_layer);
    const size_t vertex_size = grid->vertexCoord_vec.size();

    if (cell_center_attr.size() < total_cells * total_layer) {
        Error("[HIPBackend::Kernel]::LaunchCalcCellCenterToVertex input size mismatch");
    }

    cell_vertex_attr.assign(vertex_size * total_layer, 0.0);

    vec3* d_vertex_coord = DeviceAllocAndCopy(grid->vertexCoord_vec, "hipMalloc/hipMemcpy vertexCoord");
    vec3* d_cell_coord = DeviceAllocAndCopy(grid->cellCoord_vec, "hipMalloc/hipMemcpy cellCoord");
    size_t* d_cells_on_vertex = DeviceAllocAndCopy(grid->cellsOnVertex_vec, "hipMalloc/hipMemcpy cellsOnVertex");
    double* d_cell_center_attr = DeviceAllocAndCopy(cell_center_attr, "hipMalloc/hipMemcpy cellCenterAttr");
    double* d_cell_vertex_attr = nullptr;

    CheckHip(hipMalloc(reinterpret_cast<void**>(&d_cell_vertex_attr), cell_vertex_attr.size() * sizeof(double)), "hipMalloc cellVertexAttr");

    const size_t total = vertex_size * total_layer;
    constexpr int kBlock = 256;
    const int grid_dim = static_cast<int>((total + kBlock - 1) / kBlock);
    KernelCalcCellCenterToVertex<<<grid_dim, kBlock>>>(
        d_vertex_coord,
        d_cell_coord,
        d_cells_on_vertex,
        total_cells,
        vertex_size,
        total_layer,
        d_cell_center_attr,
        d_cell_vertex_attr);

    CheckHip(hipGetLastError(), "KernelCalcCellCenterToVertex launch");
    CheckHip(hipDeviceSynchronize(), "KernelCalcCellCenterToVertex sync");

    CopyBackAndFree(cell_vertex_attr, d_cell_vertex_attr, "hipMemcpy/hipFree cellVertexAttr");
    FreeDev(d_cell_center_attr, "hipFree cellCenterAttr");
    FreeDev(d_cells_on_vertex, "hipFree cellsOnVertex");
    FreeDev(d_cell_coord, "hipFree cellCoord");
    FreeDev(d_vertex_coord, "hipFree vertexCoord");
}

void LaunchCalcCellCenterVelocityByZM(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_zonal_velocity,
    const std::vector<double>& cell_meridional_velocity,
    std::vector<vec3>& cell_center_velocity,
    const std::vector<size_t>& grid_info)
{
    (void)grid_info;
    if (cells_size <= 0 || total_ztop_layer <= 0) {
        cell_center_velocity.clear();
        return;
    }

    const size_t total_cells = static_cast<size_t>(cells_size);
    const size_t total_layer = static_cast<size_t>(total_ztop_layer);
    const size_t total = total_cells * total_layer;

    if (cell_zonal_velocity.size() < total || cell_meridional_velocity.size() < total) {
        Error("[HIPBackend::Kernel]::LaunchCalcCellCenterVelocityByZM input size mismatch");
    }

    cell_center_velocity.assign(total, vec3{0.0, 0.0, 0.0});

    vec3* d_cell_coord = DeviceAllocAndCopy(grid->cellCoord_vec, "hipMalloc/hipMemcpy cellCoord");
    double* d_cell_zonal = DeviceAllocAndCopy(cell_zonal_velocity, "hipMalloc/hipMemcpy cellZonalVelocity");
    double* d_cell_meridional = DeviceAllocAndCopy(cell_meridional_velocity, "hipMalloc/hipMemcpy cellMeridionalVelocity");
    vec3* d_cell_center_velocity = nullptr;

    CheckHip(hipMalloc(reinterpret_cast<void**>(&d_cell_center_velocity), total * sizeof(vec3)), "hipMalloc cellCenterVelocity");

    constexpr int kBlock = 256;
    const int grid_dim = static_cast<int>((total + kBlock - 1) / kBlock);
    KernelCalcCellCenterVelocityByZM<<<grid_dim, kBlock>>>(
        d_cell_coord,
        total_cells,
        total_layer,
        d_cell_zonal,
        d_cell_meridional,
        d_cell_center_velocity);

    CheckHip(hipGetLastError(), "KernelCalcCellCenterVelocityByZM launch");
    CheckHip(hipDeviceSynchronize(), "KernelCalcCellCenterVelocityByZM sync");

    CopyBackAndFree(cell_center_velocity, d_cell_center_velocity, "hipMemcpy/hipFree cellCenterVelocity");
    FreeDev(d_cell_meridional, "hipFree cellMeridionalVelocity");
    FreeDev(d_cell_zonal, "hipFree cellZonalVelocity");
    FreeDev(d_cell_coord, "hipFree cellCoord");
}

void LaunchCalcCellVertexVelocityByZM(
    MPASOGrid* grid,
    int vertex_size,
    int total_ztop_layer,
    const std::vector<double>& cell_vertex_zonal_velocity,
    const std::vector<double>& cell_vertex_meridional_velocity,
    std::vector<vec3>& cell_vertex_velocity,
    const std::vector<size_t>& grid_info)
{
    (void)grid_info;
    if (vertex_size <= 0 || total_ztop_layer <= 0) {
        cell_vertex_velocity.clear();
        return;
    }

    const size_t total_vertex = static_cast<size_t>(vertex_size);
    const size_t total_layer = static_cast<size_t>(total_ztop_layer);
    const size_t total = total_vertex * total_layer;

    if (cell_vertex_zonal_velocity.size() < total || cell_vertex_meridional_velocity.size() < total) {
        Error("[HIPBackend::Kernel]::LaunchCalcCellVertexVelocityByZM input size mismatch");
    }
    if (grid->vertexCoord_vec.size() < total_vertex) {
        Error("[HIPBackend::Kernel]::LaunchCalcCellVertexVelocityByZM grid vertex size mismatch");
    }

    cell_vertex_velocity.assign(total, vec3{0.0, 0.0, 0.0});

    vec3* d_vertex_coord = nullptr;
    double* d_vertex_zonal = nullptr;
    double* d_vertex_meridional = nullptr;
    vec3* d_vertex_velocity = nullptr;

    CheckHip(hipMalloc(reinterpret_cast<void**>(&d_vertex_coord), total_vertex * sizeof(vec3)), "hipMalloc vertexCoord");
    CheckHip(
        hipMemcpy(
            d_vertex_coord,
            grid->vertexCoord_vec.data(),
            total_vertex * sizeof(vec3),
            hipMemcpyHostToDevice),
        "hipMemcpy vertexCoord");

    d_vertex_zonal = DeviceAllocAndCopy(cell_vertex_zonal_velocity, "hipMalloc/hipMemcpy vertexZonalVelocity");
    d_vertex_meridional = DeviceAllocAndCopy(cell_vertex_meridional_velocity, "hipMalloc/hipMemcpy vertexMeridionalVelocity");
    CheckHip(hipMalloc(reinterpret_cast<void**>(&d_vertex_velocity), total * sizeof(vec3)), "hipMalloc vertexVelocity");

    constexpr int kBlock = 256;
    const int grid_dim = static_cast<int>((total + kBlock - 1) / kBlock);
    KernelCalcCellVertexVelocityByZM<<<grid_dim, kBlock>>>(
        d_vertex_coord,
        total_vertex,
        total_layer,
        d_vertex_zonal,
        d_vertex_meridional,
        d_vertex_velocity);

    CheckHip(hipGetLastError(), "KernelCalcCellVertexVelocityByZM launch");
    CheckHip(hipDeviceSynchronize(), "KernelCalcCellVertexVelocityByZM sync");

    CopyBackAndFree(cell_vertex_velocity, d_vertex_velocity, "hipMemcpy/hipFree vertexVelocity");
    FreeDev(d_vertex_meridional, "hipFree vertexMeridionalVelocity");
    FreeDev(d_vertex_zonal, "hipFree vertexZonalVelocity");
    FreeDev(d_vertex_coord, "hipFree vertexCoord");
}

void LaunchCalcCellCenterVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_normal_velocity,
    std::vector<vec3>& cell_center_velocity,
    const std::vector<size_t>& grid_info)
{
    (void)grid_info;
    if (cells_size <= 0 || total_ztop_layer <= 0) {
        cell_center_velocity.clear();
        return;
    }

    const size_t total_cells = static_cast<size_t>(cells_size);
    const size_t total_layer = static_cast<size_t>(total_ztop_layer);
    const size_t total = total_cells * total_layer;
    if (grid->numberVertexOnCell_vec.size() < total_cells) {
        Error("[HIPBackend::Kernel]::LaunchCalcCellCenterVelocity numberVertexOnCell size mismatch");
    }
    if (cell_normal_velocity.empty()) {
        Error("[HIPBackend::Kernel]::LaunchCalcCellCenterVelocity input normal velocity is empty");
    }

    cell_center_velocity.assign(total, vec3{0.0, 0.0, 0.0});

    vec3* d_cell_coord = DeviceAllocAndCopy(grid->cellCoord_vec, "hipMalloc/hipMemcpy cellCoord");
    vec3* d_edge_coord = DeviceAllocAndCopy(grid->edgeCoord_vec, "hipMalloc/hipMemcpy edgeCoord");
    size_t* d_number_vertex_on_cell = DeviceAllocAndCopy(grid->numberVertexOnCell_vec, "hipMalloc/hipMemcpy numberVertexOnCell");
    size_t* d_edges_on_cell = DeviceAllocAndCopy(grid->edgesOnCell_vec, "hipMalloc/hipMemcpy edgesOnCell");
    size_t* d_cells_on_edge = DeviceAllocAndCopy(grid->cellsOnEdge_vec, "hipMalloc/hipMemcpy cellsOnEdge");
    double* d_cell_normal_velocity = DeviceAllocAndCopy(cell_normal_velocity, "hipMalloc/hipMemcpy cellNormalVelocity");
    vec3* d_cell_center_velocity = nullptr;
    CheckHip(hipMalloc(reinterpret_cast<void**>(&d_cell_center_velocity), total * sizeof(vec3)), "hipMalloc cellCenterVelocity(normal)");

    constexpr int kBlock = 256;
    const int grid_dim = static_cast<int>((total + kBlock - 1) / kBlock);
    KernelCalcCellCenterVelocity<<<grid_dim, kBlock>>>(
        d_cell_coord,
        d_edge_coord,
        d_number_vertex_on_cell,
        d_edges_on_cell,
        d_cells_on_edge,
        total_cells,
        total_layer,
        d_cell_normal_velocity,
        d_cell_center_velocity);

    CheckHip(hipGetLastError(), "KernelCalcCellCenterVelocity launch");
    CheckHip(hipDeviceSynchronize(), "KernelCalcCellCenterVelocity sync");

    CopyBackAndFree(cell_center_velocity, d_cell_center_velocity, "hipMemcpy/hipFree cellCenterVelocity(normal)");
    FreeDev(d_cell_normal_velocity, "hipFree cellNormalVelocity");
    FreeDev(d_cells_on_edge, "hipFree cellsOnEdge");
    FreeDev(d_edges_on_cell, "hipFree edgesOnCell");
    FreeDev(d_number_vertex_on_cell, "hipFree numberVertexOnCell");
    FreeDev(d_edge_coord, "hipFree edgeCoord");
    FreeDev(d_cell_coord, "hipFree cellCoord");
}

void LaunchCalcCellVertexVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<vec3>& cell_center_velocity,
    std::vector<vec3>& cell_vertex_velocity,
    const std::vector<size_t>& grid_info)
{
    (void)grid_info;
    if (cells_size <= 0 || total_ztop_layer <= 0) {
        cell_vertex_velocity.clear();
        return;
    }

    const size_t total_cells = static_cast<size_t>(cells_size);
    const size_t total_layer = static_cast<size_t>(total_ztop_layer);
    const size_t vertex_size = grid->vertexCoord_vec.size();
    if (cell_center_velocity.size() < total_cells * total_layer) {
        Error("[HIPBackend::Kernel]::LaunchCalcCellVertexVelocity input size mismatch");
    }

    cell_vertex_velocity.assign(vertex_size * total_layer, vec3{0.0, 0.0, 0.0});

    vec3* d_vertex_coord = DeviceAllocAndCopy(grid->vertexCoord_vec, "hipMalloc/hipMemcpy vertexCoord");
    vec3* d_cell_coord = DeviceAllocAndCopy(grid->cellCoord_vec, "hipMalloc/hipMemcpy cellCoord");
    size_t* d_cells_on_vertex = DeviceAllocAndCopy(grid->cellsOnVertex_vec, "hipMalloc/hipMemcpy cellsOnVertex");
    vec3* d_cell_center_velocity = DeviceAllocAndCopy(cell_center_velocity, "hipMalloc/hipMemcpy cellCenterVelocity(input)");
    vec3* d_cell_vertex_velocity = nullptr;
    CheckHip(hipMalloc(reinterpret_cast<void**>(&d_cell_vertex_velocity), cell_vertex_velocity.size() * sizeof(vec3)), "hipMalloc cellVertexVelocity");

    const size_t total = vertex_size * total_layer;
    constexpr int kBlock = 256;
    const int grid_dim = static_cast<int>((total + kBlock - 1) / kBlock);
    KernelCalcCellVertexVelocity<<<grid_dim, kBlock>>>(
        d_vertex_coord,
        d_cell_coord,
        d_cells_on_vertex,
        total_cells,
        vertex_size,
        total_layer,
        d_cell_center_velocity,
        d_cell_vertex_velocity);

    CheckHip(hipGetLastError(), "KernelCalcCellVertexVelocity launch");
    CheckHip(hipDeviceSynchronize(), "KernelCalcCellVertexVelocity sync");

    CopyBackAndFree(cell_vertex_velocity, d_cell_vertex_velocity, "hipMemcpy/hipFree cellVertexVelocity");
    FreeDev(d_cell_center_velocity, "hipFree cellCenterVelocity(input)");
    FreeDev(d_cells_on_vertex, "hipFree cellsOnVertex");
    FreeDev(d_cell_coord, "hipFree cellCoord");
    FreeDev(d_vertex_coord, "hipFree vertexCoord");
}

void LaunchCalcCellVertexVertVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer_p1,
    const std::vector<double>& cell_center_vert_velocity,
    std::vector<double>& cell_vertex_vert_velocity,
    const std::vector<size_t>& grid_info)
{
    (void)grid_info;
    if (cells_size <= 0 || total_ztop_layer_p1 <= 0) {
        cell_vertex_vert_velocity.clear();
        return;
    }

    const size_t total_cells = static_cast<size_t>(cells_size);
    const size_t total_layer = static_cast<size_t>(total_ztop_layer_p1);
    const size_t vertex_size = grid->vertexCoord_vec.size();
    if (cell_center_vert_velocity.size() < total_cells * total_layer) {
        Error("[HIPBackend::Kernel]::LaunchCalcCellVertexVertVelocity input size mismatch");
    }

    cell_vertex_vert_velocity.assign(vertex_size * total_layer, 0.0);

    vec3* d_vertex_coord = DeviceAllocAndCopy(grid->vertexCoord_vec, "hipMalloc/hipMemcpy vertexCoord");
    vec3* d_cell_coord = DeviceAllocAndCopy(grid->cellCoord_vec, "hipMalloc/hipMemcpy cellCoord");
    size_t* d_cells_on_vertex = DeviceAllocAndCopy(grid->cellsOnVertex_vec, "hipMalloc/hipMemcpy cellsOnVertex");
    double* d_cell_center_vert_velocity = DeviceAllocAndCopy(cell_center_vert_velocity, "hipMalloc/hipMemcpy cellCenterVertVelocity");
    double* d_cell_vertex_vert_velocity = nullptr;
    CheckHip(hipMalloc(reinterpret_cast<void**>(&d_cell_vertex_vert_velocity), cell_vertex_vert_velocity.size() * sizeof(double)), "hipMalloc cellVertexVertVelocity");

    const size_t total = vertex_size * total_layer;
    constexpr int kBlock = 256;
    const int grid_dim = static_cast<int>((total + kBlock - 1) / kBlock);
    KernelCalcCellVertexVertVelocity<<<grid_dim, kBlock>>>(
        d_vertex_coord,
        d_cell_coord,
        d_cells_on_vertex,
        total_cells,
        vertex_size,
        total_layer,
        d_cell_center_vert_velocity,
        d_cell_vertex_vert_velocity);

    CheckHip(hipGetLastError(), "KernelCalcCellVertexVertVelocity launch");
    CheckHip(hipDeviceSynchronize(), "KernelCalcCellVertexVertVelocity sync");

    CopyBackAndFree(cell_vertex_vert_velocity, d_cell_vertex_vert_velocity, "hipMemcpy/hipFree cellVertexVertVelocity");
    FreeDev(d_cell_center_vert_velocity, "hipFree cellCenterVertVelocity");
    FreeDev(d_cells_on_vertex, "hipFree cellsOnVertex");
    FreeDev(d_cell_coord, "hipFree cellCoord");
    FreeDev(d_vertex_coord, "hipFree vertexCoord");
}

} // namespace MOPS::GPU::HIPBackend::Kernel::HIPImpl
