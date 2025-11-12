#include "SYCL/SYCLKernel.h"

using namespace MOPS;

void SYCLKernel::SearchKDTree(int* cell_id_vec, MPASOGrid* grid, int width, int height, double minLat, double maxLat, double minLon, double maxLon)
{
    for (auto i = 0; i < height; i++)
    {
        for (auto j = 0; j < width; j++)
        {
            vec2 pixel = vec2(i, j);
            vec2 latlon_r;
            GeoConverter::convertPixelToLatLonToRadians(width, height, minLat, maxLat, minLon, maxLon, pixel, latlon_r);
            vec3 current_position;
            GeoConverter::convertRadianLatLonToXYZ(latlon_r, current_position);
            int cell_id_value = -1;
            grid->searchKDT(current_position, cell_id_value);
            int global_id = i * width + j;
            cell_id_vec[global_id] = cell_id_value;
        }
    }
}

SYCL_EXTERNAL
void SYCLKernel::GetCellVerticesIdx(int cell_id, int current_cell_vertices_number, size_t* current_cell_vertices_idx, const int VLA, const int max_edge,
        sycl::accessor<size_t, 1, sycl::access::mode::read> acc_verticesOnCell_buf)
{
    // Find all candidate vertices
    for (size_t k = 0; k < VLA; ++k)
    {
        current_cell_vertices_idx[k] = acc_verticesOnCell_buf[cell_id * max_edge + k] - 1; // Assuming max_edge is the max number of vertices per cell
    }
    // Set non-existent vertices to nan
    auto nan = std::numeric_limits<size_t>::max();
    for (size_t k = current_cell_vertices_number; k < VLA; ++k)
    {
        current_cell_vertices_idx[k] = nan;
    }
}
SYCL_EXTERNAL
bool SYCLKernel::IsInMesh(int cell_id, int max_edge, vec3 current_position, 
        sycl::accessor<size_t, 1, sycl::access::mode::read> acc_numberVertexOnCell_buf,
        sycl::accessor<size_t, 1, sycl::access::mode::read> acc_verticesOnCell_buf, 
        sycl::accessor<vec3, 1, sycl::access::mode::read> acc_vertexCoord_buf)
{

    if (!sycl::isfinite(current_position.x()) ||
        !sycl::isfinite(current_position.y()) ||
        !sycl::isfinite(current_position.z())) 
    {
        return false;
    }

    
    auto nan = std::numeric_limits<size_t>::max();
    auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
    if (current_cell_vertices_number == 0) return false;
    auto idx_at = [&](size_t k)->size_t {
        return acc_verticesOnCell_buf[cell_id * max_edge + k] - 1;
    };

    for (auto k = 0; k < current_cell_vertices_number; k++)
    {
        auto A_idx = idx_at(k);
        auto B_idx = idx_at((k + 1) % current_cell_vertices_number);

        auto A = acc_vertexCoord_buf[A_idx];
        auto B = acc_vertexCoord_buf[B_idx];
        
        vec3 surface_normal = YOSEF_CROSS(A, B);
        double direction = YOSEF_DOT(surface_normal, current_position);
        
        if (direction < 0) return false;
    }
    return true;
}

SYCL_EXTERNAL
        void SYCLKernel::GetCellNeighborsIdx(int cell_id, int current_cell_vertices_number, int* current_cell_neighbors_idx, const int VLA, const int max_edge,
                                        sycl::accessor<size_t, 1, sycl::access::mode::read> acc_cells_onCell_buf)
{
    if (current_cell_vertices_number > VLA) return;
    current_cell_neighbors_idx[0] = cell_id;
    int copyN = current_cell_vertices_number;
    if (copyN > VLA-1) copyN = VLA-1;
    for (int k=0; k<copyN; ++k) {
        int nid1 = (int)acc_cells_onCell_buf[cell_id * max_edge + k];  
        current_cell_neighbors_idx[k] = nid1 - 1;
    }

    // Put self at the last valid position
    current_cell_neighbors_idx[copyN] = cell_id;
    for (auto k = copyN + 1; k < VLA; k++)
    {
        current_cell_neighbors_idx[k] = -1;
    }

}
SYCL_EXTERNAL
bool SYCLKernel::GetCellVertexPos(vec3* current_cell_vertex_pos, size_t* current_cell_vertices_idx, const int VLA, int current_cell_vertices_number, sycl::accessor<vec3, 1, sycl::access::mode::read> acc_vertexCoord_buf)
{
    if (current_cell_vertices_number > VLA)
    {
        return false;
    }
    auto double_nan = std::numeric_limits<double>::quiet_NaN();
    vec3 vec3_nan = { double_nan, double_nan, double_nan };
    for (auto v_idx = 0; v_idx < current_cell_vertices_number; v_idx++)
    {
        auto VID = current_cell_vertices_idx[v_idx];
        vec3 pos = acc_vertexCoord_buf[VID];
        current_cell_vertex_pos[v_idx] = pos;
    }
    for (auto v_idx = current_cell_vertices_number; v_idx < VLA; v_idx++)
    {
        current_cell_vertex_pos[v_idx] = vec3_nan;
    }
    return true;
}
SYCL_EXTERNAL
vec3 SYCLKernel::CalcVelocity(size_t* current_cell_vertices_idx, double* current_cell_vertex_weight, 
            const int VLA, int current_cell_vertices_number, int TOTAY_ZTOP_LAYER, int layer,
            sycl::accessor<vec3, 1, sycl::access::mode::read> acc_cellVertexVelocity_buf)
{
    vec3 current_point_vel1 = { 0.0, 0.0, 0.0 };
    const int VLA_SIZE = 20;
    vec3 vertex_vel1[VLA_SIZE];
    for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
    {
        auto VID = current_cell_vertices_idx[v_idx];
        vec3 vel1 = acc_cellVertexVelocity_buf[VID * TOTAY_ZTOP_LAYER + layer]; 
        vertex_vel1[v_idx] = vel1;
    }
    for (auto v_idx = current_cell_vertices_number; v_idx < VLA_SIZE; v_idx++)
    {
        vertex_vel1[v_idx] = { 0.0, 0.0, 0.0 };
    }
    for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
    {
        current_point_vel1.x() += current_cell_vertex_weight[v_idx] * vertex_vel1[v_idx].x(); // layer
        current_point_vel1.y() += current_cell_vertex_weight[v_idx] * vertex_vel1[v_idx].y();
        current_point_vel1.z() += current_cell_vertex_weight[v_idx] * vertex_vel1[v_idx].z();
    }
    return current_point_vel1;
}
SYCL_EXTERNAL
double SYCLKernel::CalcAttribute(size_t* current_cell_vertices_idx, double* current_cell_vertex_weight, 
                                    const int VLA, int current_cell_vertices_number, int TOTAY_ZTOP_LAYER, int layer,
                                    sycl::accessor<double, 1, sycl::access::mode::read> acc_cellAttribute_buf)
{
    double current_point_attr_value = 0.0;
    const int VLA_SIZE = 20;
    double vertex_value1[VLA_SIZE];
    for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
    {
        auto VID = current_cell_vertices_idx[v_idx];
        double value1 = acc_cellAttribute_buf[VID * TOTAY_ZTOP_LAYER + layer]; 
        vertex_value1[v_idx] = value1;
    }
    for (auto v_idx = current_cell_vertices_number; v_idx < VLA_SIZE; v_idx++)
    {
        vertex_value1[v_idx] = 0.0;
    }
    for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
    {
        current_point_attr_value += current_cell_vertex_weight[v_idx] * vertex_value1[v_idx];
    }
    return current_point_attr_value;
}

SYCL_EXTERNAL
vec3 SYCLKernel::CalcRotationAxis(const vec3& position, const vec3& velocity)
{
    vec3 axis;
    axis.x() = position.y() * velocity.z() - position.z() * velocity.y();
    axis.y() = position.z() * velocity.x() - position.x() * velocity.z();
    axis.z() = position.x() * velocity.y() - position.y() * velocity.x();
    return axis;
}

SYCL_EXTERNAL
vec3 SYCLKernel::CalcPositionAfterRotation(const vec3& position, const vec3& axis, double theta_rad)
{
    double thetaRad = theta_rad;
    double cosTheta = sycl::cos(thetaRad);
    double sinTheta = sycl::sin(thetaRad);

    // normalize
    double tmp_length = YOSEF_LENGTH(axis);
    vec3 u;
    u.x() = axis.x() / tmp_length;
    u.y() = axis.y() / tmp_length;
    u.z() = axis.z() / tmp_length;

    vec3 rotated;
    rotated.x() = (cosTheta + u.x() * u.x() * (1.0 - cosTheta)) * position.x() +
        (u.x() * u.y() * (1.0 - cosTheta) - u.z() * sinTheta) * position.y() +
        (u.x() * u.z() * (1.0 - cosTheta) + u.y() * sinTheta) * position.z();

    rotated.y() = (u.y() * u.x() * (1.0 - cosTheta) + u.z() * sinTheta) * position.x() +
        (cosTheta + u.y() * u.y() * (1.0 - cosTheta)) * position.y() +
        (u.y() * u.z() * (1.0 - cosTheta) - u.x() * sinTheta) * position.z();

    rotated.z() = (u.z() * u.x() * (1.0 - cosTheta) - u.y() * sinTheta) * position.x() +
        (u.z() * u.y() * (1.0 - cosTheta) + u.x() * sinTheta) * position.y() +
        (cosTheta + u.z() * u.z() * (1.0 - cosTheta)) * position.z();

    return rotated;
}