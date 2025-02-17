#include "SYCLKernel.h"


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

void SYCLKernel::GetCellVerticesIdx(int cell_id, int current_cell_vertices_number, size_t* current_cell_vertices_idx, const int VLA, const int max_edge,
        sycl::accessor<size_t, 1, sycl::access::mode::read> acc_verticesOnCell_buf)
{
    // 找出所有候选顶点
    for (size_t k = 0; k < VLA; ++k)
    {
        current_cell_vertices_idx[k] = acc_verticesOnCell_buf[cell_id * max_edge + k] - 1; // Assuming max_edge is the max number of vertices per cell
    }
    // 不存在的顶点设置为nan
    auto nan = std::numeric_limits<size_t>::max();
    for (size_t k = current_cell_vertices_number; k < VLA; ++k)
    {
        current_cell_vertices_idx[k] = nan;
    }
}

bool SYCLKernel::IsInOcean(int cell_id, int max_edge, vec3 current_position, 
        sycl::accessor<size_t, 1, sycl::access::mode::read> acc_numberVertexOnCell_buf,
        sycl::accessor<size_t, 1, sycl::access::mode::read> acc_verticesOnCell_buf, 
        sycl::accessor<vec3, 1, sycl::access::mode::read> acc_vertexCoord_buf)
{
    const int VLA_SIZE = 20;
    // 判断是否在大陆上
    bool is_land = false;
    // 1.1 计算这个CELL 有多少个顶点
    auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
    // 1.2 找出所有候选顶点
    size_t current_cell_vertices_idx[VLA_SIZE];
    for (size_t k = 0; k < max_edge; ++k)
    {
        current_cell_vertices_idx[k] = acc_verticesOnCell_buf[cell_id * max_edge + k] - 1; // Assuming max_edge is the max number of vertices per cell
    }
    // 1.3 不存在的顶点设置为nan
    auto nan = std::numeric_limits<size_t>::max();
    for (size_t k = current_cell_vertices_number; k < max_edge; ++k)
    {
        current_cell_vertices_idx[k] = nan;
    }
    // =============================== 找到max_edge个顶点
    double normalsConsistency[VLA_SIZE];
    for (auto k = 0; k < current_cell_vertices_number; k++)
    {
        auto A_idx = current_cell_vertices_idx[k];
        auto B_idx = current_cell_vertices_idx[(k + 1) % current_cell_vertices_number];
        auto A = acc_vertexCoord_buf[A_idx];
        auto B = acc_vertexCoord_buf[B_idx];
        vec3 O(0.0, 0.0, 0.0);
        auto AO = O - A;
        auto BO = O - B;
        auto A_point = current_position - A;
        vec3 surface_normal = YOSEF_CROSS(AO, BO);
        double direction = YOSEF_DOT(surface_normal, A_point);
        normalsConsistency[k] = direction;
    }
    int sign = (normalsConsistency[0] > 0) ? 1 : -1;
    for (auto k = 1; k < current_cell_vertices_number; k++)
    {
        int currentSign = (normalsConsistency[k] > 0) ? 1 : -1;
        if (currentSign != sign)
        {
            // 这个点在大陆上
            is_land = true;
            break;
        }
    }

    return is_land;
}

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