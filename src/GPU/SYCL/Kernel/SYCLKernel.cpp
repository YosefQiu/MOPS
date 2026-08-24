#include "GPU/SYCL/Kernel/SYCLKernel.h"

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
        
        vec3 surface_normal = MOPS_CROSS(A, B);
        double direction = MOPS_DOT(surface_normal, current_position);
        
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
    const int VLA_SIZE = 10;
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
    const int VLA_SIZE = 10;
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
    double tmp_length = MOPS_LENGTH(axis);
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

// ============================================================================
// Advanced Pathline Helper Functions - SYCL Port from CUDA
// ============================================================================

SYCL_EXTERNAL
vec3 SYCLKernel::AdvectOnSphereSYCL(const vec3& pos, const vec3& vel, double dt)
{
    double r = MOPS_LENGTH(pos);
    double speed = MOPS_LENGTH(vel);

    if (speed < 1e-20) {
        return pos;
    }

    vec3 rotation_axis = CalcRotationAxis(pos, vel);
    double theta = speed * dt / r;
    vec3 new_pos = CalcPositionAfterRotation(pos, rotation_axis, theta);

    return new_pos;
}

SYCL_EXTERNAL
int SYCLKernel::FindContainingCellInCurrentOrNeighborsSYCL(
    const vec3& pos,
    int current_cell_id,
    int max_edge,
    sycl::accessor<size_t, 1, sycl::access::mode::read> acc_numberVertexOnCell_buf,
    sycl::accessor<size_t, 1, sycl::access::mode::read> acc_cellsOnCell_buf,
    sycl::accessor<size_t, 1, sycl::access::mode::read> acc_verticesOnCell_buf,
    sycl::accessor<vec3, 1, sycl::access::mode::read> acc_vertexCoord_buf)
{
    constexpr int MAX_VERTEX_NUM = 10;

    // First check current cell (fast path)
    bool in_current = IsInMesh(
        current_cell_id,
        max_edge,
        pos,
        acc_numberVertexOnCell_buf,
        acc_verticesOnCell_buf,
        acc_vertexCoord_buf);

    if (in_current) {
        return current_cell_id;
    }

    // Not in current cell - check neighbor cells
    int current_cell_vertices_number = static_cast<int>(acc_numberVertexOnCell_buf[current_cell_id]);
    if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
        return -1;
    }

    // Get neighbor cells
    int cell_neig_vec[MAX_VERTEX_NUM + 1];
    for (int i = 0; i < MAX_VERTEX_NUM + 1; ++i) {
        cell_neig_vec[i] = -1;
    }
    GetCellNeighborsIdx(
        current_cell_id,
        current_cell_vertices_number,
        cell_neig_vec,
        MAX_VERTEX_NUM,
        max_edge,
        acc_cellsOnCell_buf);

    // Check each neighbor cell
    for (int i = 0; i < current_cell_vertices_number; ++i) {
        int neighbor_cell_id = static_cast<int>(cell_neig_vec[i]);
        if (neighbor_cell_id < 0) {
            continue;
        }

        bool in_neighbor = IsInMesh(
            neighbor_cell_id,
            max_edge,
            pos,
            acc_numberVertexOnCell_buf,
            acc_verticesOnCell_buf,
            acc_vertexCoord_buf);

        if (in_neighbor) {
            return neighbor_cell_id;
        }
    }

    // Not in current cell or any neighbor - truly outside mesh
    return -1;
}

// Helper function: Hash function for deterministic pseudo-random number generation
static unsigned int hash_sycl(unsigned int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x);
    return x;
}

// Helper function: Generate random float in [0, 1] using hash-based PRNG
static float random_float_sycl(int particle_id, int timestep, int component)
{
    unsigned int seed = hash_sycl(static_cast<unsigned int>(particle_id * 73856093 + timestep * 19349663 + component * 83492791));
    return (seed & 0xFFFFFF) / 16777216.0f;
}

SYCL_EXTERNAL
vec3 SYCLKernel::GenerateRandomTangentVelocitySYCL(
    const vec3& pos,
    int particle_id,
    int timestep,
    double magnitude,
    int attempt)
{
    // Normalize position to get radial direction
    vec3 radial = pos;
    double radial_len = MOPS_LENGTH(radial);
    if (radial_len < 1e-12) {
        return vec3{0.0, 0.0, 0.0};
    }
    radial = radial / radial_len;

    // Generate two random angles using hash-based PRNG
    float rand1 = random_float_sycl(particle_id, timestep, attempt * 2);
    float rand2 = random_float_sycl(particle_id, timestep, attempt * 2 + 1);

    double theta = rand1 * 2.0 * M_PI;  // Azimuthal angle [0, 2π]
    double phi = rand2 * M_PI;          // Polar angle [0, π]

    // Create a random tangent direction perpendicular to radial
    vec3 arbitrary{1.0, 0.0, 0.0};
    if (sycl::fabs(radial.x()) > 0.9) {
        arbitrary = vec3{0.0, 1.0, 0.0};
    }

    vec3 tangent1 = MOPS_CROSS(radial, arbitrary);
    double len1 = MOPS_LENGTH(tangent1);
    if (len1 > 1e-12) {
        tangent1 = tangent1 / len1;
    }

    vec3 tangent2 = MOPS_CROSS(radial, tangent1);
    double len2 = MOPS_LENGTH(tangent2);
    if (len2 > 1e-12) {
        tangent2 = tangent2 / len2;
    }

    // Combine the two tangent vectors with random angles
    vec3 random_tangent = tangent1 * sycl::cos(theta) + tangent2 * sycl::sin(theta);

    // Scale by magnitude
    return random_tangent * magnitude;
}

SYCL_EXTERNAL
void SYCLKernel::FindNearestCellEdgeSYCL(
    const vec3& pos,
    int cell_id,
    int max_edge,
    sycl::accessor<size_t, 1, sycl::access::mode::read> acc_numberVertexOnCell_buf,
    sycl::accessor<size_t, 1, sycl::access::mode::read> acc_verticesOnCell_buf,
    sycl::accessor<vec3, 1, sycl::access::mode::read> acc_vertexCoord_buf,
    int& edge_va_idx,
    int& edge_vb_idx)
{
    constexpr int MAX_VERTEX_NUM = 10;

    edge_va_idx = -1;
    edge_vb_idx = -1;

    int current_cell_vertices_number = static_cast<int>(acc_numberVertexOnCell_buf[cell_id]);
    if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
        return;
    }

    size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
    GetCellVerticesIdx(
        cell_id,
        current_cell_vertices_number,
        current_cell_vertices_idx,
        MAX_VERTEX_NUM,
        max_edge,
        acc_verticesOnCell_buf);

    double min_distance = 1e300;

    for (int k = 0; k < current_cell_vertices_number; ++k) {
        int va_idx = static_cast<int>(current_cell_vertices_idx[k]);
        int vb_idx = static_cast<int>(current_cell_vertices_idx[(k + 1) % current_cell_vertices_number]);

        vec3 va = acc_vertexCoord_buf[va_idx];
        vec3 vb = acc_vertexCoord_buf[vb_idx];

        // Compute distance from pos to edge midpoint
        vec3 edge_midpoint = (va + vb) * 0.5;
        double dist = MOPS_LENGTH(pos - edge_midpoint);

        if (dist < min_distance) {
            min_distance = dist;
            edge_va_idx = va_idx;
            edge_vb_idx = vb_idx;
        }
    }
}

SYCL_EXTERNAL
vec3 SYCLKernel::ProjectVelocityOntoEdgeSYCL(
    const vec3& vel,
    const vec3& pos,
    const vec3& va,
    const vec3& vb)
{
    // Edge vector
    vec3 edge_vec = vb - va;

    // Radial direction
    vec3 radial = pos;
    double radial_len = MOPS_LENGTH(radial);
    if (radial_len < 1e-12) {
        return vec3{0.0, 0.0, 0.0};
    }
    radial = radial / radial_len;

    // Project edge vector onto tangent plane
    double edge_radial_component = MOPS_DOT(edge_vec, radial);
    vec3 edge_tangent = edge_vec - radial * edge_radial_component;

    double edge_tangent_len = MOPS_LENGTH(edge_tangent);
    if (edge_tangent_len < 1e-12) {
        return vec3{0.0, 0.0, 0.0};
    }
    edge_tangent = edge_tangent / edge_tangent_len;

    // Project velocity onto edge tangent direction
    double vel_projection = MOPS_DOT(vel, edge_tangent);
    vec3 projected_vel = edge_tangent * vel_projection;

    return projected_vel;
}

SYCL_EXTERNAL
double SYCLKernel::CalcZTopAtLevel0SYCL(
    const vec3& pos,
    int cell_id,
    int max_edge,
    int vertex_size,
    sycl::accessor<size_t, 1, sycl::access::mode::read> acc_numberVertexOnCell_buf,
    sycl::accessor<size_t, 1, sycl::access::mode::read> acc_verticesOnCell_buf,
    sycl::accessor<vec3, 1, sycl::access::mode::read> acc_vertexCoord_buf,
    sycl::accessor<double, 1, sycl::access::mode::read> acc_cellVertexZTop_buf)
{
    constexpr int MAX_VERTEX_NUM = 10;

    int current_cell_vertices_number = static_cast<int>(acc_numberVertexOnCell_buf[cell_id]);
    if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
        return 0.0;
    }

    // Get cell vertices
    size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
    GetCellVerticesIdx(
        cell_id,
        current_cell_vertices_number,
        current_cell_vertices_idx,
        MAX_VERTEX_NUM,
        max_edge,
        acc_verticesOnCell_buf);

    vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
    GetCellVertexPos(
        current_cell_vertex_pos,
        current_cell_vertices_idx,
        MAX_VERTEX_NUM,
        current_cell_vertices_number,
        acc_vertexCoord_buf);

    // Compute Wachspress weights (simplified version - use uniform weights for now)
    double weights[MAX_VERTEX_NUM];
    for (int i = 0; i < current_cell_vertices_number; ++i) {
        weights[i] = 1.0 / current_cell_vertices_number;
    }

    // Interpolate zTop at level 0
    double ztop = 0.0;
    for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
        int vid = static_cast<int>(current_cell_vertices_idx[v_idx]);
        if (vid >= 0 && vid < vertex_size) {
            ztop += weights[v_idx] * acc_cellVertexZTop_buf[vid * 1 + 0];  // Level 0
        }
    }

    return ztop;
}

SYCL_EXTERNAL
double SYCLKernel::CalcZTopAtBottomSYCL(
    const vec3& pos,
    int cell_id,
    int max_edge,
    int vertex_size,
    int ztop_layer,
    sycl::accessor<size_t, 1, sycl::access::mode::read> acc_numberVertexOnCell_buf,
    sycl::accessor<size_t, 1, sycl::access::mode::read> acc_verticesOnCell_buf,
    sycl::accessor<vec3, 1, sycl::access::mode::read> acc_vertexCoord_buf,
    sycl::accessor<double, 1, sycl::access::mode::read> acc_cellVertexZTop_buf)
{
    constexpr int MAX_VERTEX_NUM = 10;

    int current_cell_vertices_number = static_cast<int>(acc_numberVertexOnCell_buf[cell_id]);
    if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
        return 0.0;
    }

    // Get cell vertices
    size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
    GetCellVerticesIdx(
        cell_id,
        current_cell_vertices_number,
        current_cell_vertices_idx,
        MAX_VERTEX_NUM,
        max_edge,
        acc_verticesOnCell_buf);

    vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
    GetCellVertexPos(
        current_cell_vertex_pos,
        current_cell_vertices_idx,
        MAX_VERTEX_NUM,
        current_cell_vertices_number,
        acc_vertexCoord_buf);

    // Compute Wachspress weights (simplified version - use uniform weights for now)
    double weights[MAX_VERTEX_NUM];
    for (int i = 0; i < current_cell_vertices_number; ++i) {
        weights[i] = 1.0 / current_cell_vertices_number;
    }

    // Interpolate zTop at bottom level
    double ztop = 0.0;
    int bottom_level = ztop_layer - 1;
    for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
        int vid = static_cast<int>(current_cell_vertices_idx[v_idx]);
        if (vid >= 0 && vid < vertex_size) {
            ztop += weights[v_idx] * acc_cellVertexZTop_buf[vid * ztop_layer + bottom_level];
        }
    }

    return ztop;
}