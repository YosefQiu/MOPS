#include "CPU/TBB/MPASOSolutionTBB.h"

#include "Utils/GeoConverter.hpp"
#include "Utils/Interpolation.hpp"
#include <tbb/parallel_for.h>

namespace MOPS::CPU::TBBBackend {

void CalcCellVertexZtop(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_center_ztop,
    std::vector<double>& cell_vertex_ztop,
    const std::vector<size_t>& grid_info)
{
    (void)cells_size;
    const int CELL_SIZE = static_cast<int>(grid_info[0]);
    const int VERTLEVELS = static_cast<int>(grid_info[4]);
    const int vertex_size = static_cast<int>(grid->vertexCoord_vec.size());

    tbb::parallel_for(0, vertex_size * total_ztop_layer, [&](int idx) {
        const int vertex_id = idx / total_ztop_layer;
        const int current_layer = idx % total_ztop_layer;

        size_t tmp_cell_id[3];
        tmp_cell_id[0] = grid->cellsOnVertex_vec[3 * vertex_id + 0] - 1;
        tmp_cell_id[1] = grid->cellsOnVertex_vec[3 * vertex_id + 1] - 1;
        tmp_cell_id[2] = grid->cellsOnVertex_vec[3 * vertex_id + 2] - 1;

        bool bBoundary = false;
        double tmp_cell_center_ztop[3] = {0.0, 0.0, 0.0};
        for (int t = 0; t < 3; ++t) {
            if (tmp_cell_id[t] > static_cast<size_t>(CELL_SIZE + 1)) {
                bBoundary = true;
            } else {
                const auto ztop_idx = VERTLEVELS * static_cast<int>(tmp_cell_id[t]) + current_layer;
                tmp_cell_center_ztop[t] = cell_center_ztop[ztop_idx];
            }
        }

        double out_val = 0.0;
        if (!bBoundary) {
            double u, v, w;
            vec3 p1 = grid->cellCoord_vec[tmp_cell_id[0]];
            vec3 p2 = grid->cellCoord_vec[tmp_cell_id[1]];
            vec3 p3 = grid->cellCoord_vec[tmp_cell_id[2]];
            Interpolator::TRIANGLE tri(p1, p2, p3);
            Interpolator::calcTriangleBarycentric(grid->vertexCoord_vec[vertex_id], &tri, u, v, w);
            out_val = u * tmp_cell_center_ztop[0] + v * tmp_cell_center_ztop[1] + w * tmp_cell_center_ztop[2];
        }

        cell_vertex_ztop[vertex_id * total_ztop_layer + current_layer] = out_val;
    });
}

void CalcCellCenterToVertex(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_center_attr,
    std::vector<double>& cell_vertex_attr,
    const std::vector<size_t>& grid_info)
{
    (void)cells_size;
    const int CELL_SIZE = static_cast<int>(grid_info[0]);
    const int VERTLEVELS = static_cast<int>(grid_info[4]);
    const int vertex_size = static_cast<int>(grid->vertexCoord_vec.size());

    tbb::parallel_for(0, vertex_size * total_ztop_layer, [&](int idx) {
        const int vertex_id = idx / total_ztop_layer;
        const int current_layer = idx % total_ztop_layer;

        size_t tmp_cell_id[3];
        tmp_cell_id[0] = grid->cellsOnVertex_vec[3 * vertex_id + 0] - 1;
        tmp_cell_id[1] = grid->cellsOnVertex_vec[3 * vertex_id + 1] - 1;
        tmp_cell_id[2] = grid->cellsOnVertex_vec[3 * vertex_id + 2] - 1;

        bool bBoundary = false;
        double tmp_cell_center_attr[3] = {0.0, 0.0, 0.0};
        for (int t = 0; t < 3; ++t) {
            if (tmp_cell_id[t] > static_cast<size_t>(CELL_SIZE + 1)) {
                bBoundary = true;
            } else {
                const auto attr_idx = VERTLEVELS * static_cast<int>(tmp_cell_id[t]) + current_layer;
                tmp_cell_center_attr[t] = cell_center_attr[attr_idx];
            }
        }

        double out_val = 0.0;
        if (!bBoundary) {
            double u, v, w;
            vec3 p1 = grid->cellCoord_vec[tmp_cell_id[0]];
            vec3 p2 = grid->cellCoord_vec[tmp_cell_id[1]];
            vec3 p3 = grid->cellCoord_vec[tmp_cell_id[2]];
            Interpolator::TRIANGLE tri(p1, p2, p3);
            Interpolator::calcTriangleBarycentric(grid->vertexCoord_vec[vertex_id], &tri, u, v, w);
            out_val = u * tmp_cell_center_attr[0] + v * tmp_cell_center_attr[1] + w * tmp_cell_center_attr[2];
            if (out_val < 0.0) {
                out_val = 0.0;
            }
        }

        cell_vertex_attr[vertex_id * total_ztop_layer + current_layer] = out_val;
    });
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
    (void)grid_info;
    tbb::parallel_for(0, cells_size * total_ztop_layer, [&](int idx) {
        const int cell_id = idx / total_ztop_layer;
        const int current_layer = idx % total_ztop_layer;

        vec3 velocity = {0.0, 0.0, 0.0};
        const vec3 cell_position = grid->cellCoord_vec[cell_id];
        const double zonal = cell_zonal_velocity[cell_id * total_ztop_layer + current_layer];
        const double mer = cell_meridional_velocity[cell_id * total_ztop_layer + current_layer];
        GeoConverter::convertENUVelocityToXYZ(cell_position, zonal, mer, 0.0, velocity);
        cell_center_velocity[cell_id * total_ztop_layer + current_layer] = velocity;
    });
}

void CalcCellCenterVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_normal_velocity,
    std::vector<vec3>& cell_center_velocity,
    const std::vector<size_t>& grid_info)
{
    const int CELL_SIZE = static_cast<int>(grid_info[0]);
    const int max_edge = static_cast<int>(grid_info[2]);
    const int TOTAY_ZTOP_LAYER = static_cast<int>(grid_info[4]);
    const int MAX_VERTEX_NUM = 7;

    tbb::parallel_for(0, cells_size * total_ztop_layer, [&](int idx) {
        const int cell_id = idx / total_ztop_layer;
        const int current_layer = idx % total_ztop_layer;

        vec3 cell_position = grid->cellCoord_vec[cell_id];

        size_t current_cell_vertices_number = grid->numberVertexOnCell_vec[cell_id];
        size_t current_cell_edges_id[MAX_VERTEX_NUM];
        const auto nan = std::numeric_limits<size_t>::max();
        for (size_t k = 0; k < current_cell_vertices_number; ++k) {
            current_cell_edges_id[k] = grid->edgesOnCell_vec[cell_id * max_edge + k] - 1;
        }
        for (size_t k = current_cell_vertices_number; k < MAX_VERTEX_NUM; ++k) {
            current_cell_edges_id[k] = nan;
        }

        vec3 up = cell_position / MOPS_LENGTH(cell_position);
        vec3 k = vec3(0.0, 0.0, 1.0);
        vec3 east = MOPS_CROSS(k, up);
        if (MOPS_LENGTH(east) < 1e-6) {
            east = MOPS_CROSS(vec3(0.0, 1.0, 0.0), up);
        }
        east = east / MOPS_LENGTH(east);
        vec3 north = MOPS_CROSS(up, east);

        double planeBasisVector[2][3] = {
            {east.x(), east.y(), east.z()},
            {north.x(), north.y(), north.z()}
        };
        double cellCenter[3] = {cell_position.x(), cell_position.y(), cell_position.z()};
        int pointCount = MAX_VERTEX_NUM;
        double edge_center[MAX_VERTEX_NUM][3] = {{0.0}};
        double unit_vector[MAX_VERTEX_NUM][3] = {{0.0}};
        double noraml_vel[MAX_VERTEX_NUM][1] = {{0.0}};
        double coeffs[MAX_VERTEX_NUM][3] = {{0.0}};

        for (int kidx = 0; kidx < MAX_VERTEX_NUM; ++kidx) {
            const auto edge_id = current_cell_edges_id[kidx];
            if (edge_id == nan) {
                continue;
            }
            vec3 edge_position = grid->edgeCoord_vec[edge_id];
            edge_center[kidx][0] = edge_position.x();
            edge_center[kidx][1] = edge_position.y();
            edge_center[kidx][2] = edge_position.z();

            size_t tmp_cell_id[2];
            tmp_cell_id[0] = grid->cellsOnEdge_vec[edge_id * 2 + 0] - 1;
            tmp_cell_id[1] = grid->cellsOnEdge_vec[edge_id * 2 + 1] - 1;
            const auto min_cell_id = tmp_cell_id[0] < tmp_cell_id[1] ? tmp_cell_id[0] : tmp_cell_id[1];
            const auto max_cell_id = tmp_cell_id[0] > tmp_cell_id[1] ? tmp_cell_id[0] : tmp_cell_id[1];

            vec3 normal_vector;
            double length;
            if (max_cell_id > static_cast<size_t>(CELL_SIZE)) {
                vec3 min_cell_position = grid->cellCoord_vec[min_cell_id];
                normal_vector = edge_position - min_cell_position;
                length = MOPS_LENGTH(normal_vector);
                if (length == 0.0) {
                    continue;
                }
                normal_vector /= length;
            } else {
                vec3 min_cell_position = grid->cellCoord_vec[min_cell_id];
                vec3 max_cell_position = grid->cellCoord_vec[max_cell_id];
                normal_vector = max_cell_position - min_cell_position;
                length = MOPS_LENGTH(normal_vector);
                if (length == 0.0) {
                    continue;
                }
                normal_vector /= length;
            }

            noraml_vel[kidx][0] = cell_normal_velocity[edge_id * TOTAY_ZTOP_LAYER + current_layer];
            unit_vector[kidx][0] = normal_vector.x();
            unit_vector[kidx][1] = normal_vector.y();
            unit_vector[kidx][2] = normal_vector.z();
        }

        double alpha = Interpolator::compute_alpha(edge_center, pointCount, cellCenter);
        alpha = 1.0;
        Interpolator::mpas_rbf_interp_func_3D_plane_vec_const_dir_comp_coeffs(
            pointCount,
            edge_center,
            unit_vector,
            cellCenter,
            alpha,
            planeBasisVector,
            coeffs);

        double xVel = 0.0;
        double yVel = 0.0;
        double zVel = 0.0;
        for (int kidx = 0; kidx < MAX_VERTEX_NUM; ++kidx) {
            xVel += coeffs[kidx][0] * noraml_vel[kidx][0];
            yVel += coeffs[kidx][1] * noraml_vel[kidx][0];
            zVel += coeffs[kidx][2] * noraml_vel[kidx][0];
        }

        cell_center_velocity[cell_id * TOTAY_ZTOP_LAYER + current_layer] = vec3(xVel, yVel, zVel);
    });
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
    (void)grid_info;
    tbb::parallel_for(0, vertex_size * total_ztop_layer, [&](int idx) {
        const int vertex_id = idx / total_ztop_layer;
        const int current_layer = idx % total_ztop_layer;

        vec3 velocity = {0.0, 0.0, 0.0};
        vec3 vertex_position = grid->vertexCoord_vec[vertex_id];
        const double zonal = cell_vertex_zonal_velocity[vertex_id * total_ztop_layer + current_layer];
        const double mer = cell_vertex_meridional_velocity[vertex_id * total_ztop_layer + current_layer];
        GeoConverter::convertENUVelocityToXYZ(vertex_position, zonal, mer, 0.0, velocity);
        cell_vertex_velocity[vertex_id * total_ztop_layer + current_layer] = velocity;
    });
}

void CalcCellVertexVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<vec3>& cell_center_velocity,
    std::vector<vec3>& cell_vertex_velocity,
    const std::vector<size_t>& grid_info)
{
    (void)cells_size;
    const int CELL_SIZE = static_cast<int>(grid_info[0]);
    const int VERTLEVELS = static_cast<int>(grid_info[4]);
    const int vertex_size = static_cast<int>(grid->vertexCoord_vec.size());

    tbb::parallel_for(0, vertex_size * total_ztop_layer, [&](int idx) {
        const int vertex_id = idx / total_ztop_layer;
        const int current_layer = idx % total_ztop_layer;

        size_t tmp_cell_id[3];
        tmp_cell_id[0] = grid->cellsOnVertex_vec[3 * vertex_id + 0] - 1;
        tmp_cell_id[1] = grid->cellsOnVertex_vec[3 * vertex_id + 1] - 1;
        tmp_cell_id[2] = grid->cellsOnVertex_vec[3 * vertex_id + 2] - 1;

        bool bBoundary = false;
        vec3 tmp_cell_center_vels[3] = {
            vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0)
        };
        for (int t = 0; t < 3; ++t) {
            if (tmp_cell_id[t] > static_cast<size_t>(CELL_SIZE + 1)) {
                bBoundary = true;
            } else {
                const auto vel_idx = VERTLEVELS * static_cast<int>(tmp_cell_id[t]) + current_layer;
                tmp_cell_center_vels[t] = cell_center_velocity[vel_idx];
            }
        }

        vec3 out_val = {0.0, 0.0, 0.0};
        if (!bBoundary) {
            double u, v, w;
            vec3 p1 = grid->cellCoord_vec[tmp_cell_id[0]];
            vec3 p2 = grid->cellCoord_vec[tmp_cell_id[1]];
            vec3 p3 = grid->cellCoord_vec[tmp_cell_id[2]];
            Interpolator::TRIANGLE tri(p1, p2, p3);
            Interpolator::calcTriangleBarycentric(grid->vertexCoord_vec[vertex_id], &tri, u, v, w);
            out_val = u * tmp_cell_center_vels[0] + v * tmp_cell_center_vels[1] + w * tmp_cell_center_vels[2];
        }

        cell_vertex_velocity[vertex_id * total_ztop_layer + current_layer] = out_val;
    });
}

void CalcCellVertexVertVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer_p1,
    const std::vector<double>& cell_center_vert_velocity,
    std::vector<double>& cell_vertex_vert_velocity,
    const std::vector<size_t>& grid_info)
{
    (void)cells_size;
    const int CELL_SIZE = static_cast<int>(grid_info[0]);
    const int VERTLEVELS = static_cast<int>(grid_info[5]);
    const int vertex_size = static_cast<int>(grid->vertexCoord_vec.size());

    tbb::parallel_for(0, vertex_size * total_ztop_layer_p1, [&](int idx) {
        const int vertex_id = idx / total_ztop_layer_p1;
        const int current_layer = idx % total_ztop_layer_p1;

        size_t tmp_cell_id[3];
        tmp_cell_id[0] = grid->cellsOnVertex_vec[3 * vertex_id + 0] - 1;
        tmp_cell_id[1] = grid->cellsOnVertex_vec[3 * vertex_id + 1] - 1;
        tmp_cell_id[2] = grid->cellsOnVertex_vec[3 * vertex_id + 2] - 1;

        bool bBoundary = false;
        double tmp_cell_center_vels[3] = {0.0, 0.0, 0.0};
        for (int t = 0; t < 3; ++t) {
            if (tmp_cell_id[t] > static_cast<size_t>(CELL_SIZE + 1)) {
                bBoundary = true;
            } else {
                const auto vel_idx = VERTLEVELS * static_cast<int>(tmp_cell_id[t]) + current_layer;
                tmp_cell_center_vels[t] = cell_center_vert_velocity[vel_idx];
            }
        }

        double out_val = 0.0;
        if (!bBoundary) {
            double u, v, w;
            vec3 p1 = grid->cellCoord_vec[tmp_cell_id[0]];
            vec3 p2 = grid->cellCoord_vec[tmp_cell_id[1]];
            vec3 p3 = grid->cellCoord_vec[tmp_cell_id[2]];
            Interpolator::TRIANGLE tri(p1, p2, p3);
            Interpolator::calcTriangleBarycentric(grid->vertexCoord_vec[vertex_id], &tri, u, v, w);
            out_val = u * tmp_cell_center_vels[0] + v * tmp_cell_center_vels[1] + w * tmp_cell_center_vels[2];
        }

        cell_vertex_vert_velocity[vertex_id * total_ztop_layer_p1 + current_layer] = out_val;
    });
}

} // namespace MOPS::CPU::TBBBackend
