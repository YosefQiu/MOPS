#pragma once

#include "Core/MPASOField.h"
#include "Utils/GeoConverter.hpp"

namespace MOPS {

class CUDAKernel
{
public:
    static void SearchKDTree(
        int* cell_id_vec,
        MPASOGrid* grid,
        int width,
        int height,
        double minLat,
        double maxLat,
        double minLon,
        double maxLon);

    MOPS_HOST_DEVICE static inline bool IsInMesh(
        int cell_id,
        int max_edge,
        vec3 current_position,
        const size_t* numberVertexOnCell,
        const size_t* verticesOnCell,
        const vec3* vertexCoord)
    {
        if (!isfinite(current_position.x()) ||
            !isfinite(current_position.y()) ||
            !isfinite(current_position.z())) {
            return false;
        }

        auto current_cell_vertices_number = numberVertexOnCell[cell_id];
        if (current_cell_vertices_number == 0) {
            return false;
        }

        for (size_t k = 0; k < current_cell_vertices_number; ++k) {
            auto a_idx = verticesOnCell[cell_id * max_edge + k] - 1;
            auto b_idx = verticesOnCell[cell_id * max_edge + ((k + 1) % current_cell_vertices_number)] - 1;

            auto a = vertexCoord[a_idx];
            auto b = vertexCoord[b_idx];
            auto surface_normal = MOPS_CROSS(a, b);
            auto direction = MOPS_DOT(surface_normal, current_position);
            if (direction < 0.0) {
                return false;
            }
        }

        return true;
    }

    MOPS_HOST_DEVICE static inline void GetCellVerticesIdx(
        int cell_id,
        int current_cell_vertices_number,
        size_t* current_cell_vertices_idx,
        const int VLA,
        const int max_edge,
        const size_t* verticesOnCell)
    {
        for (int k = 0; k < VLA; ++k) {
            current_cell_vertices_idx[k] = verticesOnCell[cell_id * max_edge + k] - 1;
        }

        const size_t nan = static_cast<size_t>(-1);
        for (int k = current_cell_vertices_number; k < VLA; ++k) {
            current_cell_vertices_idx[k] = nan;
        }
    }

    MOPS_HOST_DEVICE static inline void GetCellNeighborsIdx(
        int cell_id,
        int current_cell_vertices_number,
        int* current_cell_neighbors_idx,
        const int VLA,
        const int max_edge,
        const size_t* cells_on_cell)
    {
        if (current_cell_vertices_number > VLA) {
            return;
        }

        current_cell_neighbors_idx[0] = cell_id;
        int copyN = current_cell_vertices_number;
        if (copyN > VLA - 1) {
            copyN = VLA - 1;
        }

        for (int k = 0; k < copyN; ++k) {
            int nid1 = static_cast<int>(cells_on_cell[cell_id * max_edge + k]);
            current_cell_neighbors_idx[k] = nid1 - 1;
        }

        current_cell_neighbors_idx[copyN] = cell_id;
        for (int k = copyN + 1; k < VLA; ++k) {
            current_cell_neighbors_idx[k] = -1;
        }
    }

    MOPS_HOST_DEVICE static inline bool GetCellVertexPos(
        vec3* current_cell_vertex_pos,
        const size_t* current_cell_vertices_idx,
        const int VLA,
        int current_cell_vertices_number,
        const vec3* vertexCoord)
    {
        if (current_cell_vertices_number > VLA) {
            return false;
        }

        auto double_nan = nan("");
        vec3 vec3_nan = {double_nan, double_nan, double_nan};
        for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
            auto vid = current_cell_vertices_idx[v_idx];
            current_cell_vertex_pos[v_idx] = vertexCoord[vid];
        }
        for (auto v_idx = current_cell_vertices_number; v_idx < VLA; ++v_idx) {
            current_cell_vertex_pos[v_idx] = vec3_nan;
        }
        return true;
    }

    MOPS_HOST_DEVICE static inline vec3 CalcVelocity(
        const size_t* current_cell_vertices_idx,
        const double* current_cell_vertex_weight,
        const int VLA,
        int current_cell_vertices_number,
        int total_ztop_layer,
        int layer,
        const vec3* cellVertexVelocity)
    {
        (void)VLA;
        vec3 current_point_vel = {0.0, 0.0, 0.0};
        for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
            auto vid = current_cell_vertices_idx[v_idx];
            auto vel = cellVertexVelocity[vid * total_ztop_layer + layer];
            current_point_vel.x() += current_cell_vertex_weight[v_idx] * vel.x();
            current_point_vel.y() += current_cell_vertex_weight[v_idx] * vel.y();
            current_point_vel.z() += current_cell_vertex_weight[v_idx] * vel.z();
        }
        return current_point_vel;
    }

    MOPS_HOST_DEVICE static inline double CalcAttribute(
        const size_t* current_cell_vertices_idx,
        const double* current_cell_vertex_weight,
        const int VLA,
        int current_cell_vertices_number,
        int total_ztop_layer,
        int layer,
        const double* cellAttribute)
    {
        (void)VLA;
        double current_point_attr_value = 0.0;
        for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
            auto vid = current_cell_vertices_idx[v_idx];
            auto value = cellAttribute[vid * total_ztop_layer + layer];
            current_point_attr_value += current_cell_vertex_weight[v_idx] * value;
        }
        return current_point_attr_value;
    }

    MOPS_HOST_DEVICE static inline vec3 CalcRotationAxis(const vec3& position, const vec3& velocity)
    {
        vec3 axis;
        axis.x() = position.y() * velocity.z() - position.z() * velocity.y();
        axis.y() = position.z() * velocity.x() - position.x() * velocity.z();
        axis.z() = position.x() * velocity.y() - position.y() * velocity.x();
        return axis;
    }

    MOPS_HOST_DEVICE static inline vec3 CalcPositionAfterRotation(const vec3& position, const vec3& axis, double theta_rad)
    {
        const double cosTheta = MOPS::math::cos(theta_rad);
        const double sinTheta = MOPS::math::sin(theta_rad);

        const double axis_len = MOPS_LENGTH(axis);
        if (axis_len <= 1e-12) {
            return position;
        }

        vec3 u;
        u.x() = axis.x() / axis_len;
        u.y() = axis.y() / axis_len;
        u.z() = axis.z() / axis_len;

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
};

} // namespace MOPS
