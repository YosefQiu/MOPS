#pragma once

#include "CUDAKernel.h"
#include "VisualizerKernels_Common.cuh"
#include "CUDAConstants.cuh"

namespace MOPS {
namespace GPU {
namespace CUDABackend {
namespace Kernel {

// Use common device functions
using namespace MOPS::GPU::CUDABackend::VisualizerInternal;
using namespace MOPS::GPU::CUDABackend::Constants;

// StreamLine-specific struct and helper function
struct StreamlineVelocityState {
    vec3 h_vel;
    double v_vel;
    bool ok;
};

MOPS_DEVICE inline StreamlineVelocityState CalcVelocityAtCUDA(
    const vec3& pos,
    int cell_id,
    double current_depth,
    int actual_max_edge_size,
    int actual_vertex_size,
    int actual_ztop_layer,
    int actual_ztop_layer_p1,
    const size_t* number_vertex_on_cell,
    const size_t* vertices_on_cell,
    const vec3* vertex_coord,
    const vec3* cell_vertex_velocity,
    const double* cell_vertex_ztop,
    const double* cell_vertex_vert_velocity)
{
    if (cell_id < 0) {
        return {vec3{0.0, 0.0, 0.0}, 0.0, false};
    }
    if (actual_ztop_layer <= 1 || actual_ztop_layer > MAX_VERTICAL_LEVELS) {
        return {vec3{0.0, 0.0, 0.0}, 0.0, false};
    }

    vec3 current_position = pos;
    const int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
    if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
        return {vec3{0.0, 0.0, 0.0}, 0.0, false};
    }

    bool is_in_mesh = MOPS::CUDAKernel::IsInMesh(
        cell_id,
        actual_max_edge_size,
        current_position,
        number_vertex_on_cell,
        vertices_on_cell,
        vertex_coord);
    if (!is_in_mesh) {
        return {vec3{0.0, 0.0, 0.0}, 0.0, false};
    }

    size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
    MOPS::CUDAKernel::GetCellVerticesIdx(
        cell_id,
        current_cell_vertices_number,
        current_cell_vertices_idx,
        MAX_VERTEX_NUM,
        actual_max_edge_size,
        vertices_on_cell);

    vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
    if (!MOPS::CUDAKernel::GetCellVertexPos(
            current_cell_vertex_pos,
            current_cell_vertices_idx,
            MAX_VERTEX_NUM,
            current_cell_vertices_number,
            vertex_coord)) {
        return {vec3{0.0, 0.0, 0.0}, 0.0, false};
    }

    double current_cell_vertex_weight[MAX_VERTEX_NUM];
    for (int i = 0; i < MAX_VERTEX_NUM; ++i) {
        current_cell_vertex_weight[i] = 0.0;
    }
    Interpolator::CalcPolygonWachspress(
        current_position,
        current_cell_vertex_pos,
        current_cell_vertex_weight,
        current_cell_vertices_number);

    double current_point_ztop_vec[MAX_ZTOP_LAYERS];
    for (int k = 0; k < actual_ztop_layer; ++k) {
        double current_point_ztop_in_layer = 0.0;
        for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
            int vid = static_cast<int>(current_cell_vertices_idx[v_idx]);
            if (vid < 0 || vid >= actual_vertex_size) {
                return {vec3{0.0, 0.0, 0.0}, 0.0, false};
            }
            double ztop = cell_vertex_ztop[vid * actual_ztop_layer + k];
            current_point_ztop_in_layer += current_cell_vertex_weight[v_idx] * ztop;
        }
        current_point_ztop_vec[k] = current_point_ztop_in_layer;
    }

    for (int k = 1; k < actual_ztop_layer; ++k) {
        if (current_point_ztop_vec[k] > current_point_ztop_vec[k - 1]) {
            current_point_ztop_vec[k] = current_point_ztop_vec[k - 1] - 1e-9;
        }
    }

    const double eps = 1e-8;
    int local_layer = -1;
    if (current_depth > current_point_ztop_vec[0] + eps) {
        local_layer = 1;
    } else if (current_depth < current_point_ztop_vec[actual_ztop_layer - 1] - eps) {
        local_layer = actual_ztop_layer - 1;
    } else {
        int lo = 1;
        int hi = actual_ztop_layer - 1;
        int ans = 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            double top_i = current_point_ztop_vec[mid - 1];
            double bot_i = current_point_ztop_vec[mid];

            if (current_depth <= top_i + eps && current_depth >= bot_i - eps) {
                ans = mid;
                break;
            }

            if (current_depth > top_i + eps) {
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        if (ans < 1) {
            ans = 1;
        }
        if (ans > actual_ztop_layer - 1) {
            ans = actual_ztop_layer - 1;
        }
        local_layer = ans;
    }

    if (local_layer < 0) {
        return {vec3{0.0, 0.0, 0.0}, 0.0, false};
    }

    const double ztop_dn = current_point_ztop_vec[local_layer];
    const double ztop_up = current_point_ztop_vec[local_layer - 1];

    double x = current_depth;
    x = fmax(ztop_dn, fmin(x, ztop_up));
    const double denom = ztop_up - ztop_dn;
    if (fabs(denom) < 1e-12) {
        return {vec3{0.0, 0.0, 0.0}, 0.0, false};
    }
    const double t = (x - ztop_dn) / denom;

    vec3 current_point_vel_dn = MOPS::CUDAKernel::CalcVelocity(
        current_cell_vertices_idx,
        current_cell_vertex_weight,
        MAX_VERTEX_NUM,
        current_cell_vertices_number,
        actual_ztop_layer,
        local_layer,
        cell_vertex_velocity);

    vec3 current_point_vel_up = MOPS::CUDAKernel::CalcVelocity(
        current_cell_vertices_idx,
        current_cell_vertex_weight,
        MAX_VERTEX_NUM,
        current_cell_vertices_number,
        actual_ztop_layer,
        local_layer - 1,
        cell_vertex_velocity);

    const double vel_dn_mag = MOPS_LENGTH(current_point_vel_dn);
    const double vel_up_mag = MOPS_LENGTH(current_point_vel_up);
    if (vel_dn_mag < 1e-12 || vel_up_mag < 1e-12) {
        return {vec3{0.0, 0.0, 0.0}, 0.0, false};
    }

    vec3 final_vel;
    final_vel.x() = t * current_point_vel_up.x() + (1.0 - t) * current_point_vel_dn.x();
    final_vel.y() = t * current_point_vel_up.y() + (1.0 - t) * current_point_vel_dn.y();
    final_vel.z() = t * current_point_vel_up.z() + (1.0 - t) * current_point_vel_dn.z();

    const double vel_mag = MOPS_LENGTH(final_vel);
    if (vel_mag < 1e-12) {
        return {vec3{0.0, 0.0, 0.0}, 0.0, false};
    }

    int dn_if = local_layer;
    int up_if = (local_layer > 0) ? (local_layer - 1) : 0;
    if (dn_if >= actual_ztop_layer_p1) {
        dn_if = actual_ztop_layer_p1 - 1;
    }
    if (up_if >= actual_ztop_layer_p1) {
        up_if = actual_ztop_layer_p1 - 1;
    }

    double w_dn = MOPS::CUDAKernel::CalcAttribute(
        current_cell_vertices_idx,
        current_cell_vertex_weight,
        MAX_VERTEX_NUM,
        current_cell_vertices_number,
        actual_ztop_layer_p1,
        dn_if,
        cell_vertex_vert_velocity);

    double w_up = MOPS::CUDAKernel::CalcAttribute(
        current_cell_vertices_idx,
        current_cell_vertex_weight,
        MAX_VERTEX_NUM,
        current_cell_vertices_number,
        actual_ztop_layer_p1,
        up_if,
        cell_vertex_vert_velocity);

    double current_vertical_velocity = t * w_up + (1.0 - t) * w_dn;
    return {final_vel, current_vertical_velocity, true};
}


// ============================================================================
// StreamLine Kernel
// Extracted from MPASOVisualizerKernels.cu for better maintainability
// ============================================================================

__global__ void KernelStreamLine(
    int particle_count,
    int each_points_size,
    int times,
    int record_t,
    int delta_t,
    bool use_euler,
    int actual_cell_size,
    int actual_max_edge_size,
    int actual_vertex_size,
    int actual_ztop_layer,
    int actual_ztop_layer_p1,
    const int* default_cell_id,
    vec3* sample_points,
    vec3* write_points,
    vec3* write_vels,
    float* particle_depths,
    const vec3* vertex_coord,
    const vec3* cell_coord,
    const size_t* number_vertex_on_cell,
    const size_t* vertices_on_cell,
    const size_t* cells_on_cell,
    const vec3* cell_vertex_velocity,
    const double* cell_vertex_ztop,
    const double* cell_vertex_vert_velocity)
{
    const int global_id = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (global_id >= particle_count || each_points_size <= 0 || times <= 0) {
        return;
    }

    int run_time = 0;
    bool first_loop = true;
    bool first_vel = true;

    const int base_idx = global_id * each_points_size;
    int update_points_idx = 0;

    vec3 sample_point_position;
    vec3 new_position;
    int cell_id = -1;
    int current_cell_vertices_number = 0;  // Declare here to be accessible throughout the loop
    int cell_neig_vec[MAX_CELL_NEIGHBOR_NUM];

    for (int i = 0; i < MAX_CELL_NEIGHBOR_NUM; ++i) {
        cell_neig_vec[i] = -1;
    }

    for (int times_i = 0; times_i < times; ++times_i) {
        run_time += abs(delta_t);
        sample_point_position = sample_points[global_id];
        double current_depth = -1.0 * static_cast<double>(particle_depths[global_id]);

        int first_cell_id = default_cell_id[global_id];
        if (first_loop) {
            first_loop = false;
            cell_id = first_cell_id;
            if (cell_id < 0 || cell_id >= actual_cell_size) {
                return;
            }

            int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
            MOPS::CUDAKernel::GetCellNeighborsIdx(
                cell_id,
                current_cell_vertices_number,
                cell_neig_vec,
                MAX_CELL_NEIGHBOR_NUM,
                actual_max_edge_size,
                cells_on_cell);
            write_points[base_idx] = sample_point_position;
        } else {
            if (cell_id < 0 || cell_id >= actual_cell_size) {
                return;
            }

            current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
            double max_len = 1e300;
            for (int idx = 0; idx < current_cell_vertices_number + 1; ++idx) {
                int cid = cell_neig_vec[idx];
                if (cid < 0 || cid >= actual_cell_size) {
                    continue;
                }

                vec3 cell_center_position = cell_coord[cid];
                double len = MOPS_LENGTH(cell_center_position - sample_point_position);
                if (len < max_len) {
                    max_len = len;
                    cell_id = cid;
                }
            }

            current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
            MOPS::CUDAKernel::GetCellNeighborsIdx(
                cell_id,
                current_cell_vertices_number,
                cell_neig_vec,
                MAX_CELL_NEIGHBOR_NUM,
                actual_max_edge_size,
                cells_on_cell);
        }

        vec3 current_position = sample_point_position;
        vec3 current_horizontal_velocity = {0.0, 0.0, 0.0};
        double current_vertical_velocity = 0.0;
        double r = MOPS_LENGTH(current_position);
        vec3 rk4_next_position = current_position;

        if (use_euler) {
            StreamlineVelocityState state = CalcVelocityAtCUDA(
                current_position,
                cell_id,
                current_depth,
                actual_max_edge_size,
                actual_vertex_size,
                actual_ztop_layer,
                actual_ztop_layer_p1,
                number_vertex_on_cell,
                vertices_on_cell,
                vertex_coord,
                cell_vertex_velocity,
                cell_vertex_ztop,
                cell_vertex_vert_velocity);
            if (!state.ok) {
                return;
            }

            current_horizontal_velocity = state.h_vel;
            current_vertical_velocity = state.v_vel;
        } else {
            double dt = static_cast<double>(delta_t);

            StreamlineVelocityState s1 = CalcVelocityAtCUDA(
                current_position,
                cell_id,
                current_depth,
                actual_max_edge_size,
                actual_vertex_size,
                actual_ztop_layer,
                actual_ztop_layer_p1,
                number_vertex_on_cell,
                vertices_on_cell,
                vertex_coord,
                cell_vertex_velocity,
                cell_vertex_ztop,
                cell_vertex_vert_velocity);
            if (!s1.ok) {
                return;
            }

            vec3 p2 = AdvectOnSphereCUDA(current_position, s1.h_vel, dt * 0.5);
            StreamlineVelocityState s2 = CalcVelocityAtCUDA(
                p2,
                cell_id,
                current_depth,
                actual_max_edge_size,
                actual_vertex_size,
                actual_ztop_layer,
                actual_ztop_layer_p1,
                number_vertex_on_cell,
                vertices_on_cell,
                vertex_coord,
                cell_vertex_velocity,
                cell_vertex_ztop,
                cell_vertex_vert_velocity);
            if (!s2.ok) {
                return;
            }

            vec3 p3 = AdvectOnSphereCUDA(current_position, s2.h_vel, dt * 0.5);
            StreamlineVelocityState s3 = CalcVelocityAtCUDA(
                p3,
                cell_id,
                current_depth,
                actual_max_edge_size,
                actual_vertex_size,
                actual_ztop_layer,
                actual_ztop_layer_p1,
                number_vertex_on_cell,
                vertices_on_cell,
                vertex_coord,
                cell_vertex_velocity,
                cell_vertex_ztop,
                cell_vertex_vert_velocity);
            if (!s3.ok) {
                return;
            }

            vec3 p4 = AdvectOnSphereCUDA(current_position, s3.h_vel, dt);
            StreamlineVelocityState s4 = CalcVelocityAtCUDA(
                p4,
                cell_id,
                current_depth,
                actual_max_edge_size,
                actual_vertex_size,
                actual_ztop_layer,
                actual_ztop_layer_p1,
                number_vertex_on_cell,
                vertices_on_cell,
                vertex_coord,
                cell_vertex_velocity,
                cell_vertex_ztop,
                cell_vertex_vert_velocity);
            if (!s4.ok) {
                return;
            }

            current_horizontal_velocity = (s1.h_vel + 2.0 * s2.h_vel + 2.0 * s3.h_vel + s4.h_vel) / 6.0;
            current_vertical_velocity = (s1.v_vel + 2.0 * s2.v_vel + 2.0 * s3.v_vel + s4.v_vel) / 6.0;

            vec3 x_trial = current_position + current_horizontal_velocity * dt;
            double x_trial_len = MOPS_LENGTH(x_trial);
            if (x_trial_len > 1e-12) {
                rk4_next_position = (x_trial / x_trial_len) * r;
            } else {
                rk4_next_position = current_position;
            }
        }

        if (use_euler) {
            vec3 rotation_axis = MOPS::CUDAKernel::CalcRotationAxis(current_position, current_horizontal_velocity);
            double speed = MOPS_LENGTH(current_horizontal_velocity);
            double theta_rad = (speed * delta_t) / r;
            new_position = MOPS::CUDAKernel::CalcPositionAfterRotation(current_position, rotation_axis, theta_rad);
        } else {
            new_position = rk4_next_position;
        }

        double old_depth = static_cast<double>(particle_depths[global_id]);
        double new_depth = old_depth - current_vertical_velocity * static_cast<double>(delta_t);
        new_depth = fmax(0.0, new_depth);
        double r_new = fmax(1.0, r + current_vertical_velocity * static_cast<double>(delta_t));
        particle_depths[global_id] = static_cast<float>(new_depth);

        double new_len = MOPS_LENGTH(new_position);
        if (new_len > 1e-12) {
            new_position = (new_position / new_len) * r_new;
        }

        if (first_vel) {
            first_vel = false;
            write_vels[base_idx] = current_horizontal_velocity;
        }

        sample_points[global_id] = new_position;

        if (record_t > 0 && (run_time % record_t) == 0) {
            int write_idx = base_idx + update_points_idx;
            if (write_idx >= 0 && write_idx < ((global_id + 1) * each_points_size)) {
                write_points[write_idx] = new_position;
                write_vels[write_idx] = current_horizontal_velocity;
                update_points_idx += 1;  // Only increment if write succeeded
            }
        }
    }
}

}  // namespace Kernel
}  // namespace CUDABackend
}  // namespace GPU
}  // namespace MOPS
