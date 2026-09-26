#pragma once

#include "CUDAKernel.h"
#include "VisualizerKernels_Common.cuh"
#include "CUDAConstants.cuh"  // Unified CUDA constants
#include "PathLineDebug.cuh"  // Debug utilities for PathLine

namespace MOPS {
namespace GPU {
namespace CUDABackend {
namespace Kernel {

// Use common device functions
using namespace MOPS::GPU::CUDABackend::VisualizerInternal;
// Use unified CUDA constants
using namespace MOPS::GPU::CUDABackend::Constants;

// ============================================================================
// PathLine Kernel and Helper Functions
// Extracted from MPASOVisualizerKernels.cu for better maintainability
// ============================================================================

struct PathlineVelocityState {
    vec3 h_vel;
    double v_vel;
    vec3 attr;
    bool ok;
};

// Helper function: Find nearest cell edge to a given position
MOPS_DEVICE inline void FindNearestCellEdgeCUDA(
    const vec3& pos,
    int cell_id,
    int actual_max_edge_size,
    const size_t* number_vertex_on_cell,
    const size_t* vertices_on_cell,
    const vec3* vertex_coord,
    int& edge_vertex_a_idx,
    int& edge_vertex_b_idx)
{
    edge_vertex_a_idx = -1;
    edge_vertex_b_idx = -1;

    if (cell_id < 0) {
        return;
    }

    int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
    if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
        return;
    }

    // Get cell vertex indices
    size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
    MOPS::CUDAKernel::GetCellVerticesIdx(
        cell_id,
        current_cell_vertices_number,
        current_cell_vertices_idx,
        MAX_VERTEX_NUM,
        actual_max_edge_size,
        vertices_on_cell);

    // Find nearest edge by finding minimum distance from point to edge
    double min_dist = 1e300;
    for (int i = 0; i < current_cell_vertices_number; ++i) {
        int next_i = (i + 1) % current_cell_vertices_number;
        int va_idx = static_cast<int>(current_cell_vertices_idx[i]);
        int vb_idx = static_cast<int>(current_cell_vertices_idx[next_i]);

        vec3 va = vertex_coord[va_idx];
        vec3 vb = vertex_coord[vb_idx];

        // Calculate distance from pos to edge (va, vb)
        // Using point-to-line-segment distance on the sphere
        vec3 edge_vec = vb - va;
        vec3 pos_vec = pos - va;

        double edge_len_sq = MOPS_DOT(edge_vec, edge_vec);
        if (edge_len_sq < 1e-12) {
            continue;
        }

        double t = MOPS_DOT(pos_vec, edge_vec) / edge_len_sq;
        t = fmax(0.0, fmin(1.0, t));  // clamp to [0, 1]

        vec3 closest_point = va + edge_vec * t;
        double dist = MOPS_LENGTH(pos - closest_point);

        if (dist < min_dist) {
            min_dist = dist;
            edge_vertex_a_idx = va_idx;
            edge_vertex_b_idx = vb_idx;
        }
    }
}

MOPS_DEVICE inline bool TryMultipleRandomDirectionsCUDA(
    const vec3& current_position,
    int cell_id,
    int global_id,
    int i_step,
    double r,
    double r_new,
    double delta_t,
    int actual_max_edge_size,
    const size_t* number_vertex_on_cell,
    const size_t* vertices_on_cell,
    const vec3* vertex_coord,
    vec3& out_position,
    double& out_kick_speed)
{
    // Progressive kick speeds: start small, increase if needed
    constexpr int MAX_ATTEMPTS = 20;
    constexpr double BASE_KICK_SPEED = 0.01;  // 1 cm/s

    for (int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
        // Increase kick speed every 5 attempts
        double kick_speed = BASE_KICK_SPEED * (1.0 + (attempt / 5) * 2.0);

        // Generate random velocity with different seed for each attempt
        vec3 random_vel = GenerateRandomTangentVelocityCUDA(
            current_position,
            global_id,
            i_step,
            kick_speed,
            attempt);  // Different attempt number = different random direction

        // Calculate new position with random velocity
        vec3 random_rotation_axis = MOPS::CUDAKernel::CalcRotationAxis(
            current_position,
            random_vel);
        double random_theta_rad = (kick_speed * delta_t) / r;

        vec3 random_candidate = MOPS::CUDAKernel::CalcPositionAfterRotation(
            current_position,
            random_rotation_axis,
            random_theta_rad);

        // Rescale to correct radius
        double random_len = MOPS_LENGTH(random_candidate);
        if (random_len > 1e-12) {
            random_candidate = (random_candidate / random_len) * r_new;
        }

        // Validate random direction - check if it's in the SAME cell first (fast)
        bool random_is_valid = MOPS::CUDAKernel::IsInMesh(
            cell_id,
            actual_max_edge_size,
            random_candidate,
            number_vertex_on_cell,
            vertices_on_cell,
            vertex_coord);

        if (random_is_valid) {
            out_position = random_candidate;
            out_kick_speed = kick_speed;
            return true;
        }

        // If not in same cell, try a few more attempts before giving up
        // This allows particles to potentially move to neighboring cells
    }

    // All attempts failed
    return false;
}
MOPS_DEVICE inline PathlineVelocityState CalcVelocityAtPathlineCUDA(
    const vec3& pos,
    int cell_id,
    double current_depth,
    double alpha,
    int actual_max_edge_size,
    int actual_vertex_size,
    int actual_ztop_layer,
    int actual_ztop_layer_p1,
    bool has_double_attributes,
    int attr_count,
    const size_t* number_vertex_on_cell,
    const size_t* vertices_on_cell,
    const vec3* vertex_coord,
    const vec3* cell_vertex_velocity_front,
    const vec3* cell_vertex_velocity_back,
    const double* cell_vertex_ztop_front,
    const double* cell_vertex_ztop_back,
    const double* cell_vertex_vert_velocity_front,
    const double* cell_vertex_vert_velocity_back,
    const double* attr0_front,
    const double* attr1_front,
    const double* attr0_back,
    const double* attr1_back,
    const int* max_level_cell,
    double masked_attr_value)
{
    auto ret0 = []() -> PathlineVelocityState {
        return {vec3{0.0, 0.0, 0.0}, 0.0, vec3{0.0, 0.0, 0.0}, false};
    };

    if (cell_id < 0) {
        return ret0();
    }
    if (actual_ztop_layer <= 1 || actual_ztop_layer > MAX_VERTICAL_LEVELS) {
        return ret0();
    }

    vec3 current_position = pos;
    int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
    if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
        return ret0();
    }

    bool is_in_mesh = MOPS::CUDAKernel::IsInMesh(
        cell_id,
        actual_max_edge_size,
        current_position,
        number_vertex_on_cell,
        vertices_on_cell,
        vertex_coord);
    if (!is_in_mesh) {
        return ret0();
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
        return ret0();
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

    double current_point_ztop_front_vec[MAX_ZTOP_LAYERS];
    double current_point_ztop_back_vec[MAX_ZTOP_LAYERS];
    for (int k = 0; k < actual_ztop_layer; ++k) {
        double current_point_ztop_in_layer_front = 0.0;
        double current_point_ztop_in_layer_back = 0.0;
        for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
            int vid = static_cast<int>(current_cell_vertices_idx[v_idx]);
            if (vid < 0 || vid >= actual_vertex_size) {
                return ret0();
            }

            double ztop_front = cell_vertex_ztop_front[vid * actual_ztop_layer + k];
            double ztop_back = cell_vertex_ztop_back[vid * actual_ztop_layer + k];
            current_point_ztop_in_layer_front += current_cell_vertex_weight[v_idx] * ztop_front;
            current_point_ztop_in_layer_back += current_cell_vertex_weight[v_idx] * ztop_back;
        }
        current_point_ztop_front_vec[k] = current_point_ztop_in_layer_front;
        current_point_ztop_back_vec[k] = current_point_ztop_in_layer_back;
    }

    for (int k = 1; k < actual_ztop_layer; ++k) {
        if (current_point_ztop_front_vec[k] > current_point_ztop_front_vec[k - 1]) {
            current_point_ztop_front_vec[k] = current_point_ztop_front_vec[k - 1] - 1e-9;
        }
        if (current_point_ztop_back_vec[k] > current_point_ztop_back_vec[k - 1]) {
            current_point_ztop_back_vec[k] = current_point_ztop_back_vec[k - 1] - 1e-9;
        }
    }

    const double eps = 1e-8;
    int local_layer_front = -1;
    int local_layer_back = -1;
    bool skip_loop_front = false;
    bool skip_loop_back = false;

    if (current_depth > current_point_ztop_front_vec[0] + eps) {
        local_layer_front = 0;
        skip_loop_front = true;
    } else if (current_depth < current_point_ztop_front_vec[actual_ztop_layer - 1] - eps) {
        local_layer_front = actual_ztop_layer - 1;
        skip_loop_front = true;
    }

    if (current_depth > current_point_ztop_back_vec[0] + eps) {
        local_layer_back = 0;
        skip_loop_back = true;
    } else if (current_depth < current_point_ztop_back_vec[actual_ztop_layer - 1] - eps) {
        local_layer_back = actual_ztop_layer - 1;
        skip_loop_back = true;
    }

    if (!skip_loop_front) {
        for (int k = 1; k < actual_ztop_layer; ++k) {
            double top_i = current_point_ztop_front_vec[k - 1];
            double bot_i = current_point_ztop_front_vec[k];

            // Check if this layer is masked (zero thickness)
            double layer_thickness = top_i - bot_i;
            if (layer_thickness < 1e-6) {
                // Skip masked layers
                continue;
            }

            if (current_depth <= top_i + eps && current_depth >= bot_i - eps) {
                local_layer_front = k;
                break;
            }
        }
    }

    if (!skip_loop_back) {
        for (int k = 1; k < actual_ztop_layer; ++k) {
            double top_i = current_point_ztop_back_vec[k - 1];
            double bot_i = current_point_ztop_back_vec[k];

            // Check if this layer is masked (zero thickness)
            double layer_thickness = top_i - bot_i;
            if (layer_thickness < 1e-6) {
                // Skip masked layers
                continue;
            }

            if (current_depth <= top_i + eps && current_depth >= bot_i - eps) {
                local_layer_back = k;
                break;
            }
        }
    }

    // If no valid layer found (e.g., particle deeper than last valid layer),
    // find the last non-masked layer
    if (local_layer_front < 0) {
        for (int k = actual_ztop_layer - 1; k >= 1; --k) {
            double top_i = current_point_ztop_front_vec[k - 1];
            double bot_i = current_point_ztop_front_vec[k];
            double layer_thickness = top_i - bot_i;
            if (layer_thickness >= 1e-6) {
                local_layer_front = k;
                break;
            }
        }
    }

    if (local_layer_back < 0) {
        for (int k = actual_ztop_layer - 1; k >= 1; --k) {
            double top_i = current_point_ztop_back_vec[k - 1];
            double bot_i = current_point_ztop_back_vec[k];
            double layer_thickness = top_i - bot_i;
            if (layer_thickness >= 1e-6) {
                local_layer_back = k;
                break;
            }
        }
    }

    if (local_layer_front < 0 || local_layer_back < 0) {
        return ret0();
    }

    // Check if particle is below valid ocean depth (layer >= max_level_cell)
    // If so, pull particle up to max_level_cell - 1
    if (max_level_cell != nullptr) {
        int max_valid_layer = max_level_cell[cell_id];
        if (max_valid_layer > 0 && max_valid_layer <= actual_ztop_layer) {
            if (local_layer_front >= max_valid_layer) {
                local_layer_front = max_valid_layer - 1;
                if (local_layer_front < 1) local_layer_front = 1;
            }
            if (local_layer_back >= max_valid_layer) {
                local_layer_back = max_valid_layer - 1;
                if (local_layer_back < 1) local_layer_back = 1;
            }
        }
    }

    // CRITICAL: Check if lower layer (local_layer) is sea bottom (masked layer)
    // If so, MOVE the particle UP until we find a valid layer pair
    //
    // Original: particle between layer (k-1, k), but k is masked (seafloor)
    // Fixed:    move particle to between layer (k-2, k-1), both valid
    //
    // This prevents interpolation with zero-velocity masked layers

    // Check front field - keep moving up until we find a non-masked LOWER layer
    // We interpolate between layer (local_layer-1) and layer (local_layer)
    // Need to check if the LOWER layer (local_layer) is masked
    #ifdef __CUDA_ARCH__
    int original_layer_front = local_layer_front;
    #endif

    while (local_layer_front > 1 && local_layer_front < actual_ztop_layer - 1) {
        // Check the thickness of the LOWER layer (local_layer)
        double ztop_lower_top = current_point_ztop_front_vec[local_layer_front];
        double ztop_lower_bot = current_point_ztop_front_vec[local_layer_front + 1];
        double lower_layer_thickness = ztop_lower_top - ztop_lower_bot;

        #ifdef __CUDA_ARCH__
        // if (local_layer_front >= 50) {
        //     printf("[LAYER CHECK] Lower layer %d: thickness=%.6e\n",
        //            local_layer_front, lower_layer_thickness);
        // }
        #endif

        // Check if lower layer should be considered seafloor
        bool is_masked = (fabs(lower_layer_thickness) < 1e-6);
        bool is_near_seafloor = false;

        // If not completely masked, check velocity to detect near-seafloor layers
        if (!is_masked && lower_layer_thickness < 50.0) {  // Check layers with thickness < 50m
            vec3 lower_vel = MOPS::CUDAKernel::CalcVelocity(
                current_cell_vertices_idx,
                current_cell_vertex_weight,
                MAX_VERTEX_NUM,
                current_cell_vertices_number,
                actual_ztop_layer,
                local_layer_front,
                cell_vertex_velocity_front);
            double lower_vel_mag = MOPS_LENGTH(lower_vel);
            is_near_seafloor = (lower_vel_mag < 5e-4);  // < 0.5 mm/s

            #ifdef __CUDA_ARCH__
            if (local_layer_front >= 50) {
                // printf("  -> Checking vel: %.6e, is_near_seafloor=%d\n",
                //        lower_vel_mag, is_near_seafloor ? 1 : 0);
            }
            #endif
        }

        if (is_masked || is_near_seafloor) {
            // Lower layer is seafloor or near-seafloor - move particle up one layer
            #ifdef __CUDA_ARCH__
            if (local_layer_front >= 50) {
                // printf("[BOTTOM BOUNDARY] Layer %d seafloor, moving up\n", local_layer_front);
            }
            #endif
            local_layer_front = local_layer_front - 1;
        } else {
            // Found a valid lower layer
            break;
        }
    }

    #ifdef __CUDA_ARCH__
    if (original_layer_front >= 50 && original_layer_front != local_layer_front) {
        // printf("[BOTTOM BOUNDARY] Final: moved from layer %d to %d\n",
        //        original_layer_front, local_layer_front);
    }
    #endif

    // Check back field - keep moving up until we find a valid LOWER layer
    while (local_layer_back > 1 && local_layer_back < actual_ztop_layer - 1) {
        // Check the thickness of the LOWER layer (local_layer_back)
        double ztop_lower_top = current_point_ztop_back_vec[local_layer_back];
        double ztop_lower_bot = current_point_ztop_back_vec[local_layer_back + 1];
        double lower_layer_thickness = ztop_lower_top - ztop_lower_bot;

        // Check if lower layer should be considered seafloor
        bool is_masked = (fabs(lower_layer_thickness) < 1e-6);
        bool is_near_seafloor = false;

        // If not completely masked, check velocity to detect near-seafloor layers
        if (!is_masked && lower_layer_thickness < 50.0) {  // Check layers with thickness < 50m
            vec3 lower_vel = MOPS::CUDAKernel::CalcVelocity(
                current_cell_vertices_idx,
                current_cell_vertex_weight,
                MAX_VERTEX_NUM,
                current_cell_vertices_number,
                actual_ztop_layer,
                local_layer_back,
                cell_vertex_velocity_back);
            double lower_vel_mag = MOPS_LENGTH(lower_vel);
            is_near_seafloor = (lower_vel_mag < 5e-4);
        }

        if (is_masked || is_near_seafloor) {
            // Lower layer is seafloor or near-seafloor - move particle up one layer
            local_layer_back = local_layer_back - 1;
        } else {
            // Found a valid lower layer
            break;
        }
    }

    // Now get the layer bounds for interpolation (both layers should be valid)
    double ztop_layer_front_dn = current_point_ztop_front_vec[local_layer_front];
    double ztop_layer_front_up = current_point_ztop_front_vec[local_layer_front - 1];
    double ztop_layer_back_dn = current_point_ztop_back_vec[local_layer_back];
    double ztop_layer_back_up = current_point_ztop_back_vec[local_layer_back - 1];

    // Calculate interpolation weights
    double layer_thickness_front = ztop_layer_front_up - ztop_layer_front_dn;
    double layer_thickness_back = ztop_layer_back_up - ztop_layer_back_dn;

    if (fabs(layer_thickness_front) < 1e-12 || fabs(layer_thickness_back) < 1e-12) {
        return ret0();
    }

    double x_front = current_depth;
    x_front = fmax(ztop_layer_front_dn, fmin(x_front, ztop_layer_front_up));
    double t_front = (x_front - ztop_layer_front_dn) / layer_thickness_front;

    double x_back = current_depth;
    x_back = fmax(ztop_layer_back_dn, fmin(x_back, ztop_layer_back_up));
    double t_back = (x_back - ztop_layer_back_dn) / layer_thickness_back;

    // DEBUG: Check ALL layers for this cell
    // For single particle runs, show all layers once
    static int all_layers_debug_count = 0;
    if (all_layers_debug_count < 1) {  // Show once
        all_layers_debug_count++;
        DEBUG_PATHLINE("\n[ALL LAYERS VELOCITY DEBUG] Cell %d:", cell_id);
        DEBUG_PATHLINE("  Total layers: %d", actual_ztop_layer);
        DEBUG_PATHLINE("  Current particle layer: %d", local_layer_front);

        // For each layer, compute interpolated velocity
        for (int layer = 0; layer < actual_ztop_layer; ++layer) {
            vec3 layer_vel = {0.0, 0.0, 0.0};

            // Interpolate velocity at this layer using Wachspress weights
            for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
                int vid = static_cast<int>(current_cell_vertices_idx[v_idx]);
                vec3 vertex_vel = cell_vertex_velocity_front[vid * actual_ztop_layer + layer];
                layer_vel.x() += current_cell_vertex_weight[v_idx] * vertex_vel.x();
                layer_vel.y() += current_cell_vertex_weight[v_idx] * vertex_vel.y();
                layer_vel.z() += current_cell_vertex_weight[v_idx] * vertex_vel.z();
            }

            double vel_mag = sqrt(layer_vel.x()*layer_vel.x() +
                                 layer_vel.y()*layer_vel.y() +
                                 layer_vel.z()*layer_vel.z());

            // Print with marker for current particle layer
            if (layer == local_layer_front) {
        DEBUG_PATHLINE("  Layer %2d: vel_mag=%.6e m/s  ← PARTICLE HERE (dn)", layer, vel_mag);
            } else if (layer == local_layer_front - 1) {
        DEBUG_PATHLINE("  Layer %2d: vel_mag=%.6e m/s  ← (up)", layer, vel_mag);
            } else if (vel_mag == 0.0) {
        DEBUG_PATHLINE("  Layer %2d: vel_mag=%.6e m/s  (ZERO)", layer, vel_mag);
            } else {
        DEBUG_PATHLINE("  Layer %2d: vel_mag=%.6e m/s", layer, vel_mag);
            }
        }
        DEBUG_PATHLINE("");
        // all_layers_debug_done = 1;  // Disabled
    }

    vec3 current_point_vel_dn_front = MOPS::CUDAKernel::CalcVelocity(
        current_cell_vertices_idx,
        current_cell_vertex_weight,
        MAX_VERTEX_NUM,
        current_cell_vertices_number,
        actual_ztop_layer,
        local_layer_front,
        cell_vertex_velocity_front);

    vec3 current_point_vel_up_front = MOPS::CUDAKernel::CalcVelocity(
        current_cell_vertices_idx,
        current_cell_vertex_weight,
        MAX_VERTEX_NUM,
        current_cell_vertices_number,
        actual_ztop_layer,
        local_layer_front - 1,
        cell_vertex_velocity_front);

    vec3 final_vel_front;
    final_vel_front.x() = t_front * current_point_vel_up_front.x() + (1.0 - t_front) * current_point_vel_dn_front.x();
    final_vel_front.y() = t_front * current_point_vel_up_front.y() + (1.0 - t_front) * current_point_vel_dn_front.y();
    final_vel_front.z() = t_front * current_point_vel_up_front.z() + (1.0 - t_front) * current_point_vel_dn_front.z();

    vec3 current_point_vel_dn_back = MOPS::CUDAKernel::CalcVelocity(
        current_cell_vertices_idx,
        current_cell_vertex_weight,
        MAX_VERTEX_NUM,
        current_cell_vertices_number,
        actual_ztop_layer,
        local_layer_back,
        cell_vertex_velocity_back);

    vec3 current_point_vel_up_back = MOPS::CUDAKernel::CalcVelocity(
        current_cell_vertices_idx,
        current_cell_vertex_weight,
        MAX_VERTEX_NUM,
        current_cell_vertices_number,
        actual_ztop_layer,
        local_layer_back - 1,
        cell_vertex_velocity_back);

    vec3 final_vel_back;
    final_vel_back.x() = t_back * current_point_vel_up_back.x() + (1.0 - t_back) * current_point_vel_dn_back.x();
    final_vel_back.y() = t_back * current_point_vel_up_back.y() + (1.0 - t_back) * current_point_vel_dn_back.y();
    final_vel_back.z() = t_back * current_point_vel_up_back.z() + (1.0 - t_back) * current_point_vel_dn_back.z();

    vec3 current_horizontal_velocity;
    current_horizontal_velocity.x() = alpha * final_vel_back.x() + (1.0 - alpha) * final_vel_front.x();
    current_horizontal_velocity.y() = alpha * final_vel_back.y() + (1.0 - alpha) * final_vel_front.y();
    current_horizontal_velocity.z() = alpha * final_vel_back.z() + (1.0 - alpha) * final_vel_front.z();

    // CRITICAL: Check for NaN/Inf in calculated velocity
    bool vel_is_invalid = !isfinite(current_horizontal_velocity.x()) ||
                          !isfinite(current_horizontal_velocity.y()) ||
                          !isfinite(current_horizontal_velocity.z());

    if (vel_is_invalid) {
        printf("[NaN VELOCITY] Cell %d, Depth %.2f, Alpha %.6f\n",
               cell_id, current_depth, alpha);
        printf("  Layer: front=%d, back=%d, t_front=%.6f, t_back=%.6f\n",
               local_layer_front, local_layer_back, t_front, t_back);
        printf("  vel_up_front: (%.6e, %.6e, %.6e)\n",
               current_point_vel_up_front.x(), current_point_vel_up_front.y(),
               current_point_vel_up_front.z());
        printf("  vel_dn_front: (%.6e, %.6e, %.6e)\n",
               current_point_vel_dn_front.x(), current_point_vel_dn_front.y(),
               current_point_vel_dn_front.z());
        printf("  vel_up_back: (%.6e, %.6e, %.6e)\n",
               current_point_vel_up_back.x(), current_point_vel_up_back.y(),
               current_point_vel_up_back.z());
        printf("  vel_dn_back: (%.6e, %.6e, %.6e)\n",
               current_point_vel_dn_back.x(), current_point_vel_dn_back.y(),
               current_point_vel_dn_back.z());
        printf("  final_vel_front: (%.6e, %.6e, %.6e)\n",
               final_vel_front.x(), final_vel_front.y(), final_vel_front.z());
        printf("  final_vel_back: (%.6e, %.6e, %.6e)\n",
               final_vel_back.x(), final_vel_back.y(), final_vel_back.z());
        printf("  current_horizontal_velocity: (%.6e, %.6e, %.6e)\n",
               current_horizontal_velocity.x(), current_horizontal_velocity.y(),
               current_horizontal_velocity.z());
        DEBUG_PATHLINE("  → Using default velocity (1e-4 m/s)");

        // Use default velocity to prevent NaN propagation
        // Direction: try to use a valid component or set to zero
        current_horizontal_velocity.x() = 1e-4;
        current_horizontal_velocity.y() = 0.0;
        current_horizontal_velocity.z() = 0.0;
    }

    // DEBUG: Print velocity interpolation details (disabled by default)
    if (false) {  // Change to true and recompile to enable detailed velocity debug
        double vel_up_front_mag = sqrt(current_point_vel_up_front.x()*current_point_vel_up_front.x() +
                                       current_point_vel_up_front.y()*current_point_vel_up_front.y() +
                                       current_point_vel_up_front.z()*current_point_vel_up_front.z());
        double vel_dn_front_mag = sqrt(current_point_vel_dn_front.x()*current_point_vel_dn_front.x() +
                                       current_point_vel_dn_front.y()*current_point_vel_dn_front.y() +
                                       current_point_vel_dn_front.z()*current_point_vel_dn_front.z());
        double vel_up_back_mag = sqrt(current_point_vel_up_back.x()*current_point_vel_up_back.x() +
                                      current_point_vel_up_back.y()*current_point_vel_up_back.y() +
                                      current_point_vel_up_back.z()*current_point_vel_up_back.z());
        double vel_dn_back_mag = sqrt(current_point_vel_dn_back.x()*current_point_vel_dn_back.x() +
                                      current_point_vel_dn_back.y()*current_point_vel_dn_back.y() +
                                      current_point_vel_dn_back.z()*current_point_vel_dn_back.z());
        double final_front_mag = sqrt(final_vel_front.x()*final_vel_front.x() +
                                      final_vel_front.y()*final_vel_front.y() +
                                      final_vel_front.z()*final_vel_front.z());
        double final_back_mag = sqrt(final_vel_back.x()*final_vel_back.x() +
                                     final_vel_back.y()*final_vel_back.y() +
                                     final_vel_back.z()*final_vel_back.z());
        double final_mag = sqrt(current_horizontal_velocity.x()*current_horizontal_velocity.x() +
                               current_horizontal_velocity.y()*current_horizontal_velocity.y() +
                               current_horizontal_velocity.z()*current_horizontal_velocity.z());

        DEBUG_PATHLINE("[VELOCITY INTERPOLATION]:");
        DEBUG_PATHLINE("  Cell %d, Depth %.2f, Alpha %.6f", cell_id, current_depth, alpha);
        DEBUG_PATHLINE("  Layer: front=%d, back=%d", local_layer_front, local_layer_back);
        DEBUG_PATHLINE("  t_front=%.6f, t_back=%.6f", t_front, t_back);
        DEBUG_PATHLINE("  Front field velocities:");
        printf("    vel_up:   (%.6e, %.6e, %.6e) mag=%.6e\n",
               current_point_vel_up_front.x(), current_point_vel_up_front.y(),
               current_point_vel_up_front.z(), vel_up_front_mag);
        printf("    vel_dn:   (%.6e, %.6e, %.6e) mag=%.6e\n",
               current_point_vel_dn_front.x(), current_point_vel_dn_front.y(),
               current_point_vel_dn_front.z(), vel_dn_front_mag);
        DEBUG_PATHLINE("    final_front: mag=%.6e", final_front_mag);
        DEBUG_PATHLINE("  Back field velocities:");
        printf("    vel_up:   (%.6e, %.6e, %.6e) mag=%.6e\n",
               current_point_vel_up_back.x(), current_point_vel_up_back.y(),
               current_point_vel_up_back.z(), vel_up_back_mag);
        printf("    vel_dn:   (%.6e, %.6e, %.6e) mag=%.6e\n",
               current_point_vel_dn_back.x(), current_point_vel_dn_back.y(),
               current_point_vel_dn_back.z(), vel_dn_back_mag);
        DEBUG_PATHLINE("    final_back: mag=%.6e", final_back_mag);
        DEBUG_PATHLINE("  Time interpolation (alpha=%.6f):", alpha);
        printf("    Final velocity: (%.6e, %.6e, %.6e) mag=%.6e\n",
               current_horizontal_velocity.x(), current_horizontal_velocity.y(),
               current_horizontal_velocity.z(), final_mag);

        // debug_call_count++;  // Disabled
    }

    int dn_if_front = local_layer_front;
    int up_if_front = (local_layer_front > 0) ? (local_layer_front - 1) : 0;
    int dn_if_back = local_layer_back;
    int up_if_back = (local_layer_back > 0) ? (local_layer_back - 1) : 0;

    double w_dn_front = MOPS::CUDAKernel::CalcAttribute(
        current_cell_vertices_idx,
        current_cell_vertex_weight,
        MAX_VERTEX_NUM,
        current_cell_vertices_number,
        actual_ztop_layer_p1,
        dn_if_front,
        cell_vertex_vert_velocity_front);
    double w_up_front = MOPS::CUDAKernel::CalcAttribute(
        current_cell_vertices_idx,
        current_cell_vertex_weight,
        MAX_VERTEX_NUM,
        current_cell_vertices_number,
        actual_ztop_layer_p1,
        up_if_front,
        cell_vertex_vert_velocity_front);
    double w_front = t_front * w_up_front + (1.0 - t_front) * w_dn_front;

    double w_dn_back = MOPS::CUDAKernel::CalcAttribute(
        current_cell_vertices_idx,
        current_cell_vertex_weight,
        MAX_VERTEX_NUM,
        current_cell_vertices_number,
        actual_ztop_layer_p1,
        dn_if_back,
        cell_vertex_vert_velocity_back);
    double w_up_back = MOPS::CUDAKernel::CalcAttribute(
        current_cell_vertices_idx,
        current_cell_vertex_weight,
        MAX_VERTEX_NUM,
        current_cell_vertices_number,
        actual_ztop_layer_p1,
        up_if_back,
        cell_vertex_vert_velocity_back);
    double w_back = t_back * w_up_back + (1.0 - t_back) * w_dn_back;

    double current_vertical_velocity = alpha * w_back + (1.0 - alpha) * w_front;

    vec3 current_point_attr_value = {0.0, 0.0, 0.0};
    if (has_double_attributes) {
        if (attr_count >= 1 && attr0_front != nullptr && attr0_back != nullptr) {
            double attr_dn_front = MOPS::CUDAKernel::CalcAttribute(
                current_cell_vertices_idx,
                current_cell_vertex_weight,
                MAX_VERTEX_NUM,
                current_cell_vertices_number,
                actual_ztop_layer,
                local_layer_front,
                attr0_front);
            double attr_up_front = MOPS::CUDAKernel::CalcAttribute(
                current_cell_vertices_idx,
                current_cell_vertex_weight,
                MAX_VERTEX_NUM,
                current_cell_vertices_number,
                actual_ztop_layer,
                local_layer_front - 1,
                attr0_front);
            double attr_front = t_front * attr_up_front + (1.0 - t_front) * attr_dn_front;

            double attr_dn_back = MOPS::CUDAKernel::CalcAttribute(
                current_cell_vertices_idx,
                current_cell_vertex_weight,
                MAX_VERTEX_NUM,
                current_cell_vertices_number,
                actual_ztop_layer,
                local_layer_back,
                attr0_back);
            double attr_up_back = MOPS::CUDAKernel::CalcAttribute(
                current_cell_vertices_idx,
                current_cell_vertex_weight,
                MAX_VERTEX_NUM,
                current_cell_vertices_number,
                actual_ztop_layer,
                local_layer_back - 1,
                attr0_back);
            double attr_back = t_back * attr_up_back + (1.0 - t_back) * attr_dn_back;

            current_point_attr_value.x() = alpha * attr_back + (1.0 - alpha) * attr_front;
        }

        if (attr_count >= 2 && attr1_front != nullptr && attr1_back != nullptr) {
            double attr_dn_front = MOPS::CUDAKernel::CalcAttribute(
                current_cell_vertices_idx,
                current_cell_vertex_weight,
                MAX_VERTEX_NUM,
                current_cell_vertices_number,
                actual_ztop_layer,
                local_layer_front,
                attr1_front);
            double attr_up_front = MOPS::CUDAKernel::CalcAttribute(
                current_cell_vertices_idx,
                current_cell_vertex_weight,
                MAX_VERTEX_NUM,
                current_cell_vertices_number,
                actual_ztop_layer,
                local_layer_front - 1,
                attr1_front);
            double attr_front = t_front * attr_up_front + (1.0 - t_front) * attr_dn_front;

            double attr_dn_back = MOPS::CUDAKernel::CalcAttribute(
                current_cell_vertices_idx,
                current_cell_vertex_weight,
                MAX_VERTEX_NUM,
                current_cell_vertices_number,
                actual_ztop_layer,
                local_layer_back,
                attr1_back);
            double attr_up_back = MOPS::CUDAKernel::CalcAttribute(
                current_cell_vertices_idx,
                current_cell_vertex_weight,
                MAX_VERTEX_NUM,
                current_cell_vertices_number,
                actual_ztop_layer,
                local_layer_back - 1,
                attr1_back);
            double attr_back = t_back * attr_up_back + (1.0 - t_back) * attr_dn_back;

            current_point_attr_value.y() = alpha * attr_back + (1.0 - alpha) * attr_front;
        }
    }

    // DEBUG: Print interpolation details for low velocity cases (DISABLED)
    if (false) {  // Disabled - use ALL LAYERS DEBUG instead
        double h_vel_mag = MOPS_LENGTH(current_horizontal_velocity);
        double vel_up_front_mag = MOPS_LENGTH(current_point_vel_up_front);
        double vel_dn_front_mag = MOPS_LENGTH(current_point_vel_dn_front);
        double vel_up_back_mag = MOPS_LENGTH(current_point_vel_up_back);
        double vel_dn_back_mag = MOPS_LENGTH(current_point_vel_dn_back);

        printf("[LOW VEL DEBUG] Final h_vel=%.6e, depth=%.1f, alpha=%.6f\n",
               h_vel_mag, current_depth, alpha);
        printf("  Front: layer=%d, t=%.6f, vel_up=%.6e, vel_dn=%.6e\n",
               local_layer_front, t_front, vel_up_front_mag, vel_dn_front_mag);
        printf("  Back:  layer=%d, t=%.6f, vel_up=%.6e, vel_dn=%.6e\n",
               local_layer_back, t_back, vel_up_back_mag, vel_dn_back_mag);
    }

    // CRITICAL: Check for NaN/Inf/masked values in temperature and salinity attributes
    bool attr_is_invalid = !isfinite(current_point_attr_value.x()) ||
                           !isfinite(current_point_attr_value.y()) ||
                           !isfinite(current_point_attr_value.z()) ||
                           fabs(current_point_attr_value.x() - masked_attr_value) < 1e-10 ||
                           fabs(current_point_attr_value.y() - masked_attr_value) < 1e-10;

    if (attr_is_invalid) {
        printf("[INVALID ATTRIBUTE] Cell %d, Depth %.2f, Alpha %.6f\n",
               cell_id, current_depth, alpha);
        printf("  attr_value: (%.6e, %.6e, %.6e), masked_value=%.6e\n",
               current_point_attr_value.x(), current_point_attr_value.y(),
               current_point_attr_value.z(), masked_attr_value);
        DEBUG_PATHLINE("  → Using default values (temp=0, sal=0)");

        // Use default attribute values to prevent NaN/masked propagation
        current_point_attr_value.x() = 0.0;  // temperature
        current_point_attr_value.y() = 0.0;  // salinity
        current_point_attr_value.z() = 0.0;
    }

    return {current_horizontal_velocity, current_vertical_velocity, current_point_attr_value, true};
}
MOPS_DEVICE inline PathlineVelocityState CalcVelocityAtPathlineWithConstraintsCUDA(
    const vec3& pos,
    int cell_id,
    double current_depth,
    double alpha,
    int actual_max_edge_size,
    int actual_vertex_size,
    int actual_ztop_layer,
    int actual_ztop_layer_p1,
    bool has_double_attributes,
    int attr_count,
    const size_t* number_vertex_on_cell,
    const size_t* vertices_on_cell,
    const vec3* vertex_coord,
    const vec3* cell_vertex_velocity_front,
    const vec3* cell_vertex_velocity_back,
    const double* cell_vertex_ztop_front,
    const double* cell_vertex_ztop_back,
    const double* cell_vertex_vert_velocity_front,
    const double* cell_vertex_vert_velocity_back,
    const double* attr0_front,
    const double* attr1_front,
    const double* attr0_back,
    const double* attr1_back,
    const int* max_level_cell,
    double masked_attr_value)
{
    // Calculate velocity normally
    PathlineVelocityState state = CalcVelocityAtPathlineCUDA(
        pos,
        cell_id,
        current_depth,
        alpha,
        actual_max_edge_size,
        actual_vertex_size,
        actual_ztop_layer,
        actual_ztop_layer_p1,
        has_double_attributes,
        attr_count,
        number_vertex_on_cell,
        vertices_on_cell,
        vertex_coord,
        cell_vertex_velocity_front,
        cell_vertex_velocity_back,
        cell_vertex_ztop_front,
        cell_vertex_ztop_back,
        cell_vertex_vert_velocity_front,
        cell_vertex_vert_velocity_back,
        attr0_front,
        attr1_front,
        attr0_back,
        attr1_back,
        max_level_cell,
        masked_attr_value);

    if (!state.ok) {
        return state;
    }

    // Apply edge projection constraint for stuck particles
    constexpr double VELOCITY_THRESHOLD = 1e-9;
    double speed = MOPS_LENGTH(state.h_vel);

    if (speed < VELOCITY_THRESHOLD && speed > 1e-20) {  // Very small but non-zero
        // Find nearest edge
        int edge_va_idx = -1;
        int edge_vb_idx = -1;
        FindNearestCellEdgeCUDA(
            pos,
            cell_id,
            actual_max_edge_size,
            number_vertex_on_cell,
            vertices_on_cell,
            vertex_coord,
            edge_va_idx,
            edge_vb_idx);

        if (edge_va_idx >= 0 && edge_vb_idx >= 0) {
            vec3 va = vertex_coord[edge_va_idx];
            vec3 vb = vertex_coord[edge_vb_idx];

            // Project velocity onto edge
            vec3 projected_vel = ProjectVelocityOntoEdgeCUDA(state.h_vel, pos, va, vb);
            state.h_vel = projected_vel;
        }
    }

    return state;
}
__global__ void KernelPathLine(
    int particle_count,
    int each_points_size,
    int n_steps,
    int simulation_duration,
    int record_t,
    int delta_t,
    bool use_euler,
    int actual_cell_size,
    int actual_max_edge_size,
    int actual_vertex_size,
    int actual_ztop_layer,
    int actual_ztop_layer_p1,
    bool has_double_attributes,
    int attr_count,
    const int* default_cell_id,
    vec3* sample_points,
    vec3* write_points,
    vec3* write_vels,
    vec3* write_attrs,
    float* particle_depths,
    const vec3* vertex_coord,
    const vec3* cell_coord,
    const size_t* number_vertex_on_cell,
    const size_t* vertices_on_cell,
    const size_t* cells_on_cell,
    const int* max_level_cell,
    const vec3* cell_vertex_velocity_front,
    const vec3* cell_vertex_velocity_back,
    const double* cell_vertex_ztop_front,
    const double* cell_vertex_ztop_back,
    const double* cell_vertex_vert_velocity_front,
    const double* cell_vertex_vert_velocity_back,
    const double* attr0_front,
    const double* attr1_front,
    const double* attr0_back,
    const double* attr1_back,
    int* boundary_hit_count,
    double sigma_h,
    double sigma_v,
    double masked_attr_value)
{
    const int global_id = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (global_id >= particle_count || each_points_size <= 0 || n_steps <= 0) {
        return;
    }

    // Initialize boundary hit counter for this particle
    if (boundary_hit_count != nullptr) {
        boundary_hit_count[global_id] = 0;
    }

    double run_time = 0.0;
    bool first_loop = true;
    bool first_vel = true;
    bool first_attr = true;

    int base_idx = global_id * each_points_size;
    int update_points_idx = 0;

    vec3 sample_point_position;
    vec3 new_position;
    int cell_id = -1;
    int current_cell_vertices_number = 0;  // Declare here to be accessible throughout the loop
    int cell_neig_vec[MAX_CELL_NEIGHBOR_NUM];

    for (int i = 0; i < MAX_CELL_NEIGHBOR_NUM; ++i) {
        cell_neig_vec[i] = -1;
    }

    // DEBUG: Print parameters for first particle only
    if (global_id == 0) {
        // printf("[DEBUG PathLine] Particle 0:\n");
        // printf("  simulation_duration = %d seconds\n", simulation_duration);
        // printf("  delta_t = %d seconds\n", delta_t);
        // printf("  n_steps = %d\n", n_steps);
        // printf("  record_t = %d seconds\n", record_t);
        // printf("  each_points_size = %d\n", each_points_size);
    }

    for (int i_step = 0; i_step < n_steps; ++i_step) {
        double alpha = static_cast<double>(i_step) / static_cast<double>(n_steps);
        run_time += delta_t;
        (void)run_time;

        // DEBUG: Print alpha for first particle at selected steps
        if (global_id == 0 && (i_step < 5 || i_step == 595 || i_step == 596 || i_step % 1000 == 0)) {
            // printf("  [Step %d] alpha = %.6f (i_step=%d / n_steps=%d)\n",
            //        i_step, alpha, i_step, n_steps);
        }

        sample_point_position = sample_points[global_id];
        int first_cell_id = default_cell_id[global_id];
        double current_depth = -1.0 * static_cast<double>(particle_depths[global_id]);

        if (first_loop) {
            first_loop = false;
            cell_id = first_cell_id;
            if (cell_id < 0 || cell_id >= actual_cell_size) {
                return;
            }

            current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
            MOPS::CUDAKernel::GetCellNeighborsIdx(
                cell_id,
                current_cell_vertices_number,
                cell_neig_vec,
                MAX_VERTEX_NUM,
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
                vec3 p = cell_coord[cid];
                double len = MOPS_LENGTH(p - sample_point_position);
                if (len < max_len) {
                    max_len = len;
                    cell_id = cid;
                }
            }

            // BUG FIX: Update vertex count after cell_id changed
            // Must match new cell's vertex count, not old cell's
            current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);

            MOPS::CUDAKernel::GetCellNeighborsIdx(
                cell_id,
                current_cell_vertices_number,
                cell_neig_vec,
                MAX_VERTEX_NUM,
                actual_max_edge_size,
                cells_on_cell);
        }

        vec3 current_position = sample_point_position;
        double r = MOPS_LENGTH(current_position);
        vec3 rk4_next_position = current_position;
        vec3 current_horizontal_velocity = {0.0, 0.0, 0.0};
        double current_vertical_velocity = 0.0;
        vec3 current_attrs = {0.0, 0.0, 0.0};

        if (use_euler) {
            PathlineVelocityState s = CalcVelocityAtPathlineWithConstraintsCUDA(
                current_position,
                cell_id,
                current_depth,
                alpha,
                actual_max_edge_size,
                actual_vertex_size,
                actual_ztop_layer,
                actual_ztop_layer_p1,
                has_double_attributes,
                attr_count,
                number_vertex_on_cell,
                vertices_on_cell,
                vertex_coord,
                cell_vertex_velocity_front,
                cell_vertex_velocity_back,
                cell_vertex_ztop_front,
                cell_vertex_ztop_back,
                cell_vertex_vert_velocity_front,
                cell_vertex_vert_velocity_back,
                attr0_front,
                attr1_front,
                attr0_back,
                attr1_back,
                max_level_cell,
                masked_attr_value);

            current_horizontal_velocity = s.h_vel;
            current_vertical_velocity = s.v_vel;
            current_attrs = s.attr;

            // DEBUG: Print attributes (temperature/salinity) for first particle at selected steps
            #ifdef __CUDA_ARCH__
            if (global_id == 0 && (i_step < 3 || i_step == 595 || i_step == 596 || i_step % 500 == 0)) {
                // printf("  [Attrs Euler] Step %d: alpha=%.6f, T=%.4f°C, S=%.4f PSU\n",
                //        i_step, alpha, current_attrs.x(), current_attrs.y());
            }
            #endif
        } else {
            const double dt = static_cast<double>(delta_t);
            const double dalpha = dt / static_cast<double>(simulation_duration);

            double a1 = alpha;
            PathlineVelocityState s1 = CalcVelocityAtPathlineWithConstraintsCUDA(
                current_position,
                cell_id,
                current_depth,
                a1,
                actual_max_edge_size,
                actual_vertex_size,
                actual_ztop_layer,
                actual_ztop_layer_p1,
                has_double_attributes,
                attr_count,
                number_vertex_on_cell,
                vertices_on_cell,
                vertex_coord,
                cell_vertex_velocity_front,
                cell_vertex_velocity_back,
                cell_vertex_ztop_front,
                cell_vertex_ztop_back,
                cell_vertex_vert_velocity_front,
                cell_vertex_vert_velocity_back,
                attr0_front,
                attr1_front,
                attr0_back,
                attr1_back,
                max_level_cell,
                masked_attr_value);

            // PERTURBATION: Add random velocity if speed is too low (k1 stage)
            constexpr double VEL_THRESH = 1e-4;      // 0.1 mm/s

            double s1_h_mag = MOPS_LENGTH(s1.h_vel);
            if (s1_h_mag < VEL_THRESH) {
                vec3 perturb_h = GenerateRandomTangentVelocityCUDA(
                    current_position, global_id, i_step * 4 + 1, sigma_h);
                s1.h_vel = s1.h_vel + perturb_h;
            }
            if (fabs(s1.v_vel) < VEL_THRESH) {
                float r1 = random_float(global_id, i_step * 4 + 1, 10);
                float r2 = random_float(global_id, i_step * 4 + 1, 20);
                s1.v_vel += sigma_v * (r1 - 0.5 + r2 - 0.5);
            }

            vec3 p2 = AdvectOnSphereCUDA(current_position, s1.h_vel, dt * 0.5);
            double a2 = a1 + 0.5 * dalpha;
            if (a2 > 1.0) a2 = 1.0;
            if (a2 < 0.0) a2 = 0.0;

            // Try original cell first
            PathlineVelocityState s2 = CalcVelocityAtPathlineWithConstraintsCUDA(
                p2, cell_id, current_depth, a2,
                actual_max_edge_size, actual_vertex_size,
                actual_ztop_layer, actual_ztop_layer_p1,
                has_double_attributes, attr_count,
                number_vertex_on_cell, vertices_on_cell, vertex_coord,
                cell_vertex_velocity_front, cell_vertex_velocity_back,
                cell_vertex_ztop_front, cell_vertex_ztop_back,
                cell_vertex_vert_velocity_front, cell_vertex_vert_velocity_back,
                attr0_front, attr1_front, attr0_back, attr1_back, max_level_cell,
                masked_attr_value);

            // If failed, try neighbor cells
            if (!s2.ok) {
                for (int i = 0; i < current_cell_vertices_number + 1; ++i) {
                    int neighbor_id = cell_neig_vec[i];
                    if (neighbor_id < 0 || neighbor_id >= actual_cell_size || neighbor_id == cell_id) {
                        continue;
                    }

                    PathlineVelocityState s2_neighbor = CalcVelocityAtPathlineWithConstraintsCUDA(
                        p2, neighbor_id, current_depth, a2,
                        actual_max_edge_size, actual_vertex_size,
                        actual_ztop_layer, actual_ztop_layer_p1,
                        has_double_attributes, attr_count,
                        number_vertex_on_cell, vertices_on_cell, vertex_coord,
                        cell_vertex_velocity_front, cell_vertex_velocity_back,
                        cell_vertex_ztop_front, cell_vertex_ztop_back,
                        cell_vertex_vert_velocity_front, cell_vertex_vert_velocity_back,
                        attr0_front, attr1_front, attr0_back, attr1_back, max_level_cell,
                masked_attr_value);

                    if (s2_neighbor.ok) {
                        s2 = s2_neighbor;
                        if (global_id == 0) {
                            DEBUG_PATHLINE_STEP(i_step, "  [RK4 k2 NEIGHBOR] i_step=%d: found in neighbor cell %d", i_step, neighbor_id);
                        }
                        break;
                    }
                }
            }

            // PERTURBATION: Add random velocity if speed is too low (k2 stage)
            double s2_h_mag = MOPS_LENGTH(s2.h_vel);
            if (s2_h_mag < VEL_THRESH) {
                vec3 perturb_h = GenerateRandomTangentVelocityCUDA(
                    p2, global_id, i_step * 4 + 2, sigma_h);
                s2.h_vel = s2.h_vel + perturb_h;
            }
            if (fabs(s2.v_vel) < VEL_THRESH) {
                float r1 = random_float(global_id, i_step * 4 + 2, 10);
                float r2 = random_float(global_id, i_step * 4 + 2, 20);
                s2.v_vel += sigma_v * (r1 - 0.5 + r2 - 0.5);
            }

            vec3 p3 = AdvectOnSphereCUDA(current_position, s2.h_vel, dt * 0.5);
            double a3 = a1 + 0.5 * dalpha;
            if (a3 > 1.0) a3 = 1.0;
            if (a3 < 0.0) a3 = 0.0;
            PathlineVelocityState s3 = CalcVelocityAtPathlineWithConstraintsCUDA(
                p3,
                cell_id,
                current_depth,
                a3,
                actual_max_edge_size,
                actual_vertex_size,
                actual_ztop_layer,
                actual_ztop_layer_p1,
                has_double_attributes,
                attr_count,
                number_vertex_on_cell,
                vertices_on_cell,
                vertex_coord,
                cell_vertex_velocity_front,
                cell_vertex_velocity_back,
                cell_vertex_ztop_front,
                cell_vertex_ztop_back,
                cell_vertex_vert_velocity_front,
                cell_vertex_vert_velocity_back,
                attr0_front,
                attr1_front,
                attr0_back,
                attr1_back,
                max_level_cell,
                masked_attr_value);

            // PERTURBATION: Add random velocity if speed is too low (k3 stage)
            double s3_h_mag = MOPS_LENGTH(s3.h_vel);
            if (s3_h_mag < VEL_THRESH) {
                vec3 perturb_h = GenerateRandomTangentVelocityCUDA(
                    p3, global_id, i_step * 4 + 3, sigma_h);
                s3.h_vel = s3.h_vel + perturb_h;
            }
            if (fabs(s3.v_vel) < VEL_THRESH) {
                float r1 = random_float(global_id, i_step * 4 + 3, 10);
                float r2 = random_float(global_id, i_step * 4 + 3, 20);
                s3.v_vel += sigma_v * (r1 - 0.5 + r2 - 0.5);
            }

            vec3 p4 = AdvectOnSphereCUDA(current_position, s3.h_vel, dt);
            double a4 = a1 + dalpha;
            if (a4 > 1.0) a4 = 1.0;
            if (a4 < 0.0) a4 = 0.0;

            // Try original cell first
            PathlineVelocityState s4 = CalcVelocityAtPathlineWithConstraintsCUDA(
                p4, cell_id, current_depth, a4,
                actual_max_edge_size, actual_vertex_size,
                actual_ztop_layer, actual_ztop_layer_p1,
                has_double_attributes, attr_count,
                number_vertex_on_cell, vertices_on_cell, vertex_coord,
                cell_vertex_velocity_front, cell_vertex_velocity_back,
                cell_vertex_ztop_front, cell_vertex_ztop_back,
                cell_vertex_vert_velocity_front, cell_vertex_vert_velocity_back,
                attr0_front, attr1_front, attr0_back, attr1_back, max_level_cell,
                masked_attr_value);

            // If failed, try neighbor cells
            if (!s4.ok) {
                for (int i = 0; i < current_cell_vertices_number + 1; ++i) {
                    int neighbor_id = cell_neig_vec[i];
                    if (neighbor_id < 0 || neighbor_id >= actual_cell_size || neighbor_id == cell_id) {
                        continue;
                    }

                    PathlineVelocityState s4_neighbor = CalcVelocityAtPathlineWithConstraintsCUDA(
                        p4, neighbor_id, current_depth, a4,
                        actual_max_edge_size, actual_vertex_size,
                        actual_ztop_layer, actual_ztop_layer_p1,
                        has_double_attributes, attr_count,
                        number_vertex_on_cell, vertices_on_cell, vertex_coord,
                        cell_vertex_velocity_front, cell_vertex_velocity_back,
                        cell_vertex_ztop_front, cell_vertex_ztop_back,
                        cell_vertex_vert_velocity_front, cell_vertex_vert_velocity_back,
                        attr0_front, attr1_front, attr0_back, attr1_back, max_level_cell,
                masked_attr_value);

                    if (s4_neighbor.ok) {
                        s4 = s4_neighbor;
                        if (global_id == 0) {
                            DEBUG_PATHLINE_STEP(i_step, "  [RK4 k4 NEIGHBOR] i_step=%d: found in neighbor cell %d", i_step, neighbor_id);
                        }
                        break;
                    }
                }
            }

            // PERTURBATION: Add random velocity if speed is too low (k4 stage)
            double s4_h_mag = MOPS_LENGTH(s4.h_vel);
            if (s4_h_mag < VEL_THRESH) {
                vec3 perturb_h = GenerateRandomTangentVelocityCUDA(
                    p4, global_id, i_step * 4 + 4, sigma_h);
                s4.h_vel = s4.h_vel + perturb_h;
            }
            if (fabs(s4.v_vel) < VEL_THRESH) {
                float r1 = random_float(global_id, i_step * 4 + 4, 10);
                float r2 = random_float(global_id, i_step * 4 + 4, 20);
                s4.v_vel += sigma_v * (r1 - 0.5 + r2 - 0.5);
            }

            // BUG FIX: Check if any RK4 stage failed
            // If s2 or s4 failed (ok=false, returns attr=0), fallback to valid stages
            bool all_stages_ok = s1.ok && s2.ok && s3.ok && s4.ok;

            if (all_stages_ok) {
                // Standard RK4 when all stages succeeded
                current_horizontal_velocity = (s1.h_vel + 2.0 * s2.h_vel + 2.0 * s3.h_vel + s4.h_vel) / 6.0;
                current_attrs = (s1.attr + 2.0 * s2.attr + 2.0 * s3.attr + s4.attr) / 6.0;
                current_vertical_velocity = (s1.v_vel + 2.0 * s2.v_vel + 2.0 * s3.v_vel + s4.v_vel) / 6.0;
            } else {
                // Fallback: use only successful stages or Euler method
                // If s1 succeeded, use it (Euler-like fallback)
                if (s1.ok) 
                {
                    current_horizontal_velocity = s1.h_vel;
                    current_attrs = s1.attr;
                    current_vertical_velocity = s1.v_vel;

                   
                } 
                else 
                {
                    // All stages failed - this shouldn't happen, but handle it
                    current_horizontal_velocity = vec3{0.0, 0.0, 0.0};
                    current_attrs = vec3{0.0, 0.0, 0.0};
                    current_vertical_velocity = 0.0;
                }
            }

           
            // BUG FIX: RK4 should use spherical rotation, same as Euler
            // Old method was: x_trial = pos + vel*dt, then normalize (Cartesian + project)
            // New method: spherical rotation along great circle (geometrically consistent)
            vec3 rotation_axis = MOPS::CUDAKernel::CalcRotationAxis(current_position, current_horizontal_velocity);
            double speed = MOPS_LENGTH(current_horizontal_velocity);
            double theta_rad = (speed * dt) / r;
            rk4_next_position = MOPS::CUDAKernel::CalcPositionAfterRotation(current_position, rotation_axis, theta_rad);
        }

        if (use_euler) {
            vec3 rotation_axis = MOPS::CUDAKernel::CalcRotationAxis(current_position, current_horizontal_velocity);
            double speed = MOPS_LENGTH(current_horizontal_velocity);
            double theta_rad = (speed * delta_t) / r;
            new_position = MOPS::CUDAKernel::CalcPositionAfterRotation(current_position, rotation_axis, theta_rad);
        } else {
            new_position = rk4_next_position;
        }

        if (first_vel) {
            first_vel = false;
            write_vels[base_idx] = current_horizontal_velocity;
        }
        if (first_attr && has_double_attributes) {
            first_attr = false;
            write_attrs[base_idx] = current_attrs;

            // // DEBUG: Print first_attr write
            // if (global_id == 0) {
            //     printf("  [FIRST_ATTR] i_step=%d, base_idx=%d, T=%.4f, S=%.4f\n",
            //            i_step, base_idx, current_attrs.x(), current_attrs.y());
            // }
        }

        // ========================================================================
        // VERTICAL BOUNDARY CONDITIONS
        // (Section 7 of pathline_cuda_summary.md - EXACT IMPLEMENTATION)
        // ========================================================================

        // Store the tentative horizontal position (already computed by Euler/RK4)
        vec3 new_position_raw = new_position;
        double old_depth = static_cast<double>(particle_depths[global_id]);

        // STEP 1: Compute bottom clearance at time t in CURRENT cell
        // (Section 7.3, lines 617-628)
        double bottom_ztop_t = CalcZTopAtBottomCUDA(
            current_position,  // ← At time t, in CURRENT cell
            cell_id,           // ← CURRENT cell
            actual_max_edge_size,
            actual_vertex_size,
            actual_ztop_layer,
            number_vertex_on_cell,
            vertices_on_cell,
            vertex_coord,
            cell_vertex_ztop_front,
            max_level_cell);

        double bottom_depth_t = -bottom_ztop_t;
        double d = bottom_depth_t - old_depth;  // Clearance from bottom
        d = fmax(d, 0.0);  // Ensure non-negative

        // STEP 2: Compute tentative depth from vertical velocity
        // (Section 7.3, line 638)
        double new_depth_raw = old_depth - current_vertical_velocity * static_cast<double>(delta_t);

        // STEP 3: SURFACE BOUNDARY PROJECTION
        // (Section 7.2, lines 487-505)

        // 3a: Evaluate surface level at tentative horizontal position
        double surface_ztop = CalcZTopAtLevel0CUDA(
            new_position_raw,  // ← Tentative NEW horizontal position
            cell_id,
            actual_max_edge_size,
            actual_vertex_size,
            actual_ztop_layer,
            number_vertex_on_cell,
            vertices_on_cell,
            vertex_coord,
            cell_vertex_ztop_front);

        double surface_depth = -surface_ztop;

        // 3b: Apply surface projection
        double new_depth = new_depth_raw;
        if (new_depth_raw < surface_depth) {
            new_depth = surface_depth;
        }
        new_depth = fmax(new_depth, 0.0);

        // STEP 4: BOTTOM BOUNDARY PROJECTION (BOTTOM-FOLLOWING)
        // (Section 7.3, lines 641-667)

        // 4a: Find which cell contains the tentative horizontal position
        int tentative_cell = FindContainingCellInCurrentOrNeighborsCUDA(
            new_position_raw,
            cell_id,
            actual_max_edge_size,
            number_vertex_on_cell,
            cells_on_cell,
            vertices_on_cell,
            vertex_coord);

        int target_cell_for_bottom = (tentative_cell >= 0) ? tentative_cell : cell_id;

        // 4b: Evaluate bottom depth at NEW horizontal position in TARGET cell
        double bottom_ztop_next = CalcZTopAtBottomCUDA(
            new_position_raw,      // ← Tentative NEW horizontal position
            target_cell_for_bottom, // ← TARGET cell
            actual_max_edge_size,
            actual_vertex_size,
            actual_ztop_layer,
            number_vertex_on_cell,
            vertices_on_cell,
            vertex_coord,
            cell_vertex_ztop_front,
            max_level_cell);

        double bottom_depth_next = -bottom_ztop_next;

        // DEBUG: Track bottom depth changes
        /*
        if (global_id == 0 && fabs(bottom_depth_next - bottom_depth_t) > 100.0) {
            DEBUG_PATHLINE_STEP(i_step, "  ⚠️ [BOTTOM DEPTH CHANGE] i_step=%d:", i_step);
            DEBUG_PATHLINE("      bottom_depth_t=%.2f (current cell=%d)", bottom_depth_t, cell_id);
            DEBUG_PATHLINE("      bottom_depth_next=%.2f (target cell=%d)", bottom_depth_next, target_cell_for_bottom);
            DEBUG_PATHLINE("      change=%.2f m", bottom_depth_next - bottom_depth_t);

            if (max_level_cell != nullptr) {
                int ml_current = max_level_cell[cell_id];
                int ml_target = max_level_cell[target_cell_for_bottom];
            DEBUG_PATHLINE("      maxLevelCell[current]=%d, maxLevelCell[target]=%d", ml_current, ml_target);

                // Verify: Check if target is a neighbor of current
                bool is_neighbor = false;
                int num_neighbors = static_cast<int>(number_vertex_on_cell[cell_id]) + 1;
                for (int i = 0; i < num_neighbors && i < 11; ++i) {
                    int neighbor_id = cell_neig_vec[i];
                    if (neighbor_id == target_cell_for_bottom) {
                        is_neighbor = true;
                        break;
                    }
                }
            DEBUG_PATHLINE("      Is target a neighbor? %s", is_neighbor ? "YES" : "NO");

                // Show actual layer depths for verification
            DEBUG_PATHLINE("      Verifying layer depths at cell vertices:");
                printf("        Current cell %d (maxLevel=%d, layer_idx=%d):\n",
                       cell_id, ml_current, ml_current-1);

                int current_verts = static_cast<int>(number_vertex_on_cell[cell_id]);
                for (int iv = 0; iv < current_verts && iv < 3; ++iv) {
                    size_t vid = vertices_on_cell[cell_id * actual_max_edge_size + iv];
                    if (vid < actual_vertex_size && ml_current > 0 && ml_current <= actual_ztop_layer) {
                        int layer_idx = ml_current - 1;
                        size_t ztop_array_idx = vid * actual_ztop_layer + layer_idx;
                        double ztop_val = cell_vertex_ztop_front[ztop_array_idx];
                        double depth_val = -ztop_val;
                        printf("          vertex[%d] id=%llu: ztop=%.2f, depth=%.2f\n",
                               iv, (unsigned long long)vid, ztop_val, depth_val);
                    }
                }

                printf("        Target cell %d (maxLevel=%d, layer_idx=%d):\n",
                       target_cell_for_bottom, ml_target, ml_target-1);

                int target_verts = static_cast<int>(number_vertex_on_cell[target_cell_for_bottom]);
                for (int iv = 0; iv < target_verts && iv < 3; ++iv) {
                    size_t vid = vertices_on_cell[target_cell_for_bottom * actual_max_edge_size + iv];
                    if (vid < actual_vertex_size && ml_target > 0 && ml_target <= actual_ztop_layer) {
                        int layer_idx = ml_target - 1;
                        size_t ztop_array_idx = vid * actual_ztop_layer + layer_idx;
                        double ztop_val = cell_vertex_ztop_front[ztop_array_idx];
                        double depth_val = -ztop_val;
                        printf("          vertex[%d] id=%llu: ztop=%.2f, depth=%.2f\n",
                               iv, (unsigned long long)vid, ztop_val, depth_val);
                    }
                }
            }
        }
        */

        // 4c: Bottom constraint - prevent particle from going below seafloor
        // Strategy: Pull particle to the MIDDLE of the last valid layer
        if (new_depth > bottom_depth_next) {
            // Get maxLevelCell to find last valid layer
            int last_valid_layer = actual_ztop_layer - 1;  // Default: last layer
            if (max_level_cell != nullptr) {
                int cell_max_level = max_level_cell[target_cell_for_bottom];
                if (cell_max_level >= 1 && cell_max_level <= actual_ztop_layer) {
                    last_valid_layer = cell_max_level - 1;  // 0-based index
                }
            }

            // Calculate layer top and bottom
            double layer_top_ztop = CalcZTopAtLayerCUDA(
                new_position_raw,
                target_cell_for_bottom,
                last_valid_layer,  // Layer top
                actual_max_edge_size,
                actual_vertex_size,
                actual_ztop_layer,
                number_vertex_on_cell,
                vertices_on_cell,
                vertex_coord,
                cell_vertex_ztop_front);

            double layer_bottom_ztop = (last_valid_layer + 1 < actual_ztop_layer) ?
                CalcZTopAtLayerCUDA(
                    new_position_raw,
                    target_cell_for_bottom,
                    last_valid_layer + 1,  // Layer bottom
                    actual_max_edge_size,
                    actual_vertex_size,
                    actual_ztop_layer,
                    number_vertex_on_cell,
                    vertices_on_cell,
                    vertex_coord,
                    cell_vertex_ztop_front) : layer_top_ztop;

            double layer_top_depth = -layer_top_ztop;
            double layer_bottom_depth = -layer_bottom_ztop;
            double layer_mid_depth = (layer_top_depth + layer_bottom_depth) / 2.0;

            // if (global_id == 0) {
            //     double theoretical_new_depth = new_depth;
            //     DEBUG_PATHLINE_STEP(i_step, "  [BOTTOM LIMIT] i_step=%d:", i_step);
            //     DEBUG_PATHLINE("    Depth in old cell: %.2fm", old_depth);
            //     DEBUG_PATHLINE("    Theoretical new depth (RK4): %.2fm", theoretical_new_depth);
            //     DEBUG_PATHLINE("    Last valid layer: %d", last_valid_layer);
            //     DEBUG_PATHLINE("    Layer range: %.2f - %.2fm", layer_top_depth, layer_bottom_depth);
            //     DEBUG_PATHLINE("    Layer middle: %.2fm", layer_mid_depth);
            // }

            // Pull to layer middle (don't descend below old depth)
            new_depth = fmin(old_depth, layer_mid_depth);

            // Check if velocity is too small after bottom limiting
            // If so, add extra random perturbation to help particle escape
            double current_h_vel_mag = MOPS_LENGTH(current_horizontal_velocity);
            constexpr double LOW_VEL_THRESH = 0.005;  // 5 mm/s
            constexpr double BOOST_sigma_h = 0.02;     // 2 cm/s extra horizontal boost

            if (current_h_vel_mag < LOW_VEL_THRESH) {
                // Add extra random perturbation
                vec3 extra_perturb = GenerateRandomTangentVelocityCUDA(
                    current_position, global_id, i_step * 10 + 7, BOOST_sigma_h);
                current_horizontal_velocity = current_horizontal_velocity + extra_perturb;

                
            }

           
        }

        // STEP 5: Final clamping to keep particle in water column
        // (Section 7.3, lines 670-671)
        new_depth = fmax(new_depth, surface_depth);
        // Note: bottom constraint already handled above in BOTTOM LIMIT section
        // Don't apply additional clamping here to avoid overriding the logic

        // STEP 6: Project position to match adjusted depth
        // (Section 7.2 line 500, Section 7.3 lines 674-678)
        // "Keep the tentative horizontal movement, but update the radial/depth position"
        new_position = ProjectPositionToDepth(
            new_position_raw,
            new_depth,
            old_depth,
            r);

        // Store corrected depth
        particle_depths[global_id] = static_cast<float>(new_depth);

        // // DEBUG: Track large depth changes
        // if (global_id == 0) {
        //     double depth_change = new_depth - old_depth;
        //     if (fabs(depth_change) > 100.0) {
        //         DEBUG_PATHLINE_STEP(i_step, "  ⚠️ [LARGE DEPTH CHANGE] i_step=%d:", i_step);
        //         printf("      old_depth=%.2f, new_depth=%.2f, change=%.2f m\n",
        //                old_depth, new_depth, depth_change);
        //         printf("      bottom_depth_t=%.2f, bottom_depth_next=%.2f\n",
        //                bottom_depth_t, bottom_depth_next);
        //         DEBUG_PATHLINE("      cell_id=%d, target_cell=%d", cell_id, target_cell_for_bottom);
        //         DEBUG_PATHLINE("      clearance d=%.2f m", d);
        //         printf("      new_depth_raw=%.2f, surface_depth=%.2f\n",
        //                new_depth_raw, surface_depth);
        //     }
        // }

        // Get the new radius from the adjusted position
        // This is needed for lateral boundary handling below
        double r_new = MOPS_LENGTH(new_position);

        // CRITICAL FIX: Check if position is in current cell OR neighbor cells
        // This distinguishes normal cell crossing from true boundary hits
        int containing_cell = FindContainingCellInCurrentOrNeighborsCUDA(
            new_position,
            cell_id,
            actual_max_edge_size,
            number_vertex_on_cell,
            cells_on_cell,
            vertices_on_cell,
            vertex_coord);

        if (containing_cell >= 0) {
            // Normal case: position is in current cell or neighbor cell
            if (containing_cell != cell_id) {
                // Cell crossing - update cell_id and vertex count
                cell_id = containing_cell;
                current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);

                // Update neighbor list for new cell
                MOPS::CUDAKernel::GetCellNeighborsIdx(
                    cell_id,
                    current_cell_vertices_number,
                    cell_neig_vec,
                    MAX_VERTEX_NUM,
                    actual_max_edge_size,
                    cells_on_cell);
            }
            // Position is valid, no boundary handling needed
        } else {
            // Truly outside mesh (hit land/boundary) - apply boundary projection
            if (boundary_hit_count != nullptr) {
                atomicAdd(&boundary_hit_count[global_id], 1);
            }

            #ifdef __CUDA_ARCH__
            // Print detailed info for debugging (limit output to avoid spam)
            if (i_step < 5 || (i_step % 100 == 0)) {
                // Convert XYZ to lat/lon for display
                double r = sqrt(current_position.x() * current_position.x() +
                               current_position.y() * current_position.y() +
                               current_position.z() * current_position.z());
                double lat = asin(current_position.z() / r) * 180.0 / 3.14159265359;
                double lon = atan2(current_position.y(), current_position.x()) * 180.0 / 3.14159265359;
                double depth = static_cast<double>(particle_depths[global_id]);  // Fixed: removed negative sign

                // printf("🌊 Particle %d: Boundary hit at step %d, cell %d\n",
                //        global_id, i_step, cell_id);
                // printf("  📍 Position: lat=%.4f°, lon=%.4f°, depth=%.1fm\n",
                //        lat, lon, depth);
            }
            #endif

            // Find nearest edge of current cell
            int edge_va_idx = -1;
            int edge_vb_idx = -1;
            FindNearestCellEdgeCUDA(
                current_position,
                cell_id,
                actual_max_edge_size,
                number_vertex_on_cell,
                vertices_on_cell,
                vertex_coord,
                edge_va_idx,
                edge_vb_idx);

            if (edge_va_idx >= 0 && edge_vb_idx >= 0) {
                vec3 va = vertex_coord[edge_va_idx];
                vec3 vb = vertex_coord[edge_vb_idx];

                // Project velocity onto boundary edge
                vec3 projected_vel = ProjectVelocityOntoEdgeCUDA(
                    current_horizontal_velocity,
                    current_position,
                    va,
                    vb);

                double original_speed = MOPS_LENGTH(current_horizontal_velocity);
                double projected_speed = MOPS_LENGTH(projected_vel);

                // CRITICAL: Handle invalid or near-perpendicular velocities
                vec3 edge_dir = vb - va;
                double edge_len = MOPS_LENGTH(edge_dir);

                // Check for NaN or invalid velocity
                if (!isfinite(original_speed) || !isfinite(projected_speed) ||
                    original_speed < 1e-12 || edge_len < 1e-12) {
                    // Invalid velocity or edge - use default boundary velocity
                    // Set a default speed along the boundary edge
                    if (edge_len > 1e-12) {
                        vec3 edge_norm = edge_dir / edge_len;
                        double default_speed = 1e-3;  // Default 1 mm/s
                        projected_vel = edge_norm * default_speed;
                        projected_speed = default_speed;
                        original_speed = default_speed;  // Update for debug output
                    } else {
                        // Cannot determine valid direction - mark particle as stuck
                        projected_vel = vec3{0, 0, 0};
                        projected_speed = 0.0;
                        original_speed = 0.0;
                    }
                } else {
                    // Valid velocity - check if perpendicular to boundary
                    double speed_ratio = projected_speed / original_speed;

                    if (speed_ratio < 0.1) {
                        // Nearly perpendicular - redirect along boundary edge
                        // Normalize edge direction
                        vec3 edge_norm = edge_dir / edge_len;

                        // Choose direction: prefer direction closer to original velocity
                        double dot_vel_edge = current_horizontal_velocity.x()*edge_norm.x() +
                                             current_horizontal_velocity.y()*edge_norm.y() +
                                             current_horizontal_velocity.z()*edge_norm.z();

                        if (dot_vel_edge < 0) {
                            edge_norm = edge_norm * (-1.0);  // Flip if opposite
                        }

                        // Set velocity along boundary with original speed
                        projected_vel = edge_norm * original_speed;
                        projected_speed = original_speed;
                    } else if (speed_ratio < 1.0) {
                        // Not perpendicular, but reduced - restore original magnitude
                        projected_vel = projected_vel * (original_speed / projected_speed);
                        projected_speed = original_speed;
                    }
                }

                #ifdef __CUDA_ARCH__
                if (i_step < 5 || (i_step % 100 == 0)) {
                    // printf("  → Edge (%d,%d): speed %.6e → %.6e (%.1f%% retained)\n",
                    //        edge_va_idx, edge_vb_idx,
                    //        original_speed, projected_speed,
                    //        original_speed > 1e-12 ? 100.0 * projected_speed / original_speed : 0.0);

                    // Calculate velocity direction relative to boundary
                    vec3 edge_dir = vb - va;
                    vec3 vel_normalized = current_horizontal_velocity / (original_speed + 1e-20);
                    double edge_len = sqrt(edge_dir.x()*edge_dir.x() + edge_dir.y()*edge_dir.y() + edge_dir.z()*edge_dir.z());
                    vec3 edge_norm = edge_dir / (edge_len + 1e-20);
                    double dot_prod = vel_normalized.x()*edge_norm.x() + vel_normalized.y()*edge_norm.y() + vel_normalized.z()*edge_norm.z();
                    double angle_deg = acos(fmin(fmax(dot_prod, -1.0), 1.0)) * 180.0 / 3.14159265359;
                    // printf("     Velocity angle to edge: %.1f° (90° = perpendicular)\n", angle_deg);
                }
                #endif

                // Recalculate position moving along the boundary edge
                vec3 rotation_axis = MOPS::CUDAKernel::CalcRotationAxis(
                    current_position,
                    projected_vel);
                double theta_rad = (projected_speed * delta_t) / r;

                vec3 candidate_position = MOPS::CUDAKernel::CalcPositionAfterRotation(
                    current_position,
                    rotation_axis,
                    theta_rad);

                // Rescale with same radius
                double candidate_len = MOPS_LENGTH(candidate_position);
                if (candidate_len > 1e-12) {
                    candidate_position = (candidate_position / candidate_len) * r_new;
                }

                // CRITICAL: Verify the projected position is still in valid mesh
                bool projected_is_valid = MOPS::CUDAKernel::IsInMesh(
                    cell_id,
                    actual_max_edge_size,
                    candidate_position,
                    number_vertex_on_cell,
                    vertices_on_cell,
                    vertex_coord);

                if (projected_is_valid) {
                    // Projection successful - use new position
                    new_position = candidate_position;
                    #ifdef __CUDA_ARCH__
                    if (i_step < 5 || (i_step % 100 == 0)) {
                        // printf("  ✓ Projected position validated (in mesh)\n");
                    }
                    #endif
                } else {
                    // Projection failed - try multiple random directions to prevent stagnation
                    #ifdef __CUDA_ARCH__
                    if (i_step < 5 || (i_step % 100 == 0)) {
                        // printf("  ⚠️  Projected position INVALID - trying multiple random directions\n");
                    }
                    #endif

                    vec3 escaped_position;
                    double used_kick_speed = 0.0;
                    bool escape_successful = TryMultipleRandomDirectionsCUDA(
                        current_position,
                        cell_id,
                        global_id,
                        i_step,
                        r,
                        r_new,
                        delta_t,
                        actual_max_edge_size,
                        number_vertex_on_cell,
                        vertices_on_cell,
                        vertex_coord,
                        escaped_position,
                        used_kick_speed);

                    if (escape_successful) {
                        new_position = escaped_position;
                        #ifdef __CUDA_ARCH__
                        if (i_step < 5 || (i_step % 100 == 0)) {
                            // printf("  🎲 Random escape successful (speed=%.6f m/s)\n", used_kick_speed);
                        }
                        #endif
                    } else {
                        // All random directions failed - particle truly trapped
                        // Keep current position (safe, doesn't produce NaN)
                        new_position = current_position;
                        #ifdef __CUDA_ARCH__
                        if (i_step < 5 || (i_step % 100 == 0)) {
                            // printf("  ⛔ Particle trapped after 20 attempts\n");
                        }
                        #endif
                    }
                }
            } else {
                // Edge detection failed - try multiple random directions
                #ifdef __CUDA_ARCH__
                if (i_step < 5 || (i_step % 100 == 0)) {
                    // printf("  ⚠️  Edge detection FAILED - trying multiple random directions\n");
                }
                #endif

                vec3 escaped_position;
                double used_kick_speed = 0.0;
                bool escape_successful = TryMultipleRandomDirectionsCUDA(
                    current_position,
                    cell_id,
                    global_id,
                    i_step,
                    r,
                    r_new,
                    delta_t,
                    actual_max_edge_size,
                    number_vertex_on_cell,
                    vertices_on_cell,
                    vertex_coord,
                    escaped_position,
                    used_kick_speed);

                if (escape_successful) {
                    new_position = escaped_position;
                    #ifdef __CUDA_ARCH__
                    if (i_step < 5 || (i_step % 100 == 0)) {
                        // printf("  🎲 Random escape successful (speed=%.6f m/s)\n", used_kick_speed);
                    }
                    #endif
                } else {
                    // All random directions failed - particle truly trapped
                    // Keep current position (safe, doesn't produce NaN)
                    new_position = current_position;
                    #ifdef __CUDA_ARCH__
                    if (i_step < 5 || (i_step % 100 == 0)) {
                        // printf("  ⛔ Particle trapped after 20 attempts\n");
                    }
                    #endif
                }
            }
        }

        sample_points[global_id] = new_position;

        int record_stride = record_t / delta_t;
        if (record_stride != 0 && ((i_step + 1) % record_stride) == 0) {
            int write_idx = base_idx + update_points_idx;
            if (write_idx >= 0 && write_idx < ((global_id + 1) * each_points_size)) {
                write_points[write_idx] = new_position;
                write_vels[write_idx] = current_horizontal_velocity;
                if (has_double_attributes) {
                    write_attrs[write_idx] = current_attrs;
                }

                // DEBUG: Print record writes for first particle, first few records
                #ifdef __CUDA_ARCH__
                if (global_id == 0 && update_points_idx < 3) {
                    // printf("  [RECORD] i_step=%d, write_idx=%d, update_points_idx=%d, T=%.4f, S=%.4f\n",
                    //        i_step, write_idx, update_points_idx, current_attrs.x(), current_attrs.y());
                }
                #endif

                update_points_idx += 1;  // Only increment if write succeeded
            }
        }
    }
}

}  // namespace Kernel
}  // namespace CUDABackend
}  // namespace GPU
}  // namespace MOPS
