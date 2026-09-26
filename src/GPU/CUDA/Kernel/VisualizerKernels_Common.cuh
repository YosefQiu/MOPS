#pragma once

/**
 * @file VisualizerKernels_Common.cuh
 * @brief Common device functions shared by MPASO visualizer kernels
 *
 * This file contains device functions used by StreamLine, PathLine, and
 * other visualizer kernels. These functions are extracted from the original
 * MPASOVisualizerKernels.cu to reduce code duplication and improve maintainability.
 *
 * Created: 2026-09-20
 * Part of: MPASO Visualizer Kernel Refactoring (Phase 1)
 */

#include "CUDAKernel.h"  // Same directory
#include "CUDAConstants.cuh"
#include "Utils/Interpolation.hpp"
#include "Utils/GeoConverter.hpp"

namespace MOPS {
namespace GPU {
namespace CUDABackend {
namespace VisualizerInternal {

// Use unified CUDA constants
using namespace MOPS::GPU::CUDABackend::Constants;

// ============================================================================
// SECTION 1: Position and Velocity Projection
// ============================================================================

/**
 * @brief Advect a position on sphere using velocity
 * @param pos Current position (Cartesian coordinates)
 * @param vel Velocity vector (tangent to sphere)
 * @param dt_local Time step
 * @return New position after advection
 */
MOPS_DEVICE inline vec3 AdvectOnSphereCUDA(const vec3& pos, const vec3& vel, double dt_local)
{
    double r = MOPS_LENGTH(pos);
    if (r < 1e-12) {
        return pos;
    }

    vec3 rotation_axis = MOPS::CUDAKernel::CalcRotationAxis(pos, vel);
    double theta_rad = (MOPS_LENGTH(vel) * dt_local) / r;

    return MOPS::CUDAKernel::CalcPositionAfterRotation(pos, rotation_axis, theta_rad);
}

/**
 * @brief Project velocity onto cell edge (on sphere's tangent plane)
 */
MOPS_DEVICE inline vec3 ProjectVelocityOntoEdgeCUDA(
    const vec3& velocity,
    const vec3& pos,
    const vec3& edge_vertex_a,
    const vec3& edge_vertex_b)
{
    // Compute edge direction on the sphere's tangent plane at pos
    vec3 edge_vec = edge_vertex_b - edge_vertex_a;

    // Project edge vector onto tangent plane at pos
    // Tangent plane is perpendicular to radial direction
    vec3 radial = pos;
    double radial_len = MOPS_LENGTH(radial);
    if (radial_len < 1e-12) {
        return velocity;
    }
    radial = radial / radial_len;

    // Remove radial component from edge vector
    double edge_radial_component = MOPS_DOT(edge_vec, radial);
    vec3 edge_tangent = edge_vec - radial * edge_radial_component;

    double edge_tangent_len = MOPS_LENGTH(edge_tangent);
    if (edge_tangent_len < 1e-12) {
        return vec3{0.0, 0.0, 0.0};
    }

    // Normalize edge tangent direction
    edge_tangent = edge_tangent / edge_tangent_len;

    // Project velocity onto edge tangent
    double vel_parallel_magnitude = MOPS_DOT(velocity, edge_tangent);
    vec3 vel_parallel = edge_tangent * vel_parallel_magnitude;

    return vel_parallel;
}

/**
 * @brief Project position to target depth while maintaining horizontal direction
 * @param position_raw Raw position vector
 * @param new_depth Target depth (positive downward)
 * @param old_depth Previous depth
 * @param current_radius Current radius from Earth center
 */
MOPS_DEVICE inline vec3 ProjectPositionToDepth(
    const vec3& position_raw,
    double new_depth,
    double old_depth,
    double current_radius)
{
    // Calculate how much the depth changed
    // depth is positive downward, radius is from Earth center
    // If depth decreases (particle rises), radius should increase
    // If depth increases (particle sinks), radius should decrease
    double depth_change = old_depth - new_depth;
    double new_radius = current_radius + depth_change;

    // Ensure minimum radius
    new_radius = fmax(1.0, new_radius);

    // Keep horizontal direction, adjust radius
    vec3 direction = position_raw;
    double direction_len = MOPS_LENGTH(direction);
    if (direction_len > 1e-12) {
        direction = direction / direction_len;
    }

    return direction * new_radius;
}

// ============================================================================
// SECTION 2: Random Number Generation and Perturbation
// ============================================================================

/**
 * @brief Hash function for deterministic pseudo-random number generation
 */
MOPS_DEVICE inline unsigned int hash(unsigned int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

/**
 * @brief Generate random float in [0, 1] using hash-based PRNG
 */
MOPS_DEVICE inline float random_float(int particle_id, int timestep, int component)
{
    unsigned int seed = hash(static_cast<unsigned int>(particle_id * 73856093 + timestep * 19349663 + component * 83492791));
    return (seed & 0xFFFFFF) / 16777216.0f;
}

/**
 * @brief Generate random tangent velocity on sphere surface
 */
MOPS_DEVICE inline vec3 GenerateRandomTangentVelocityCUDA(
    const vec3& position,
    int particle_id,
    int timestep,
    double magnitude,
    int attempt = 0)
{
    // Radial direction
    vec3 radial = position;
    double r_len = MOPS_LENGTH(radial);
    if (r_len < 1e-12) {
        return vec3{0.0, 0.0, 0.0};
    }
    radial = radial / r_len;

    // Create arbitrary vector not parallel to radial
    vec3 arbitrary = (fabs(radial.x()) < 0.9) ? vec3{1.0, 0.0, 0.0} : vec3{0.0, 1.0, 0.0};

    // First tangent vector (Gram-Schmidt orthogonalization)
    vec3 tangent1 = arbitrary - radial * MOPS_DOT(arbitrary, radial);
    double tangent1_len = MOPS_LENGTH(tangent1);
    if (tangent1_len < 1e-12) {
        return vec3{0.0, 0.0, 0.0};
    }
    tangent1 = tangent1 / tangent1_len;

    // Second tangent vector (cross product)
    vec3 tangent2 = vec3{
        radial.y() * tangent1.z() - radial.z() * tangent1.y(),
        radial.z() * tangent1.x() - radial.x() * tangent1.z(),
        radial.x() * tangent1.y() - radial.y() * tangent1.x()
    };
    double tangent2_len = MOPS_LENGTH(tangent2);
    if (tangent2_len < 1e-12) {
        return vec3{0.0, 0.0, 0.0};
    }
    tangent2 = tangent2 / tangent2_len;

    // Random angle in [0, 2π] - use attempt number to get different directions
    float rand_val = random_float(particle_id, timestep, attempt);
    double angle = static_cast<double>(rand_val) * 2.0 * 3.14159265359;

    // Random tangent direction
    vec3 random_direction = tangent1 * cos(angle) + tangent2 * sin(angle);

    return random_direction * magnitude;
}

// ============================================================================
// SECTION 3: Grid and Cell Operations
// ============================================================================

/**
 * @brief Find containing cell in current cell or its neighbors
 */
MOPS_DEVICE inline int FindContainingCellInCurrentOrNeighborsCUDA(
    const vec3& pos,
    int current_cell,
    int actual_max_edge_size,
    const size_t* number_vertex_on_cell,
    const size_t* cells_on_cell,
    const size_t* vertices_on_cell,
    const vec3* vertex_coord)
{
    // First try current cell
    if (current_cell >= 0) {
        bool in_current = MOPS::CUDAKernel::IsInMesh(
            current_cell, actual_max_edge_size, pos,
            number_vertex_on_cell, vertices_on_cell, vertex_coord);
        if (in_current) {
            return current_cell;
        }
    }

    // Try neighbors
    if (current_cell >= 0) {
        int num_vertices = static_cast<int>(number_vertex_on_cell[current_cell]);
        int num_neighbors = num_vertices + 1;  // nEdgesOnCell = nVerticesOnCell + 1

        for (int i = 0; i < num_neighbors && i < 11; ++i) {
            int neighbor_id = static_cast<int>(cells_on_cell[current_cell * actual_max_edge_size + i]) - 1;
            if (neighbor_id < 0) {
                continue;
            }

            bool in_neighbor = MOPS::CUDAKernel::IsInMesh(
                neighbor_id, actual_max_edge_size, pos,
                number_vertex_on_cell, vertices_on_cell, vertex_coord);
            if (in_neighbor) {
                return neighbor_id;
            }
        }
    }

    return -1;  // Not found
}

// ============================================================================
// SECTION 4: Depth and Layer Calculations
// ============================================================================

/**
 * @brief Calculate zTop at layer 0 (sea surface)
 */
MOPS_DEVICE inline double CalcZTopAtLevel0CUDA(
    const vec3& pos,
    int cell_id,
    int actual_max_edge_size,
    int actual_vertex_size,
    int actual_ztop_layer,
    const size_t* number_vertex_on_cell,
    const size_t* vertices_on_cell,
    const vec3* vertex_coord,
    const double* cell_vertex_ztop)
{

    if (cell_id < 0) {
        return 0.0;  // Sea surface level
    }

    int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
    if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
        return 0.0;
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

    // Get cell vertex positions
    vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
    if (!MOPS::CUDAKernel::GetCellVertexPos(
            current_cell_vertex_pos,
            current_cell_vertices_idx,
            MAX_VERTEX_NUM,
            current_cell_vertices_number,
            vertex_coord)) {
        return 0.0;
    }

    // Calculate Wachspress weights
    double current_cell_vertex_weight[MAX_VERTEX_NUM];
    for (int i = 0; i < MAX_VERTEX_NUM; ++i) {
        current_cell_vertex_weight[i] = 0.0;
    }
    Interpolator::CalcPolygonWachspress(
        pos,
        current_cell_vertex_pos,
        current_cell_vertex_weight,
        current_cell_vertices_number);

    // Interpolate zTop at layer 0 (sea surface)
    double ztop_0 = 0.0;
    for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
        int vid = static_cast<int>(current_cell_vertices_idx[v_idx]);
        if (vid < 0 || vid >= actual_vertex_size) {
            continue;
        }
        // Layer 0 is at index 0
        double ztop_at_vertex = cell_vertex_ztop[vid * actual_ztop_layer + 0];
        ztop_0 += current_cell_vertex_weight[v_idx] * ztop_at_vertex;
    }

    return ztop_0;
}

/**
 * @brief Calculate zTop at a specific layer (Wachspress interpolation)
 */
MOPS_DEVICE inline double CalcZTopAtLayerCUDA(
    const vec3& pos,
    int cell_id,
    int layer_idx,  // 0-based layer index
    int actual_max_edge_size,
    int actual_vertex_size,
    int actual_ztop_layer,
    const size_t* number_vertex_on_cell,
    const size_t* vertices_on_cell,
    const vec3* vertex_coord,
    const double* cell_vertex_ztop)
{

    if (cell_id < 0 || layer_idx < 0 || layer_idx >= actual_ztop_layer) {
        return -1e10;
    }

    int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
    if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
        return -1e10;
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

    // Get cell vertex positions
    vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
    if (!MOPS::CUDAKernel::GetCellVertexPos(
            current_cell_vertex_pos,
            current_cell_vertices_idx,
            MAX_VERTEX_NUM,
            current_cell_vertices_number,
            vertex_coord)) {
        return -1e10;
    }

    // Calculate Wachspress weights
    double current_cell_vertex_weight[MAX_VERTEX_NUM];
    for (int i = 0; i < MAX_VERTEX_NUM; ++i) {
        current_cell_vertex_weight[i] = 0.0;
    }
    Interpolator::CalcPolygonWachspress(
        pos,
        current_cell_vertex_pos,
        current_cell_vertex_weight,
        current_cell_vertices_number);

    // Interpolate zTop at specified layer
    double ztop = 0.0;
    for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
        int vid = static_cast<int>(current_cell_vertices_idx[v_idx]);
        if (vid < 0 || vid >= actual_vertex_size) {
            continue;
        }
        double ztop_at_vertex = cell_vertex_ztop[vid * actual_ztop_layer + layer_idx];
        ztop += current_cell_vertex_weight[v_idx] * ztop_at_vertex;
    }

    return ztop;
}

/**
 * @brief Calculate zTop at bottom layer (last valid ocean layer)
 */
MOPS_DEVICE inline double CalcZTopAtBottomCUDA(
    const vec3& pos,
    int cell_id,
    int actual_max_edge_size,
    int actual_vertex_size,
    int actual_ztop_layer,
    const size_t* number_vertex_on_cell,
    const size_t* vertices_on_cell,
    const vec3* vertex_coord,
    const double* cell_vertex_ztop,
    const int* max_level_cell = nullptr)
{

    if (cell_id < 0) {
        return -1e10;  // Very deep default for invalid cell
    }

    int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
    if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
        return -1e10;
    }

    if (actual_ztop_layer <= 1) {
        return -1e10;
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

    // Get cell vertex positions
    vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
    if (!MOPS::CUDAKernel::GetCellVertexPos(
            current_cell_vertex_pos,
            current_cell_vertices_idx,
            MAX_VERTEX_NUM,
            current_cell_vertices_number,
            vertex_coord)) {
        return -1e10;
    }

    // Calculate Wachspress weights
    double current_cell_vertex_weight[MAX_VERTEX_NUM];
    for (int i = 0; i < MAX_VERTEX_NUM; ++i) {
        current_cell_vertex_weight[i] = 0.0;
    }
    Interpolator::CalcPolygonWachspress(
        pos,
        current_cell_vertex_pos,
        current_cell_vertex_weight,
        current_cell_vertices_number);

    // Determine bottom layer index
    // If maxLevelCell is available, use it to get the last VALID ocean layer
    // Otherwise fall back to the last layer in the array
    int bottom_layer = actual_ztop_layer - 1;
    if (max_level_cell != nullptr) {
        int cell_max_level = max_level_cell[cell_id];
        // maxLevelCell is 1-based: [1, nVertLevels]
        if (cell_max_level >= 1 && cell_max_level <= actual_ztop_layer) {
            bottom_layer = cell_max_level - 1;  // Convert to 0-based index
        }
    }

    // Interpolate zTop at bottom layer
    double ztop_bottom = 0.0;
    for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
        int vid = static_cast<int>(current_cell_vertices_idx[v_idx]);
        if (vid < 0 || vid >= actual_vertex_size) {
            continue;
        }
        double ztop_at_vertex = cell_vertex_ztop[vid * actual_ztop_layer + bottom_layer];
        ztop_bottom += current_cell_vertex_weight[v_idx] * ztop_at_vertex;
    }

    return ztop_bottom;
}

} // namespace VisualizerInternal
} // namespace CUDABackend
} // namespace GPU
} // namespace MOPS
