#pragma once

#include "Core/GPUContext.h"
#include "Core/MPASOVisualizer.h"
#include "Core/MPASOGrid.h"

namespace MOPS::GPU::Factory {

void VisualizeFixedLayer(
    MPASOField* mpasoF,
    VisualizationSettings* config,
    ImageBuffer<double>* img,
    const GPUContext& ctx);

void VisualizeFixedDepth(
    MPASOField* mpasoF,
    VisualizationSettings* config,
    std::vector<ImageBuffer<double>>& img_vec,
    const GPUContext& ctx);

void VisualizeFixedLatitude(
    MPASOField* mpasoF,
    VisualizationSettings* config,
    ImageBuffer<double>* img,
    const GPUContext& ctx);

std::vector<TrajectoryLine> StreamLine(
    MPASOField* mpasoF,
    std::vector<CartesianCoord>& points,
    TrajectorySettings* config,
    std::vector<int>& default_cell_id,
    const GPUContext& ctx);

std::vector<TrajectoryLine> PathLine(
    MPASOField* mpasoF,
    std::vector<CartesianCoord>& points,
    TrajectorySettings* config,
    std::vector<int>& default_cell_id,
    const GPUContext& ctx);

void CalcCellVertexZtop(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_center_ztop,
    std::vector<double>& cell_vertex_ztop,
    const std::vector<size_t>& grid_info,
    const GPUContext& ctx);

void CalcCellCenterToVertex(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_center_attr,
    std::vector<double>& cell_vertex_attr,
    const std::vector<size_t>& grid_info,
    const GPUContext& ctx);

void CalcCellCenterVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_normal_velocity,
    std::vector<vec3>& cell_center_velocity,
    const std::vector<size_t>& grid_info,
    const GPUContext& ctx);

void CalcCellCenterVelocityByZM(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_zonal_velocity,
    const std::vector<double>& cell_meridional_velocity,
    std::vector<vec3>& cell_center_velocity,
    const std::vector<size_t>& grid_info,
    const GPUContext& ctx);

void CalcCellVertexVelocityByZM(
    MPASOGrid* grid,
    int vertex_size,
    int total_ztop_layer,
    const std::vector<double>& cell_vertex_zonal_velocity,
    const std::vector<double>& cell_vertex_meridional_velocity,
    std::vector<vec3>& cell_vertex_velocity,
    const std::vector<size_t>& grid_info,
    const GPUContext& ctx);

void CalcCellVertexVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<vec3>& cell_center_velocity,
    std::vector<vec3>& cell_vertex_velocity,
    const std::vector<size_t>& grid_info,
    const GPUContext& ctx);

void CalcCellVertexVertVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer_p1,
    const std::vector<double>& cell_center_vert_velocity,
    std::vector<double>& cell_vertex_vert_velocity,
    const std::vector<size_t>& grid_info,
    const GPUContext& ctx);

} // namespace MOPS::GPU::Factory
