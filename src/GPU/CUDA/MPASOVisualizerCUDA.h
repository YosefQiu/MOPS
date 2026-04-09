#pragma once

#include "Core/MPASOVisualizer.h"

namespace MOPS::GPU::CUDABackend {

void VisualizeFixedLayer(
    MPASOField* mpasoF,
    VisualizationSettings* config,
    ImageBuffer<double>* img);

void VisualizeFixedDepth(
    MPASOField* mpasoF,
    VisualizationSettings* config,
    std::vector<ImageBuffer<double>>& img_vec);

void VisualizeFixedLatitude(
    MPASOField* mpasoF,
    VisualizationSettings* config,
    ImageBuffer<double>* img);

std::vector<TrajectoryLine> StreamLine(
    MPASOField* mpasoF,
    std::vector<CartesianCoord>& points,
    TrajectorySettings* config,
    std::vector<int>& default_cell_id);

std::vector<TrajectoryLine> PathLine(
    MPASOField* mpasoF,
    std::vector<CartesianCoord>& points,
    TrajectorySettings* config,
    std::vector<int>& default_cell_id);

} // namespace MOPS::GPU::CUDABackend
