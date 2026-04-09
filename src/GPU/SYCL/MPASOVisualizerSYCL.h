#pragma once

#include "Core/MPASOVisualizer.h"

namespace MOPS::GPU::SYCLBackend {

void VisualizeFixedLayer(
    MPASOField* mpasoF,
    VisualizationSettings* config,
    ImageBuffer<double>* img,
    sycl::queue& sycl_Q);

void VisualizeFixedDepth(
    MPASOField* mpasoF,
    VisualizationSettings* config,
    std::vector<ImageBuffer<double>>& img_vec,
    sycl::queue& sycl_Q);

void VisualizeFixedLatitude(
    MPASOField* mpasoF,
    VisualizationSettings* config,
    ImageBuffer<double>* img,
    sycl::queue& sycl_Q);

std::vector<TrajectoryLine> StreamLine(
    MPASOField* mpasoF,
    std::vector<CartesianCoord>& points,
    TrajectorySettings* config,
    std::vector<int>& default_cell_id,
    sycl::queue& sycl_Q);

std::vector<TrajectoryLine> PathLine(
    MPASOField* mpasoF,
    std::vector<CartesianCoord>& points,
    TrajectorySettings* config,
    std::vector<int>& default_cell_id,
    sycl::queue& sycl_Q);

} // namespace MOPS::GPU::SYCLBackend
