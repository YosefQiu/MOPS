#include "GPU/HIP/MPASOVisualizerHIP.h"

#include "GPU/HIP/Kernel/MPASOVisualizerKernels.h"

namespace MOPS::GPU::HIPBackend {

void VisualizeFixedLayer(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img)
{
    Kernel::VisualizeFixedLayer(mpasoF, config, img);
}

void VisualizeFixedDepth(MPASOField* mpasoF, VisualizationSettings* config, std::vector<ImageBuffer<double>>& img_vec)
{
    Kernel::VisualizeFixedDepth(mpasoF, config, img_vec);
}

void VisualizeFixedLatitude(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img)
{
    Kernel::VisualizeFixedLatitude(mpasoF, config, img);
}

std::vector<TrajectoryLine> StreamLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id)
{
    return Kernel::StreamLine(mpasoF, points, config, default_cell_id);
}

std::vector<TrajectoryLine> PathLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id)
{
    return Kernel::PathLine(mpasoF, points, config, default_cell_id);
}

} // namespace MOPS::GPU::HIPBackend
