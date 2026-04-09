#include "GPU/HIP/Kernel/HIPKernel.h"

namespace MOPS {

void HIPKernel::SearchKDTree(
    int* cell_id_vec,
    MPASOGrid* grid,
    int width,
    int height,
    double minLat,
    double maxLat,
    double minLon,
    double maxLon)
{
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            vec2 pixel = make_double2(static_cast<double>(i), static_cast<double>(j));
            vec2 latlon_r;
            GeoConverter::convertPixelToLatLonToRadians(width, height, minLat, maxLat, minLon, maxLon, pixel, latlon_r);
            vec3 current_position;
            GeoConverter::convertRadianLatLonToXYZ(latlon_r, current_position);
            int cell_id_value = -1;
            grid->searchKDT(current_position, cell_id_value);
            cell_id_vec[i * width + j] = cell_id_value;
        }
    }
}

} // namespace MOPS
