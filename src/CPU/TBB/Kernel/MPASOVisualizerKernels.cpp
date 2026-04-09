#include "CPU/TBB/Kernel/MPASOVisualizerKernels.h"

#include "CPU/TBB/Kernel/TBBKernel.h"
#include "Common/TrajectoryCommon.h"
#include "Utils/GeoConverter.hpp"
#include "Utils/Interpolation.hpp"
#include <algorithm>
#include <limits>
#include <tbb/parallel_for.h>

namespace MOPS::CPU::TBBBackend::Kernel {
namespace {

inline int ClampLayer(int layer, int total_layers)
{
    if (total_layers <= 0) {
        return 0;
    }
    if (layer < 0) {
        return 0;
    }
    if (layer >= total_layers) {
        return total_layers - 1;
    }
    return layer;
}

inline int ResolveLayerFromDepth(const MPASOGrid* grid, double depth, int total_layers)
{
    if (grid == nullptr || grid->cellRefBottomDepth_vec.empty() || total_layers <= 0) {
        return 0;
    }
    double target = MOPS::math::fabs(depth);
    int best = 0;
    double best_diff = MOPS::math::fabs(grid->cellRefBottomDepth_vec[0] - target);
    const int limit = std::min(static_cast<int>(grid->cellRefBottomDepth_vec.size()), total_layers);
    for (int i = 1; i < limit; ++i) {
        const double d = MOPS::math::fabs(grid->cellRefBottomDepth_vec[i] - target);
        if (d < best_diff) {
            best = i;
            best_diff = d;
        }
    }
    return best;
}

inline double ReadTemperature(const MPASOSolution* sol, int idx)
{
    auto it = sol->mDoubleAttributes.find("temperature");
    if (it != sol->mDoubleAttributes.end() && idx >= 0 && idx < static_cast<int>(it->second.size())) {
        return it->second[idx];
    }
    return 0.0;
}

inline double ReadSalinity(const MPASOSolution* sol, int idx)
{
    auto it = sol->mDoubleAttributes.find("salinity");
    if (it != sol->mDoubleAttributes.end() && idx >= 0 && idx < static_cast<int>(it->second.size())) {
        return it->second[idx];
    }
    return 0.0;
}

inline void BuildCellIdsForMap(
    MPASOGrid* grid,
    int width,
    int height,
    double minLat,
    double maxLat,
    double minLon,
    double maxLon,
    std::vector<int>& cell_ids)
{
    cell_ids.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    tbb::parallel_for(0, width * height, [&](int id) {
        const int i = id / width;
        const int j = id % width;
        vec2 pixel(static_cast<double>(i), static_cast<double>(j));
        vec2 latlon_r;
        GeoConverter::convertPixelToLatLonToRadians(width, height, minLat, maxLat, minLon, maxLon, pixel, latlon_r);
        vec3 pos;
        GeoConverter::convertRadianLatLonToXYZ(latlon_r, pos);
        int cell_id = -1;
        grid->searchKDT(pos, cell_id);
        cell_ids[id] = cell_id;
    });
}

inline void FillRasterByCellAndLayer(
    const MPASOField* mpasoF,
    const std::vector<int>& cell_ids,
    int width,
    int height,
    int layer,
    std::vector<ImageBuffer<double>>& img_vec)
{
    const auto* sol = mpasoF->mSol_Front.get();
    const int total_layers = sol->mTotalZTopLayer;
    const int layer_clamped = ClampLayer(layer, total_layers);

    tbb::parallel_for(0, width * height, [&](int id) {
        const int i = id / width;
        const int j = id % width;
        const int cell_id = cell_ids[id];

        vec3 c0(0.0, 0.0, 0.0);
        vec3 c1(0.0, 0.0, 0.0);
        vec3 c2(0.0, 0.0, 0.0);
        vec3 c3(0.0, 0.0, 0.0);
        vec3 c4(0.0, 0.0, 0.0);
        vec3 c5(0.0, 0.0, 0.0);

        if (cell_id >= 0 && cell_id < sol->mCellsSize) {
            const int idx = cell_id * total_layers + layer_clamped;
            const double zonal = (idx < static_cast<int>(sol->cellZonalVelocity_vec.size())) ? sol->cellZonalVelocity_vec[idx] : 0.0;
            const double mer = (idx < static_cast<int>(sol->cellMeridionalVelocity_vec.size())) ? sol->cellMeridionalVelocity_vec[idx] : 0.0;
            const vec3 xyz = (idx < static_cast<int>(sol->cellCenterVelocity_vec.size())) ? sol->cellCenterVelocity_vec[idx] : vec3(0.0, 0.0, 0.0);
            const double temp = ReadTemperature(sol, idx);
            const double sal = ReadSalinity(sol, idx);

            c0 = vec3(zonal, zonal, zonal);
            c1 = vec3(mer, mer, mer);
            c2 = vec3(MOPS_LENGTH(xyz), MOPS_LENGTH(xyz), MOPS_LENGTH(xyz));
            c3 = vec3(temp, temp, temp);
            c4 = vec3(sal, sal, sal);
            c5 = vec3(0.0, 0.0, 0.0);
        }

        if (img_vec.size() > 0) img_vec[0].setPixel(i, j, c0);
        if (img_vec.size() > 1) img_vec[1].setPixel(i, j, c1);
        if (img_vec.size() > 2) img_vec[2].setPixel(i, j, c2);
        if (img_vec.size() > 3) img_vec[3].setPixel(i, j, c3);
        if (img_vec.size() > 4) img_vec[4].setPixel(i, j, c4);
        if (img_vec.size() > 5) img_vec[5].setPixel(i, j, c5);
    });
}

} // namespace

void VisualizeFixedLayer(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img)
{
    if (mpasoF == nullptr || mpasoF->mGrid == nullptr || mpasoF->mSol_Front == nullptr || config == nullptr || img == nullptr) {
        Error("[TBBBackend::Kernel]::VisualizeFixedLayer invalid inputs");
        return;
    }

    const int width = static_cast<int>(config->imageSize.x());
    const int height = static_cast<int>(config->imageSize.y());
    const double minLat = config->LatRange.x();
    const double maxLat = config->LatRange.y();
    const double minLon = config->LonRange.x();
    const double maxLon = config->LonRange.y();

    auto* grid = mpasoF->mGrid.get();
    const auto* sol = mpasoF->mSol_Front.get();

    if (width <= 0 || height <= 0 || sol->mTotalZTopLayer <= 0) {
        Error("[TBBBackend::Kernel]::VisualizeFixedLayer invalid image size or layer metadata");
        return;
    }

    const int max_edge = grid->mMaxEdgesSize;
    const int max_vertex_num = 20;
    const int total_layers = std::max(1, sol->mTotalZTopLayer);
    const int fixed_layer = ClampLayer(config->FixedLayer, total_layers);

    const auto* numberVertexOnCell = grid->numberVertexOnCell_vec.data();
    const auto* verticesOnCell = grid->verticesOnCell_vec.data();
    const auto* vertexCoord = grid->vertexCoord_vec.data();
    const auto* cellVertexVelocity = sol->cellVertexVelocity_vec.data();
    const int cell_size = grid->mCellsSize;

    std::vector<int> cell_id_vec(width * height, -1);
    TBBKernel::SearchKDTree(cell_id_vec.data(), grid, width, height, minLat, maxLat, minLon, maxLon);

    const auto double_nan = std::numeric_limits<double>::quiet_NaN();
    const vec3 vec3_nan = {double_nan, double_nan, double_nan};

    tbb::parallel_for(0, width * height, [&](int global_id) {
        const int height_index = global_id / width;
        const int width_index = global_id % width;

        vec2 pixel = {static_cast<double>(height_index), static_cast<double>(width_index)};
        SphericalCoord current_latlon_r;
        CartesianCoord current_position;
        GeoConverter::convertPixelToLatLonToRadians(width, height, minLat, maxLat, minLon, maxLon, pixel, current_latlon_r);
        GeoConverter::convertRadianLatLonToXYZ(current_latlon_r, current_position);

        const int cell_id = cell_id_vec[global_id];
        if (cell_id < 0 || cell_id >= cell_size) {
            img->setPixel(height_index, width_index, vec3_nan);
            return;
        }

        const int current_cell_vertices_number = static_cast<int>(numberVertexOnCell[cell_id]);
        if (current_cell_vertices_number <= 0 || current_cell_vertices_number > max_vertex_num) {
            img->setPixel(height_index, width_index, vec3_nan);
            return;
        }

        size_t current_cell_vertices_idx[max_vertex_num];
        TBBKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx,
                                      max_vertex_num, max_edge, verticesOnCell);

        if (!TBBKernel::IsInMesh(cell_id, max_edge, current_position, numberVertexOnCell, verticesOnCell, vertexCoord)) {
            img->setPixel(height_index, width_index, vec3_nan);
            return;
        }

        vec3 current_cell_vertex_pos[max_vertex_num];
        if (!TBBKernel::GetCellVertexPos(current_cell_vertex_pos, current_cell_vertices_idx,
                                         max_vertex_num, current_cell_vertices_number, vertexCoord)) {
            img->setPixel(height_index, width_index, vec3_nan);
            return;
        }

        double current_cell_vertex_weight[max_vertex_num];
        Interpolator::CalcPolygonWachspress(current_position, current_cell_vertex_pos,
                                            current_cell_vertex_weight, current_cell_vertices_number);

        vec3 current_point_vel = TBBKernel::CalcVelocity(current_cell_vertices_idx,
                                                         current_cell_vertex_weight,
                                                         max_vertex_num,
                                                         current_cell_vertices_number,
                                                         total_layers,
                                                         fixed_layer,
                                                         cellVertexVelocity);

        double zonal_velocity = 0.0;
        double meridional_velocity = 0.0;
        GeoConverter::convertXYZVelocityToENU(current_position, current_point_vel,
                                              zonal_velocity, meridional_velocity);
        img->setPixel(height_index, width_index, vec3(zonal_velocity, meridional_velocity, 0.0));
    });
}

void VisualizeFixedDepth(MPASOField* mpasoF, VisualizationSettings* config, std::vector<ImageBuffer<double>>& img_vec)
{
    if (mpasoF == nullptr || mpasoF->mGrid == nullptr || mpasoF->mSol_Front == nullptr || img_vec.empty()) {
        Error("[TBBBackend::Kernel]::VisualizeFixedDepth invalid inputs");
        return;
    }

    int width = static_cast<int>(config->imageSize.x());
    int height = static_cast<int>(config->imageSize.y());
    auto minLat = config->LatRange.x();
    auto maxLat = config->LatRange.y();
    auto minLon = config->LonRange.x();
    auto maxLon = config->LonRange.y();
    auto fixed_depth = -config->FixedDepth;

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(mpasoF->mGrid.get());

    std::vector<int> cell_id_vec;
    cell_id_vec.resize(width * height);
    TBBKernel::SearchKDTree(cell_id_vec.data(), mpasoF->mGrid.get(), width, height, minLat, maxLat, minLon, maxLon);

    bool bDoubleAttributes = false;
    std::vector<const std::vector<double>*> attr_vec_ptrs;
    if (mpasoF->mSol_Front->mDoubleAttributes.size() > 1) {
        bDoubleAttributes = true;
        for (const auto& [name, vec] : mpasoF->mSol_Front->mDoubleAttributes_CtoV) {
            (void)name;
            attr_vec_ptrs.push_back(&vec);
        }
    }

    const int grid_cell_size = mpasoF->mGrid->mCellsSize;
    const int CELL_SIZE = static_cast<int>(grid_info_vec[0]);
    const int max_edge = static_cast<int>(grid_info_vec[2]);
    const int MAX_VERTEX_NUM = 20;
    const int NEIGHBOR_NUM = 3;
    const int ACTUALL_VERTEX_SIZE = static_cast<int>(grid_info_vec[3]);
    const int MAX_VERTLEVELS = 100;
    const double DEPTH = fixed_depth;
    const auto nan = std::numeric_limits<size_t>::max();
    const auto double_nan = std::numeric_limits<double>::quiet_NaN();
    const vec3 vec3_nan = {double_nan, double_nan, double_nan};

    const auto& vertexCoord = mpasoF->mGrid->vertexCoord_vec;
    const auto& cellCoord = mpasoF->mGrid->cellCoord_vec;
    const auto& numberVertexOnCell = mpasoF->mGrid->numberVertexOnCell_vec;
    const auto& verticesOnCell = mpasoF->mGrid->verticesOnCell_vec;
    const auto& cellVertexVelocity = mpasoF->mSol_Front->cellVertexVelocity_vec;
    const auto& cellVertexZTop = mpasoF->mSol_Front->cellVertexZTop_vec;

    tbb::parallel_for(0, width * height, [&](int global_id) {
        int height_index = global_id / width;
        int width_index = global_id % width;

        vec2 current_pixel = {static_cast<double>(height_index), static_cast<double>(width_index)};
        CartesianCoord current_position;
        SphericalCoord current_latlon_r;
        GeoConverter::convertPixelToLatLonToRadians(width, height, minLat, maxLat, minLon, maxLon, current_pixel, current_latlon_r);
        GeoConverter::convertRadianLatLonToXYZ(current_latlon_r, current_position);

        int cell_id = cell_id_vec[global_id];
        if (cell_id < 0 || cell_id >= grid_cell_size) {
            SetPixel(img_vec[0].mPixels, width, height, height_index, width_index, vec3_nan);
            if (bDoubleAttributes && img_vec.size() > 1) {
                SetPixel(img_vec[1].mPixels, width, height, height_index, width_index, vec3_nan);
            }
            return;
        }

        auto current_cell_vertices_number = numberVertexOnCell[cell_id];
        size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
        TBBKernel::GetCellVerticesIdx(cell_id, static_cast<int>(current_cell_vertices_number), current_cell_vertices_idx,
                                      MAX_VERTEX_NUM, max_edge, verticesOnCell.data());

        bool is_inMesh = TBBKernel::IsInMesh(cell_id, max_edge, current_position,
                                             numberVertexOnCell.data(), verticesOnCell.data(), vertexCoord.data());
        if (!is_inMesh) {
            SetPixel(img_vec[0].mPixels, width, height, height_index, width_index, vec3_nan);
            if (bDoubleAttributes && img_vec.size() > 1) {
                SetPixel(img_vec[1].mPixels, width, height, height_index, width_index, vec3_nan);
            }
            return;
        }

        double current_point_ztop_vec[MAX_VERTLEVELS];
        vec3 vpos[MAX_VERTEX_NUM];
        double w[MAX_VERTEX_NUM];

        if (!TBBKernel::GetCellVertexPos(vpos, current_cell_vertices_idx, MAX_VERTEX_NUM,
                                         static_cast<int>(current_cell_vertices_number), vertexCoord.data())) {
            SetPixel(img_vec[0].mPixels, width, height, height_index, width_index, vec3_nan);
            if (bDoubleAttributes && img_vec.size() > 1) {
                SetPixel(img_vec[1].mPixels, width, height, height_index, width_index, vec3_nan);
            }
            return;
        }

        Interpolator::CalcPolygonWachspress(current_position, vpos, w, static_cast<int>(current_cell_vertices_number));

        const int ztop_levels = static_cast<int>(cellVertexZTop.size() / ACTUALL_VERTEX_SIZE);
        if (ztop_levels <= 0 || ztop_levels > MAX_VERTLEVELS) {
            SetPixel(img_vec[0].mPixels, width, height, height_index, width_index, vec3_nan);
            if (bDoubleAttributes && img_vec.size() > 1) {
                SetPixel(img_vec[1].mPixels, width, height, height_index, width_index, vec3_nan);
            }
            return;
        }

        for (int k = 0; k < ztop_levels; ++k) {
            double acc = 0.0;
            for (size_t v = 0; v < current_cell_vertices_number; ++v) {
                int VID = static_cast<int>(current_cell_vertices_idx[v]);
                double ztop = cellVertexZTop[VID * ztop_levels + k];
                acc += w[v] * ztop;
            }
            current_point_ztop_vec[k] = acc;
        }

        for (int k = 1; k < ztop_levels; ++k) {
            if (current_point_ztop_vec[k] > current_point_ztop_vec[k - 1]) {
                current_point_ztop_vec[k] = current_point_ztop_vec[k - 1] - 1e-9;
            }
        }

        double z_surf = current_point_ztop_vec[0];
        double z_bot = current_point_ztop_vec[ztop_levels - 1];
        if (z_surf < z_bot) {
            double t = z_surf;
            z_surf = z_bot;
            z_bot = t;
        }
        double epsd = std::max(1e-6, 1e-8 * MOPS::math::fabs(z_surf - z_bot));
        if (!(DEPTH <= z_surf + epsd && DEPTH >= z_bot - epsd)) {
            SetPixel(img_vec[0].mPixels, width, height, height_index, width_index, vec3_nan);
            if (bDoubleAttributes && img_vec.size() > 1) {
                SetPixel(img_vec[1].mPixels, width, height, height_index, width_index, vec3_nan);
            }
            return;
        }

        int local_layer = -1;
        for (int k = 1; k < ztop_levels; ++k) {
            double topI = current_point_ztop_vec[k - 1];
            double botI = current_point_ztop_vec[k];
            if (topI < botI) {
                double t = topI;
                topI = botI;
                botI = t;
            }
            if (DEPTH <= topI + 1e-8 && DEPTH >= botI - 1e-8) {
                local_layer = k;
                break;
            }
        }
        if (DEPTH <= current_point_ztop_vec[0]) {
            local_layer = 0;
        }
        if (local_layer < 0) {
            SetPixel(img_vec[0].mPixels, width, height, height_index, width_index, vec3_nan);
            if (bDoubleAttributes && img_vec.size() > 1) {
                SetPixel(img_vec[1].mPixels, width, height, height_index, width_index, vec3_nan);
            }
            return;
        }

        double topI = current_point_ztop_vec[std::max(0, local_layer - 1)];
        double botI = current_point_ztop_vec[local_layer];
        if (topI < botI) {
            double tmp = topI;
            topI = botI;
            botI = tmp;
        }

        double denom = topI - botI;
        double tparam = (denom > 1e-12) ? (DEPTH - botI) / denom : 0.5;

        const int vel_levels = static_cast<int>(cellVertexVelocity.size() / ACTUALL_VERTEX_SIZE);
        int j = std::clamp(local_layer - 1, 0, vel_levels - 1);
        int j_bot = std::min(j + 1, vel_levels - 1);
        int j_top = j;

        vec3 v_top = TBBKernel::CalcVelocity(current_cell_vertices_idx, w,
                                             MAX_VERTEX_NUM, static_cast<int>(current_cell_vertices_number), vel_levels, j_top,
                                             cellVertexVelocity.data());
        vec3 v_bot = TBBKernel::CalcVelocity(current_cell_vertices_idx, w,
                                             MAX_VERTEX_NUM, static_cast<int>(current_cell_vertices_number), vel_levels, j_bot,
                                             cellVertexVelocity.data());

        double mtop = MOPS_LENGTH(v_top), mbot = MOPS_LENGTH(v_bot);
        vec3 final_vel;
        if (mtop < 1e-12 && mbot < 1e-12) {
            final_vel = vec3{0.0, 0.0, 0.0};
        } else if (mtop < 1e-12) {
            final_vel = v_bot;
        } else if (mbot < 1e-12) {
            final_vel = v_top;
        } else {
            final_vel = (1.0 - tparam) * v_bot + tparam * v_top;
        }

        double u_east, v_north;
        GeoConverter::convertXYZVelocityToENU(current_position, final_vel, u_east, v_north);
        double spd = MOPS::math::sqrt(u_east * u_east + v_north * v_north);
        vec3 current_point_velocity_enu = {u_east, v_north, spd};

        vec3 current_point_attr_value = {0.0, 0.0, 0.0};
        if (bDoubleAttributes) {
            const int attr_count = static_cast<int>(attr_vec_ptrs.size());
            if (attr_count >= 1) {
                const auto& attr0 = *attr_vec_ptrs[0];
                const int attr_levels = static_cast<int>(attr0.size() / ACTUALL_VERTEX_SIZE);
                int aj = std::clamp(local_layer - 1, 0, attr_levels - 1);
                double a0 = TBBKernel::CalcAttribute(current_cell_vertices_idx, w,
                                                     MAX_VERTEX_NUM, static_cast<int>(current_cell_vertices_number),
                                                     attr_levels, aj, attr0.data());
                current_point_attr_value.x() = a0;
            }
            if (attr_count >= 2) {
                const auto& attr1 = *attr_vec_ptrs[1];
                const int attr_levels1 = static_cast<int>(attr1.size() / ACTUALL_VERTEX_SIZE);
                int aj1 = std::clamp(local_layer - 1, 0, attr_levels1 - 1);
                double a1 = TBBKernel::CalcAttribute(current_cell_vertices_idx, w,
                                                     MAX_VERTEX_NUM, static_cast<int>(current_cell_vertices_number),
                                                     attr_levels1, aj1, attr1.data());
                current_point_attr_value.y() = a1;
            }
        }

        SetPixel(img_vec[0].mPixels, width, height, height_index, width_index, current_point_velocity_enu);
        if (bDoubleAttributes && img_vec.size() > 1) {
            SetPixel(img_vec[1].mPixels, width, height, height_index, width_index, current_point_attr_value);
        }
    });
}

void VisualizeFixedLatitude(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img)
{
    if (mpasoF == nullptr || mpasoF->mGrid == nullptr || mpasoF->mSol_Front == nullptr || img == nullptr) {
        Error("[TBBBackend::Kernel]::VisualizeFixedLatitude invalid inputs");
        return;
    }

    const int width = static_cast<int>(config->imageSize.x());
    const int height = static_cast<int>(config->imageSize.y());
    if (width <= 0 || height <= 0) {
        Error("[TBBBackend::Kernel]::VisualizeFixedLatitude invalid image size");
        return;
    }

    auto* grid = mpasoF->mGrid.get();
    const auto* sol = mpasoF->mSol_Front.get();

    const auto& refBottomDepth = grid->cellRefBottomDepth_vec;
    if (refBottomDepth.empty()) {
        Error("[TBBBackend::Kernel]::VisualizeFixedLatitude refBottomDepth is empty");
        return;
    }

    const double minDepth = refBottomDepth.front();
    const double maxDepth = refBottomDepth.back();
    const double minLon = config->LonRange.x();
    const double maxLon = config->LonRange.y();
    const double fixed_lat = config->FixedLatitude;

    const int max_vertex_num = 20;
    const int total_ztop_layer = std::max(1, static_cast<int>(sol->cellVertexZTop_vec.size() /
                                      std::max<size_t>(1, grid->mVertexSize)));
    const int total_vel_layer = std::max(1, static_cast<int>(sol->cellVertexVelocity_vec.size() /
                                     std::max<size_t>(1, grid->mVertexSize)));
    const int nVert = std::min(total_ztop_layer, total_vel_layer);
    if (nVert <= 0) {
        Error("[TBBBackend::Kernel]::VisualizeFixedLatitude invalid layer metadata");
        return;
    }

    const double i_step = (height > 1) ? (maxDepth - minDepth) / (height - 1) : 0.0;
    const double j_step = (width > 1) ? (maxLon - minLon) / (width - 1) : 0.0;

    std::vector<int> cell_id_vec(width * height, -1);
    tbb::parallel_for(0, width * height, [&](int gid) {
        const int ih = gid / width;
        const int jw = gid % width;
        const double lon = minLon + jw * j_step;
        SphericalCoord latlon_r = vec2(fixed_lat * (M_PI / 180.0), lon * (M_PI / 180.0));
        CartesianCoord current_position;
        GeoConverter::convertRadianLatLonToXYZ(latlon_r, current_position);
        int cell_id_value = -1;
        grid->searchKDT(current_position, cell_id_value);
        cell_id_vec[gid] = cell_id_value;
    });

    const auto double_nan = std::numeric_limits<double>::quiet_NaN();
    const vec3 vec3_nan = {double_nan, double_nan, double_nan};

    tbb::parallel_for(0, width * height, [&](int gid) {
        const int ih = gid / width;
        const int jw = gid % width;

        const double depth_plot = minDepth + ih * i_step;
        const double DEPTH = -MOPS::math::fabs(depth_plot);
        const double lon = minLon + jw * j_step;

        SphericalCoord latlon_r = vec2(fixed_lat * (M_PI / 180.0), lon * (M_PI / 180.0));
        CartesianCoord position;
        GeoConverter::convertRadianLatLonToXYZ(latlon_r, position);

        int cell_id = cell_id_vec[gid];
        if (cell_id < 0 || cell_id >= static_cast<int>(grid->mCellsSize)) {
            img->setPixel(ih, jw, vec3_nan);
            return;
        }

        std::vector<size_t> current_cell_vertices_idx;
        const bool is_land = mpasoF->isOnOcean(position, cell_id, current_cell_vertices_idx);
        if (is_land || current_cell_vertices_idx.empty()) {
            img->setPixel(ih, jw, vec3_nan);
            return;
        }

        const int current_cell_vertices_number = static_cast<int>(current_cell_vertices_idx.size());
        if (current_cell_vertices_number <= 0 || current_cell_vertices_number > max_vertex_num) {
            img->setPixel(ih, jw, vec3_nan);
            return;
        }

        vec3 current_cell_vertex_pos[max_vertex_num];
        for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
            current_cell_vertex_pos[v_idx] = grid->vertexCoord_vec[current_cell_vertices_idx[v_idx]];
        }

        double current_cell_vertex_weight[max_vertex_num] = {0.0};
        Interpolator::CalcPolygonWachspress(position, current_cell_vertex_pos,
                                            current_cell_vertex_weight, current_cell_vertices_number);

        std::vector<double> current_point_ztop_vec(nVert, 0.0);
        for (int k = 0; k < nVert; ++k) {
            double z_acc = 0.0;
            for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
                const auto vid = current_cell_vertices_idx[v_idx];
                const double ztop = sol->cellVertexZTop_vec[vid * total_ztop_layer + k];
                z_acc += current_cell_vertex_weight[v_idx] * ztop;
            }
            current_point_ztop_vec[k] = z_acc;
        }

        for (int k = 1; k < nVert; ++k) {
            if (current_point_ztop_vec[k] > current_point_ztop_vec[k - 1]) {
                current_point_ztop_vec[k] = current_point_ztop_vec[k - 1] - 1e-9;
            }
        }

        int layer = -1;
        const double EPSILON = 1e-6;

        if (DEPTH > current_point_ztop_vec[0] + EPSILON || DEPTH < current_point_ztop_vec[nVert - 1] - EPSILON) {
            img->setPixel(ih, jw, vec3_nan);
            return;
        }

        for (int k = 1; k < nVert; ++k) {
            double z_up = current_point_ztop_vec[k - 1];
            double z_dn = current_point_ztop_vec[k];
            if (z_up < z_dn) {
                std::swap(z_up, z_dn);
            }
            if (DEPTH <= z_up + EPSILON && DEPTH >= z_dn - EPSILON) {
                layer = k;
                break;
            }
        }

        if (layer == -1) {
            img->setPixel(ih, jw, vec3_nan);
            return;
        }

        double ztop_layer_dn = current_point_ztop_vec[layer];
        double ztop_layer_up = current_point_ztop_vec[layer - 1];
        if (ztop_layer_up < ztop_layer_dn) {
            std::swap(ztop_layer_up, ztop_layer_dn);
        }

        const double denom = ztop_layer_up - ztop_layer_dn;
        if (MOPS::math::fabs(denom) < 1e-30) {
            img->setPixel(ih, jw, vec3_nan);
            return;
        }

        const double t = (DEPTH - ztop_layer_dn) / denom;

        vec3 current_point_vel_up = {0.0, 0.0, 0.0};
        vec3 current_point_vel_dn = {0.0, 0.0, 0.0};
        for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
            const auto vid = current_cell_vertices_idx[v_idx];
            const vec3 vel_up = sol->cellVertexVelocity_vec[vid * total_vel_layer + (layer - 1)];
            const vec3 vel_dn = sol->cellVertexVelocity_vec[vid * total_vel_layer + layer];

            current_point_vel_up.x() += current_cell_vertex_weight[v_idx] * vel_up.x();
            current_point_vel_up.y() += current_cell_vertex_weight[v_idx] * vel_up.y();
            current_point_vel_up.z() += current_cell_vertex_weight[v_idx] * vel_up.z();

            current_point_vel_dn.x() += current_cell_vertex_weight[v_idx] * vel_dn.x();
            current_point_vel_dn.y() += current_cell_vertex_weight[v_idx] * vel_dn.y();
            current_point_vel_dn.z() += current_cell_vertex_weight[v_idx] * vel_dn.z();
        }

        const vec3 final_vel = (1.0 - t) * current_point_vel_dn + t * current_point_vel_up;

        double zonal_velocity = 0.0;
        double meridional_velocity = 0.0;
        GeoConverter::convertXYZVelocityToENU(position, final_vel, zonal_velocity, meridional_velocity);
        img->setPixel(ih, jw, vec3(zonal_velocity, meridional_velocity, 0.0));
    });
}

std::vector<TrajectoryLine> StreamLine(
    MPASOField* mpasoF,
    std::vector<CartesianCoord>& points,
    TrajectorySettings* config,
    std::vector<int>& default_cell_id)
{
    if (mpasoF == nullptr || mpasoF->mGrid == nullptr || mpasoF->mSol_Front == nullptr || config == nullptr) {
        Error("[TBBBackend::Kernel]::StreamLine invalid inputs");
        return {};
    }
    if (points.empty()) {
        return {};
    }
    if (config->deltaT == 0 || config->recordT == 0 || config->simulationDuration == 0) {
        Error("[TBBBackend::Kernel]::StreamLine invalid trajectory settings");
        return {};
    }

    std::vector<vec3> stable_points = points;
    auto host_buffers = MOPS::Common::InitTrajectoryOutputBuffers(
        stable_points.size(),
        static_cast<int>(config->simulationDuration),
        static_cast<int>(config->recordT),
        false);
    std::vector<vec3>& update_points = host_buffers.points;
    std::vector<vec3>& update_vels = host_buffers.velocities;

    std::vector<float> effective_depths = MOPS::Common::BuildEffectiveDepths(stable_points, config, "TBB::StreamLine");
    std::vector<TrajectoryLine> trajectory_lines = MOPS::Common::InitTrajectoryLines(stable_points, effective_depths, config);

    if (default_cell_id.size() != stable_points.size()) {
        default_cell_id.assign(stable_points.size(), -1);
    }
    tbb::parallel_for(size_t(0), stable_points.size(), [&](size_t i) {
        if (default_cell_id[i] < 0) {
            mpasoF->mGrid->searchKDT(stable_points[i], default_cell_id[i]);
        }
    });

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(mpasoF->mGrid.get());
    if (grid_info_vec.size() < 6) {
        Error("[TBBBackend::Kernel]::StreamLine grid_info is incomplete");
        return {};
    }

    const int actual_cell_size = static_cast<int>(grid_info_vec[0]);
    const int actual_max_edge_size = static_cast<int>(grid_info_vec[2]);
    const int actual_vertex_size = static_cast<int>(grid_info_vec[3]);
    const int actual_ztop_layer = static_cast<int>(grid_info_vec[4]);
    const int actual_ztop_layer_p1 = static_cast<int>(grid_info_vec[5]);
    const int each_points_size = static_cast<int>(config->simulationDuration / config->recordT);
    const int times = static_cast<int>(config->simulationDuration / config->deltaT);
    const int dt_sign = (config->directionType == MOPS::CalcDirection::kForward) ? 1 : -1;
    const int delta_t = dt_sign * static_cast<int>(config->deltaT);
    const bool use_euler = (config->methodType == MOPS::CalcMethodType::kEuler);

    if (each_points_size <= 0 || times <= 0) {
        Error("[TBBBackend::Kernel]::StreamLine invalid integration steps");
        return trajectory_lines;
    }

    const auto* number_vertex_on_cell = mpasoF->mGrid->numberVertexOnCell_vec.data();
    const auto* vertices_on_cell = mpasoF->mGrid->verticesOnCell_vec.data();
    const auto* cells_on_cell = mpasoF->mGrid->cellsOnCell_vec.data();
    const auto* vertex_coord = mpasoF->mGrid->vertexCoord_vec.data();
    const auto* cell_coord = mpasoF->mGrid->cellCoord_vec.data();
    const auto* cell_vertex_velocity = mpasoF->mSol_Front->cellVertexVelocity_vec.data();
    const auto* cell_vertex_ztop = mpasoF->mSol_Front->cellVertexZTop_vec.data();
    const auto* cell_vertex_vert_velocity = mpasoF->mSol_Front->cellVertexVertVelocity_vec.data();

    struct VelocityState {
        vec3 h_vel;
        double v_vel;
        bool ok;
    };

    auto advect_on_sphere = [](const vec3& pos, const vec3& vel, double dt_local) -> vec3 {
        const double rr = MOPS_LENGTH(pos);
        const double speed_local = MOPS_LENGTH(vel);
        if (rr < 1e-12 || speed_local < 1e-12) {
            return pos;
        }
        vec3 axis = TBBKernel::CalcRotationAxis(pos, vel);
        const double theta = (speed_local * dt_local) / rr;
        return TBBKernel::CalcPositionAfterRotation(pos, axis, theta);
    };

    auto calc_velocity_at = [&](const vec3& pos, int cell_id, double current_depth) -> VelocityState {
        constexpr int MAX_VERTEX_NUM = 20;
        constexpr int MAX_VERTICAL_LEVEL_NUM = 100;

        if (cell_id < 0 || actual_ztop_layer <= 1 || actual_ztop_layer > MAX_VERTICAL_LEVEL_NUM) {
            return {vec3(0.0, 0.0, 0.0), 0.0, false};
        }

        const int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
        if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
            return {vec3(0.0, 0.0, 0.0), 0.0, false};
        }

        if (!TBBKernel::IsInMesh(cell_id, actual_max_edge_size, pos,
                                 number_vertex_on_cell, vertices_on_cell, vertex_coord)) {
            return {vec3(0.0, 0.0, 0.0), 0.0, false};
        }

        size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
        TBBKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx,
                                      MAX_VERTEX_NUM, actual_max_edge_size, vertices_on_cell);

        vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
        if (!TBBKernel::GetCellVertexPos(current_cell_vertex_pos, current_cell_vertices_idx,
                                         MAX_VERTEX_NUM, current_cell_vertices_number, vertex_coord)) {
            return {vec3(0.0, 0.0, 0.0), 0.0, false};
        }

        double current_cell_vertex_weight[MAX_VERTEX_NUM] = {0.0};
        Interpolator::CalcPolygonWachspress(pos, current_cell_vertex_pos,
                                            current_cell_vertex_weight, current_cell_vertices_number);

        double current_point_ztop_vec[MAX_VERTICAL_LEVEL_NUM];
        for (int k = 0; k < actual_ztop_layer; ++k) {
            double z = 0.0;
            for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
                const int vid = static_cast<int>(current_cell_vertices_idx[v_idx]);
                if (vid < 0 || vid >= actual_vertex_size) {
                    return {vec3(0.0, 0.0, 0.0), 0.0, false};
                }
                z += current_cell_vertex_weight[v_idx] * cell_vertex_ztop[vid * actual_ztop_layer + k];
            }
            current_point_ztop_vec[k] = z;
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
                const int mid = (lo + hi) >> 1;
                const double top_i = current_point_ztop_vec[mid - 1];
                const double bot_i = current_point_ztop_vec[mid];
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
            return {vec3(0.0, 0.0, 0.0), 0.0, false};
        }

        const double ztop_dn = current_point_ztop_vec[local_layer];
        const double ztop_up = current_point_ztop_vec[local_layer - 1];
        double x = current_depth;
        x = std::max(ztop_dn, std::min(x, ztop_up));
        const double denom = ztop_up - ztop_dn;
        if (MOPS::math::fabs(denom) < 1e-12) {
            return {vec3(0.0, 0.0, 0.0), 0.0, false};
        }
        const double t = (x - ztop_dn) / denom;

        vec3 current_point_vel_dn = TBBKernel::CalcVelocity(
            current_cell_vertices_idx, current_cell_vertex_weight, MAX_VERTEX_NUM,
            current_cell_vertices_number, actual_ztop_layer, local_layer, cell_vertex_velocity);
        vec3 current_point_vel_up = TBBKernel::CalcVelocity(
            current_cell_vertices_idx, current_cell_vertex_weight, MAX_VERTEX_NUM,
            current_cell_vertices_number, actual_ztop_layer, local_layer - 1, cell_vertex_velocity);

        if (MOPS_LENGTH(current_point_vel_dn) < 1e-12 || MOPS_LENGTH(current_point_vel_up) < 1e-12) {
            return {vec3(0.0, 0.0, 0.0), 0.0, false};
        }

        vec3 final_vel = t * current_point_vel_up + (1.0 - t) * current_point_vel_dn;
        if (MOPS_LENGTH(final_vel) < 1e-12) {
            return {vec3(0.0, 0.0, 0.0), 0.0, false};
        }

        int dn_if = local_layer;
        int up_if = (local_layer > 0) ? (local_layer - 1) : 0;
        if (dn_if >= actual_ztop_layer_p1) {
            dn_if = actual_ztop_layer_p1 - 1;
        }
        if (up_if >= actual_ztop_layer_p1) {
            up_if = actual_ztop_layer_p1 - 1;
        }

        const double w_dn = TBBKernel::CalcAttribute(
            current_cell_vertices_idx, current_cell_vertex_weight, MAX_VERTEX_NUM,
            current_cell_vertices_number, actual_ztop_layer_p1, dn_if, cell_vertex_vert_velocity);
        const double w_up = TBBKernel::CalcAttribute(
            current_cell_vertices_idx, current_cell_vertex_weight, MAX_VERTEX_NUM,
            current_cell_vertices_number, actual_ztop_layer_p1, up_if, cell_vertex_vert_velocity);

        const double vertical_vel = t * w_up + (1.0 - t) * w_dn;
        return {final_vel, vertical_vel, true};
    };

    tbb::parallel_for(size_t(0), stable_points.size(), [&](size_t pid) {
        constexpr int MAX_CELL_NEIGHBOR_NUM = 21;
        int run_time = 0;
        bool first_loop = true;
        bool first_vel = true;
        const int base_idx = static_cast<int>(pid) * each_points_size;
        int update_points_idx = 0;
        int cell_id = -1;
        int cell_neig_vec[MAX_CELL_NEIGHBOR_NUM];
        for (int& v : cell_neig_vec) {
            v = -1;
        }

        for (int times_i = 0; times_i < times; ++times_i) {
            run_time += std::abs(delta_t);
            vec3 sample_point_position = stable_points[pid];
            const double current_depth = -1.0 * static_cast<double>(effective_depths[pid]);

            if (first_loop) {
                first_loop = false;
                cell_id = default_cell_id[pid];
                if (cell_id < 0 || cell_id >= actual_cell_size) {
                    return;
                }
                int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
                TBBKernel::GetCellNeighborsIdx(cell_id, current_cell_vertices_number, cell_neig_vec,
                                               MAX_CELL_NEIGHBOR_NUM, actual_max_edge_size, cells_on_cell);
                update_points[base_idx] = sample_point_position;
            } else {
                if (cell_id < 0 || cell_id >= actual_cell_size) {
                    return;
                }
                int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
                double min_len = std::numeric_limits<double>::max();
                for (int n = 0; n < current_cell_vertices_number + 1; ++n) {
                    int cid = cell_neig_vec[n];
                    if (cid < 0 || cid >= actual_cell_size) {
                        continue;
                    }
                    const double len = MOPS_LENGTH(cell_coord[cid] - sample_point_position);
                    if (len < min_len) {
                        min_len = len;
                        cell_id = cid;
                    }
                }
                current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
                TBBKernel::GetCellNeighborsIdx(cell_id, current_cell_vertices_number, cell_neig_vec,
                                               MAX_CELL_NEIGHBOR_NUM, actual_max_edge_size, cells_on_cell);
            }

            const vec3 current_position = sample_point_position;
            const double r = MOPS_LENGTH(current_position);
            vec3 rk4_next_position = current_position;
            vec3 current_horizontal_velocity = vec3(0.0, 0.0, 0.0);
            double current_vertical_velocity = 0.0;

            if (use_euler) {
                auto state = calc_velocity_at(current_position, cell_id, current_depth);
                if (!state.ok) {
                    return;
                }
                current_horizontal_velocity = state.h_vel;
                current_vertical_velocity = state.v_vel;
            } else {
                const double dt = static_cast<double>(delta_t);
                auto s1 = calc_velocity_at(current_position, cell_id, current_depth);
                if (!s1.ok) {
                    return;
                }
                vec3 p2 = advect_on_sphere(current_position, s1.h_vel, dt * 0.5);
                auto s2 = calc_velocity_at(p2, cell_id, current_depth);
                if (!s2.ok) {
                    return;
                }
                vec3 p3 = advect_on_sphere(current_position, s2.h_vel, dt * 0.5);
                auto s3 = calc_velocity_at(p3, cell_id, current_depth);
                if (!s3.ok) {
                    return;
                }
                vec3 p4 = advect_on_sphere(current_position, s3.h_vel, dt);
                auto s4 = calc_velocity_at(p4, cell_id, current_depth);
                if (!s4.ok) {
                    return;
                }

                current_horizontal_velocity = (s1.h_vel + 2.0 * s2.h_vel + 2.0 * s3.h_vel + s4.h_vel) / 6.0;
                current_vertical_velocity = (s1.v_vel + 2.0 * s2.v_vel + 2.0 * s3.v_vel + s4.v_vel) / 6.0;

                vec3 x_trial = current_position + current_horizontal_velocity * dt;
                const double x_trial_len = MOPS_LENGTH(x_trial);
                rk4_next_position = (x_trial_len > 1e-12) ? (x_trial / x_trial_len) * r : current_position;
            }

            vec3 new_position;
            if (use_euler) {
                vec3 rotation_axis = TBBKernel::CalcRotationAxis(current_position, current_horizontal_velocity);
                const double speed = MOPS_LENGTH(current_horizontal_velocity);
                const double theta_rad = (speed * delta_t) / std::max(1e-12, r);
                new_position = TBBKernel::CalcPositionAfterRotation(current_position, rotation_axis, theta_rad);
            } else {
                new_position = rk4_next_position;
            }

            const double old_depth = static_cast<double>(effective_depths[pid]);
            double new_depth = old_depth - current_vertical_velocity * static_cast<double>(delta_t);
            new_depth = std::max(0.0, new_depth);
            const double r_new = std::max(1.0, r + current_vertical_velocity * static_cast<double>(delta_t));
            effective_depths[pid] = static_cast<float>(new_depth);

            const double nlen = MOPS_LENGTH(new_position);
            if (nlen > 1e-12) {
                new_position = (new_position / nlen) * r_new;
            }

            if (first_vel) {
                first_vel = false;
                update_vels[base_idx] = current_horizontal_velocity;
            }

            stable_points[pid] = new_position;
            if (config->recordT > 0 && (run_time % static_cast<int>(config->recordT)) == 0) {
                int write_idx = base_idx + update_points_idx;
                if (write_idx >= base_idx && write_idx < base_idx + each_points_size) {
                    update_points[write_idx] = new_position;
                    update_vels[write_idx] = current_horizontal_velocity;
                }
                ++update_points_idx;
            }
        }
    });

    const size_t each_point_size = config->simulationDuration / config->recordT;
    const size_t total_points = update_points.size();
    auto clean_traj = MOPS::Common::FinalizeTrajectoryLines(
        trajectory_lines,
        update_points,
        update_vels,
        each_point_size,
        total_points);
    trajectory_lines.clear();
    return clean_traj;
}

std::vector<TrajectoryLine> PathLine(
    MPASOField* mpasoF,
    std::vector<CartesianCoord>& points,
    TrajectorySettings* config,
    std::vector<int>& default_cell_id)
{
    if (mpasoF == nullptr || mpasoF->mGrid == nullptr || mpasoF->mSol_Front == nullptr || mpasoF->mSol_Back == nullptr || config == nullptr) {
        Error("[TBBBackend::Kernel]::PathLine invalid inputs");
        return {};
    }
    if (points.empty()) {
        return {};
    }
    if (config->deltaT == 0 || config->recordT == 0 || config->simulationDuration == 0) {
        Error("[TBBBackend::Kernel]::PathLine invalid trajectory settings");
        return {};
    }

    std::vector<vec3> stable_points = points;
    auto host_buffers = MOPS::Common::InitTrajectoryOutputBuffers(
        stable_points.size(),
        static_cast<int>(config->simulationDuration),
        static_cast<int>(config->recordT),
        true);
    std::vector<vec3>& update_points = host_buffers.points;
    std::vector<vec3>& update_vels = host_buffers.velocities;
    std::vector<vec3>& update_attrs = host_buffers.attrs;

    std::vector<float> effective_depths = MOPS::Common::BuildEffectiveDepths(stable_points, config, "TBB::PathLine");
    std::vector<TrajectoryLine> trajectory_lines = MOPS::Common::InitTrajectoryLines(stable_points, effective_depths, config);

    if (default_cell_id.size() != stable_points.size()) {
        default_cell_id.assign(stable_points.size(), -1);
    }
    tbb::parallel_for(size_t(0), stable_points.size(), [&](size_t i) {
        if (default_cell_id[i] < 0) {
            mpasoF->mGrid->searchKDT(stable_points[i], default_cell_id[i]);
        }
    });

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(mpasoF->mGrid.get());
    if (grid_info_vec.size() < 6) {
        Error("[TBBBackend::Kernel]::PathLine grid_info is incomplete");
        return {};
    }

    const int actual_cell_size = static_cast<int>(grid_info_vec[0]);
    const int actual_max_edge_size = static_cast<int>(grid_info_vec[2]);
    const int actual_vertex_size = static_cast<int>(grid_info_vec[3]);
    const int actual_ztop_layer = static_cast<int>(grid_info_vec[4]);
    const int actual_ztop_layer_p1 = static_cast<int>(grid_info_vec[5]);
    const int each_points_size = static_cast<int>(config->simulationDuration / config->recordT);
    const int n_steps = static_cast<int>(config->simulationDuration / config->deltaT);
    const int dt_sign = (config->directionType == MOPS::CalcDirection::kForward) ? 1 : -1;
    const int delta_t = dt_sign * static_cast<int>(config->deltaT);
    const bool use_euler = (config->methodType == MOPS::CalcMethodType::kEuler);
    if (each_points_size <= 0 || n_steps <= 0) {
        Error("[TBBBackend::Kernel]::PathLine invalid integration steps");
        return trajectory_lines;
    }

    const auto* number_vertex_on_cell = mpasoF->mGrid->numberVertexOnCell_vec.data();
    const auto* vertices_on_cell = mpasoF->mGrid->verticesOnCell_vec.data();
    const auto* cells_on_cell = mpasoF->mGrid->cellsOnCell_vec.data();
    const auto* vertex_coord = mpasoF->mGrid->vertexCoord_vec.data();
    const auto* cell_coord = mpasoF->mGrid->cellCoord_vec.data();

    const auto* cell_vertex_velocity_front = mpasoF->mSol_Front->cellVertexVelocity_vec.data();
    const auto* cell_vertex_ztop_front = mpasoF->mSol_Front->cellVertexZTop_vec.data();
    const auto* cell_vertex_vert_velocity_front = mpasoF->mSol_Front->cellVertexVertVelocity_vec.data();
    const auto* cell_vertex_velocity_back = mpasoF->mSol_Back->cellVertexVelocity_vec.data();
    const auto* cell_vertex_ztop_back = mpasoF->mSol_Back->cellVertexZTop_vec.data();
    const auto* cell_vertex_vert_velocity_back = mpasoF->mSol_Back->cellVertexVertVelocity_vec.data();

    std::vector<const std::vector<double>*> attr_front_ptrs;
    std::vector<const std::vector<double>*> attr_back_ptrs;
    if (mpasoF->mSol_Front->mDoubleAttributes.size() > 1) {
        for (const auto& [name, vec] : mpasoF->mSol_Front->mDoubleAttributes_CtoV) {
            (void)name;
            attr_front_ptrs.push_back(&vec);
        }
        for (const auto& [name, vec] : mpasoF->mSol_Back->mDoubleAttributes_CtoV) {
            (void)name;
            attr_back_ptrs.push_back(&vec);
        }
    }
    const int attr_count = static_cast<int>(std::min(attr_front_ptrs.size(), attr_back_ptrs.size()));
    const bool b_double_attributes = (attr_count > 0);

    struct PathVelocityState {
        vec3 h_vel;
        double v_vel;
        vec3 attr;
        bool ok;
    };

    auto advect_on_sphere = [](const vec3& pos, const vec3& vel, double dt_local) -> vec3 {
        const double rr = MOPS_LENGTH(pos);
        const double speed_local = MOPS_LENGTH(vel);
        if (rr < 1e-12 || speed_local < 1e-12) {
            return pos;
        }
        vec3 axis = TBBKernel::CalcRotationAxis(pos, vel);
        const double theta = (speed_local * dt_local) / rr;
        return TBBKernel::CalcPositionAfterRotation(pos, axis, theta);
    };

    auto calc_velocity_at = [&](const vec3& pos, int cell_id, double current_depth, double alpha) -> PathVelocityState {
        constexpr int MAX_VERTEX_NUM = 20;
        constexpr int MAX_VERTICAL_LEVEL_NUM = 100;

        if (cell_id < 0 || actual_ztop_layer <= 1 || actual_ztop_layer > MAX_VERTICAL_LEVEL_NUM) {
            return {vec3(0.0, 0.0, 0.0), 0.0, vec3(0.0, 0.0, 0.0), false};
        }

        const int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
        if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
            return {vec3(0.0, 0.0, 0.0), 0.0, vec3(0.0, 0.0, 0.0), false};
        }

        if (!TBBKernel::IsInMesh(cell_id, actual_max_edge_size, pos,
                                 number_vertex_on_cell, vertices_on_cell, vertex_coord)) {
            return {vec3(0.0, 0.0, 0.0), 0.0, vec3(0.0, 0.0, 0.0), false};
        }

        size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
        TBBKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx,
                                      MAX_VERTEX_NUM, actual_max_edge_size, vertices_on_cell);

        vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
        if (!TBBKernel::GetCellVertexPos(current_cell_vertex_pos, current_cell_vertices_idx,
                                         MAX_VERTEX_NUM, current_cell_vertices_number, vertex_coord)) {
            return {vec3(0.0, 0.0, 0.0), 0.0, vec3(0.0, 0.0, 0.0), false};
        }

        double current_cell_vertex_weight[MAX_VERTEX_NUM] = {0.0};
        Interpolator::CalcPolygonWachspress(pos, current_cell_vertex_pos,
                                            current_cell_vertex_weight, current_cell_vertices_number);

        double ztop_front[MAX_VERTICAL_LEVEL_NUM];
        double ztop_back[MAX_VERTICAL_LEVEL_NUM];
        for (int k = 0; k < actual_ztop_layer; ++k) {
            double zf = 0.0;
            double zb = 0.0;
            for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
                const int vid = static_cast<int>(current_cell_vertices_idx[v_idx]);
                if (vid < 0 || vid >= actual_vertex_size) {
                    return {vec3(0.0, 0.0, 0.0), 0.0, vec3(0.0, 0.0, 0.0), false};
                }
                zf += current_cell_vertex_weight[v_idx] * cell_vertex_ztop_front[vid * actual_ztop_layer + k];
                zb += current_cell_vertex_weight[v_idx] * cell_vertex_ztop_back[vid * actual_ztop_layer + k];
            }
            ztop_front[k] = zf;
            ztop_back[k] = zb;
        }

        for (int k = 1; k < actual_ztop_layer; ++k) {
            if (ztop_front[k] > ztop_front[k - 1]) {
                ztop_front[k] = ztop_front[k - 1] - 1e-9;
            }
            if (ztop_back[k] > ztop_back[k - 1]) {
                ztop_back[k] = ztop_back[k - 1] - 1e-9;
            }
        }

        const double eps = 1e-8;
        int local_layer_front = -1;
        int local_layer_back = -1;
        bool skip_front = false;
        bool skip_back = false;

        if (current_depth > ztop_front[0] + eps) {
            local_layer_front = 0;
            skip_front = true;
        } else if (current_depth < ztop_front[actual_ztop_layer - 1] - eps) {
            local_layer_front = actual_ztop_layer - 1;
            skip_front = true;
        }
        if (current_depth > ztop_back[0] + eps) {
            local_layer_back = 0;
            skip_back = true;
        } else if (current_depth < ztop_back[actual_ztop_layer - 1] - eps) {
            local_layer_back = actual_ztop_layer - 1;
            skip_back = true;
        }

        if (!skip_front) {
            for (int k = 1; k < actual_ztop_layer; ++k) {
                if (current_depth <= ztop_front[k - 1] + eps && current_depth >= ztop_front[k] - eps) {
                    local_layer_front = k;
                    break;
                }
            }
        }
        if (!skip_back) {
            for (int k = 1; k < actual_ztop_layer; ++k) {
                if (current_depth <= ztop_back[k - 1] + eps && current_depth >= ztop_back[k] - eps) {
                    local_layer_back = k;
                    break;
                }
            }
        }

        if (local_layer_front < 0 || local_layer_back < 0) {
            return {vec3(0.0, 0.0, 0.0), 0.0, vec3(0.0, 0.0, 0.0), false};
        }

        const double ztop_front_dn = ztop_front[local_layer_front];
        const double ztop_front_up = ztop_front[local_layer_front - 1];
        const double ztop_back_dn = ztop_back[local_layer_back];
        const double ztop_back_up = ztop_back[local_layer_back - 1];

        double x_front = std::max(ztop_front_dn, std::min(current_depth, ztop_front_up));
        double denom_front = ztop_front_up - ztop_front_dn;
        if (MOPS::math::fabs(denom_front) < 1e-12) {
            return {vec3(0.0, 0.0, 0.0), 0.0, vec3(0.0, 0.0, 0.0), false};
        }
        double t_front = (x_front - ztop_front_dn) / denom_front;

        double x_back = std::max(ztop_back_dn, std::min(current_depth, ztop_back_up));
        double denom_back = ztop_back_up - ztop_back_dn;
        if (MOPS::math::fabs(denom_back) < 1e-12) {
            return {vec3(0.0, 0.0, 0.0), 0.0, vec3(0.0, 0.0, 0.0), false};
        }
        double t_back = (x_back - ztop_back_dn) / denom_back;

        vec3 vel_dn_front = TBBKernel::CalcVelocity(
            current_cell_vertices_idx, current_cell_vertex_weight, MAX_VERTEX_NUM,
            current_cell_vertices_number, actual_ztop_layer, local_layer_front, cell_vertex_velocity_front);
        vec3 vel_up_front = TBBKernel::CalcVelocity(
            current_cell_vertices_idx, current_cell_vertex_weight, MAX_VERTEX_NUM,
            current_cell_vertices_number, actual_ztop_layer, local_layer_front - 1, cell_vertex_velocity_front);
        vec3 final_vel_front = t_front * vel_up_front + (1.0 - t_front) * vel_dn_front;

        vec3 vel_dn_back = TBBKernel::CalcVelocity(
            current_cell_vertices_idx, current_cell_vertex_weight, MAX_VERTEX_NUM,
            current_cell_vertices_number, actual_ztop_layer, local_layer_back, cell_vertex_velocity_back);
        vec3 vel_up_back = TBBKernel::CalcVelocity(
            current_cell_vertices_idx, current_cell_vertex_weight, MAX_VERTEX_NUM,
            current_cell_vertices_number, actual_ztop_layer, local_layer_back - 1, cell_vertex_velocity_back);
        vec3 final_vel_back = t_back * vel_up_back + (1.0 - t_back) * vel_dn_back;

        vec3 current_horizontal_velocity = alpha * final_vel_back + (1.0 - alpha) * final_vel_front;

        int dn_if_front = local_layer_front;
        int up_if_front = (local_layer_front > 0) ? (local_layer_front - 1) : 0;
        int dn_if_back = local_layer_back;
        int up_if_back = (local_layer_back > 0) ? (local_layer_back - 1) : 0;
        if (dn_if_front >= actual_ztop_layer_p1) dn_if_front = actual_ztop_layer_p1 - 1;
        if (up_if_front >= actual_ztop_layer_p1) up_if_front = actual_ztop_layer_p1 - 1;
        if (dn_if_back >= actual_ztop_layer_p1) dn_if_back = actual_ztop_layer_p1 - 1;
        if (up_if_back >= actual_ztop_layer_p1) up_if_back = actual_ztop_layer_p1 - 1;

        double w_dn_front = TBBKernel::CalcAttribute(
            current_cell_vertices_idx, current_cell_vertex_weight, MAX_VERTEX_NUM,
            current_cell_vertices_number, actual_ztop_layer_p1, dn_if_front, cell_vertex_vert_velocity_front);
        double w_up_front = TBBKernel::CalcAttribute(
            current_cell_vertices_idx, current_cell_vertex_weight, MAX_VERTEX_NUM,
            current_cell_vertices_number, actual_ztop_layer_p1, up_if_front, cell_vertex_vert_velocity_front);
        double w_front = t_front * w_up_front + (1.0 - t_front) * w_dn_front;

        double w_dn_back = TBBKernel::CalcAttribute(
            current_cell_vertices_idx, current_cell_vertex_weight, MAX_VERTEX_NUM,
            current_cell_vertices_number, actual_ztop_layer_p1, dn_if_back, cell_vertex_vert_velocity_back);
        double w_up_back = TBBKernel::CalcAttribute(
            current_cell_vertices_idx, current_cell_vertex_weight, MAX_VERTEX_NUM,
            current_cell_vertices_number, actual_ztop_layer_p1, up_if_back, cell_vertex_vert_velocity_back);
        double w_back = t_back * w_up_back + (1.0 - t_back) * w_dn_back;

        double current_vertical_velocity = alpha * w_back + (1.0 - alpha) * w_front;

        vec3 current_attr = vec3(0.0, 0.0, 0.0);
        if (b_double_attributes) {
            if (attr_count >= 1) {
                const auto& af = *attr_front_ptrs[0];
                const auto& ab = *attr_back_ptrs[0];
                double attr_dn_front = TBBKernel::CalcAttribute(current_cell_vertices_idx, current_cell_vertex_weight,
                    MAX_VERTEX_NUM, current_cell_vertices_number, actual_ztop_layer, local_layer_front, af.data());
                double attr_up_front = TBBKernel::CalcAttribute(current_cell_vertices_idx, current_cell_vertex_weight,
                    MAX_VERTEX_NUM, current_cell_vertices_number, actual_ztop_layer, local_layer_front - 1, af.data());
                double attr_front = t_front * attr_up_front + (1.0 - t_front) * attr_dn_front;

                double attr_dn_back = TBBKernel::CalcAttribute(current_cell_vertices_idx, current_cell_vertex_weight,
                    MAX_VERTEX_NUM, current_cell_vertices_number, actual_ztop_layer, local_layer_back, ab.data());
                double attr_up_back = TBBKernel::CalcAttribute(current_cell_vertices_idx, current_cell_vertex_weight,
                    MAX_VERTEX_NUM, current_cell_vertices_number, actual_ztop_layer, local_layer_back - 1, ab.data());
                double attr_back = t_back * attr_up_back + (1.0 - t_back) * attr_dn_back;

                current_attr.x() = alpha * attr_back + (1.0 - alpha) * attr_front;
            }
            if (attr_count >= 2) {
                const auto& af = *attr_front_ptrs[1];
                const auto& ab = *attr_back_ptrs[1];
                double attr_dn_front = TBBKernel::CalcAttribute(current_cell_vertices_idx, current_cell_vertex_weight,
                    MAX_VERTEX_NUM, current_cell_vertices_number, actual_ztop_layer, local_layer_front, af.data());
                double attr_up_front = TBBKernel::CalcAttribute(current_cell_vertices_idx, current_cell_vertex_weight,
                    MAX_VERTEX_NUM, current_cell_vertices_number, actual_ztop_layer, local_layer_front - 1, af.data());
                double attr_front = t_front * attr_up_front + (1.0 - t_front) * attr_dn_front;

                double attr_dn_back = TBBKernel::CalcAttribute(current_cell_vertices_idx, current_cell_vertex_weight,
                    MAX_VERTEX_NUM, current_cell_vertices_number, actual_ztop_layer, local_layer_back, ab.data());
                double attr_up_back = TBBKernel::CalcAttribute(current_cell_vertices_idx, current_cell_vertex_weight,
                    MAX_VERTEX_NUM, current_cell_vertices_number, actual_ztop_layer, local_layer_back - 1, ab.data());
                double attr_back = t_back * attr_up_back + (1.0 - t_back) * attr_dn_back;

                current_attr.y() = alpha * attr_back + (1.0 - alpha) * attr_front;
            }
        }

        return {current_horizontal_velocity, current_vertical_velocity, current_attr, true};
    };

    tbb::parallel_for(size_t(0), stable_points.size(), [&](size_t pid) {
        constexpr int MAX_CELL_NEIGHBOR_NUM = 21;
        int run_time = 0;
        bool first_loop = true;
        bool first_vel = true;
        bool first_attr = true;

        const int base_idx = static_cast<int>(pid) * each_points_size;
        int update_points_idx = 0;
        int cell_id = -1;
        int cell_neig_vec[MAX_CELL_NEIGHBOR_NUM];
        for (int& v : cell_neig_vec) {
            v = -1;
        }

        for (int step_i = 0; step_i < n_steps; ++step_i) {
            const double alpha = static_cast<double>(step_i) / static_cast<double>(n_steps);
            run_time += delta_t;

            vec3 sample_point_position = stable_points[pid];
            const double current_depth = -1.0 * static_cast<double>(effective_depths[pid]);

            if (first_loop) {
                first_loop = false;
                cell_id = default_cell_id[pid];
                if (cell_id < 0 || cell_id >= actual_cell_size) {
                    return;
                }
                int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
                TBBKernel::GetCellNeighborsIdx(cell_id, current_cell_vertices_number, cell_neig_vec,
                                               MAX_CELL_NEIGHBOR_NUM, actual_max_edge_size, cells_on_cell);
                update_points[base_idx] = sample_point_position;
            } else {
                if (cell_id < 0 || cell_id >= actual_cell_size) {
                    return;
                }
                int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
                double min_len = std::numeric_limits<double>::max();
                for (int n = 0; n < current_cell_vertices_number + 1; ++n) {
                    int cid = cell_neig_vec[n];
                    if (cid < 0 || cid >= actual_cell_size) {
                        continue;
                    }
                    const double len = MOPS_LENGTH(cell_coord[cid] - sample_point_position);
                    if (len < min_len) {
                        min_len = len;
                        cell_id = cid;
                    }
                }
                current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
                TBBKernel::GetCellNeighborsIdx(cell_id, current_cell_vertices_number, cell_neig_vec,
                                               MAX_CELL_NEIGHBOR_NUM, actual_max_edge_size, cells_on_cell);
            }

            const vec3 current_position = sample_point_position;
            const double r = MOPS_LENGTH(current_position);
            vec3 rk4_next_position = current_position;

            vec3 current_horizontal_velocity = vec3(0.0, 0.0, 0.0);
            double current_vertical_velocity = 0.0;
            vec3 current_attrs = vec3(0.0, 0.0, 0.0);

            if (use_euler) {
                auto state = calc_velocity_at(current_position, cell_id, current_depth, alpha);
                if (!state.ok) {
                    return;
                }
                current_horizontal_velocity = state.h_vel;
                current_vertical_velocity = state.v_vel;
                current_attrs = state.attr;
            } else {
                const double dt = static_cast<double>(delta_t);
                const double dalpha = dt / static_cast<double>(config->simulationDuration);

                double a1 = alpha;
                auto s1 = calc_velocity_at(current_position, cell_id, current_depth, a1);
                if (!s1.ok) {
                    return;
                }

                vec3 p2 = advect_on_sphere(current_position, s1.h_vel, dt * 0.5);
                double a2 = std::clamp(a1 + 0.5 * dalpha, 0.0, 1.0);
                auto s2 = calc_velocity_at(p2, cell_id, current_depth, a2);
                if (!s2.ok) {
                    return;
                }

                vec3 p3 = advect_on_sphere(current_position, s2.h_vel, dt * 0.5);
                double a3 = std::clamp(a1 + 0.5 * dalpha, 0.0, 1.0);
                auto s3 = calc_velocity_at(p3, cell_id, current_depth, a3);
                if (!s3.ok) {
                    return;
                }

                vec3 p4 = advect_on_sphere(current_position, s3.h_vel, dt);
                double a4 = std::clamp(a1 + dalpha, 0.0, 1.0);
                auto s4 = calc_velocity_at(p4, cell_id, current_depth, a4);
                if (!s4.ok) {
                    return;
                }

                current_horizontal_velocity = (s1.h_vel + 2.0 * s2.h_vel + 2.0 * s3.h_vel + s4.h_vel) / 6.0;
                current_attrs = (s1.attr + 2.0 * s2.attr + 2.0 * s3.attr + s4.attr) / 6.0;
                current_vertical_velocity = (s1.v_vel + 2.0 * s2.v_vel + 2.0 * s3.v_vel + s4.v_vel) / 6.0;

                vec3 x_trial = current_position + current_horizontal_velocity * dt;
                double x_trial_len = MOPS_LENGTH(x_trial);
                rk4_next_position = (x_trial_len > 1e-12) ? (x_trial / x_trial_len) * r : current_position;
            }

            vec3 new_position;
            if (use_euler) {
                vec3 rotation_axis = TBBKernel::CalcRotationAxis(current_position, current_horizontal_velocity);
                double speed = MOPS_LENGTH(current_horizontal_velocity);
                double theta_rad = (speed * delta_t) / std::max(1e-12, r);
                new_position = TBBKernel::CalcPositionAfterRotation(current_position, rotation_axis, theta_rad);
            } else {
                new_position = rk4_next_position;
            }

            if (first_vel) {
                first_vel = false;
                update_vels[base_idx] = current_horizontal_velocity;
            }
            if (first_attr && b_double_attributes) {
                first_attr = false;
                update_attrs[base_idx] = current_attrs;
            }

            const double old_depth = static_cast<double>(effective_depths[pid]);
            double new_depth = old_depth - current_vertical_velocity * static_cast<double>(delta_t);
            new_depth = std::max(0.0, new_depth);
            double r_new = std::max(1.0, r + current_vertical_velocity * static_cast<double>(delta_t));
            effective_depths[pid] = static_cast<float>(new_depth);

            const double nlen = MOPS_LENGTH(new_position);
            if (nlen > 1e-12) {
                new_position = (new_position / nlen) * r_new;
            }
            stable_points[pid] = new_position;

            const int record_interval = static_cast<int>(config->recordT / config->deltaT);
            if (record_interval > 0 && ((step_i + 1) % record_interval) == 0) {
                int write_idx = base_idx + update_points_idx;
                if (write_idx >= base_idx && write_idx < base_idx + each_points_size) {
                    update_points[write_idx] = new_position;
                    update_vels[write_idx] = current_horizontal_velocity;
                    if (b_double_attributes) {
                        update_attrs[write_idx] = current_attrs;
                    }
                }
                ++update_points_idx;
            }
        }
    });

    const size_t each_point_size = config->simulationDuration / config->recordT;
    const size_t total_points = update_points.size();
    auto clean_traj = MOPS::Common::FinalizeTrajectoryLinesWithAttrs(
        trajectory_lines,
        update_points,
        update_vels,
        update_attrs,
        each_point_size,
        total_points);
    trajectory_lines.clear();
    return clean_traj;
}

} // namespace MOPS::CPU::TBBBackend::Kernel
