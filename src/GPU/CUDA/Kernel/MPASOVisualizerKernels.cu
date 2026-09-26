#include "GPU/CUDA/Kernel/MPASOVisualizerKernels.h"

#include <cuda_runtime.h>

#include "GPU/CUDA/Kernel/CUDAKernel.h"
#include "GPU/CUDA/Kernel/VisualizerKernels_Common.cuh"  // Common device functions (Phase 1 refactoring)
#include "GPU/CUDA/Kernel/CUDAMemoryUtils.cuh"           // Unified CUDA memory utilities
#include "Common/ImageBuffer.hpp"
#include "Common/TrajectoryCommon.h"
#include "Utils/Interpolation.hpp"

// Forward declaration of PathLine kernel (defined in MPASOPathLineKernel.cu)
namespace MOPS {
namespace GPU {
namespace CUDABackend {
namespace Kernel {
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
        double masked_attr_value);

    // Forward declaration of StreamLine kernel (defined in MPASOStreamLineKernel.cu)
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
        const double* cell_vertex_vert_velocity);
}
}
}
}

namespace {

// Phase 1 Refactoring: Use common functions from VisualizerKernels_Common.cuh
using namespace MOPS::GPU::CUDABackend::VisualizerInternal;

// Use unified CUDA memory utilities
using namespace MOPS::GPU::CUDABackend::MemoryUtils;

MOPS_HOST_DEVICE inline int ClampInt(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void KernelVisualizeFixedDepth(
    int width,
    int height,
    double min_lat,
    double max_lat,
    double min_lon,
    double max_lon,
    double fixed_depth,
    const int* cell_id,
    int cell_size,
    int max_edge,
    int actual_vertex_size,
    const vec3* vertex_coord,
    const size_t* number_vertex_on_cell,
    const size_t* vertices_on_cell,
    const vec3* cell_vertex_velocity,
    int cell_vertex_velocity_size,
    const double* cell_vertex_ztop,
    int cell_vertex_ztop_size,
    const double* attr0,
    int attr0_size,
    const double* attr1,
    int attr1_size,
    bool has_attr,
    double* img0,
    double* img1)
{
    const int x = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int y = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }


    auto double_nan = nan("");
    vec3 vec3_nan = {double_nan, double_nan, double_nan};

    const int global_id = y * width + x;
    const int cell = cell_id[global_id];
    if (cell < 0 || cell >= cell_size) {
        MOPS::SetPixel(img0, width, height, y, x, vec3_nan);
        if (has_attr && img1 != nullptr) {
            MOPS::SetPixel(img1, width, height, y, x, vec3_nan);
        }
        return;
    }

    vec2 current_pixel = {static_cast<double>(y), static_cast<double>(x)};
    CartesianCoord current_position;
    SphericalCoord current_latlon_r;
    MOPS::GeoConverter::convertPixelToLatLonToRadians(width, height, min_lat, max_lat, min_lon, max_lon, current_pixel, current_latlon_r);
    MOPS::GeoConverter::convertRadianLatLonToXYZ(current_latlon_r, current_position);

    bool is_in_mesh = MOPS::CUDAKernel::IsInMesh(
        cell,
        max_edge,
        current_position,
        number_vertex_on_cell,
        vertices_on_cell,
        vertex_coord);
    if (!is_in_mesh) {
        MOPS::SetPixel(img0, width, height, y, x, vec3_nan);
        if (has_attr && img1 != nullptr) {
            MOPS::SetPixel(img1, width, height, y, x, vec3_nan);
        }
        return;
    }

    auto current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell]);
    if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
        MOPS::SetPixel(img0, width, height, y, x, vec3_nan);
        if (has_attr && img1 != nullptr) {
            MOPS::SetPixel(img1, width, height, y, x, vec3_nan);
        }
        return;
    }

    size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
    MOPS::CUDAKernel::GetCellVerticesIdx(
        cell,
        current_cell_vertices_number,
        current_cell_vertices_idx,
        MAX_VERTEX_NUM,
        max_edge,
        vertices_on_cell);

    vec3 vpos[MAX_VERTEX_NUM];
    double w[MAX_VERTEX_NUM];
    if (!MOPS::CUDAKernel::GetCellVertexPos(vpos, current_cell_vertices_idx, MAX_VERTEX_NUM, current_cell_vertices_number, vertex_coord)) {
        MOPS::SetPixel(img0, width, height, y, x, vec3_nan);
        if (has_attr && img1 != nullptr) {
            MOPS::SetPixel(img1, width, height, y, x, vec3_nan);
        }
        return;
    }

    for (int i = 0; i < MAX_VERTEX_NUM; ++i) {
        w[i] = 0.0;
    }
    MOPS::Interpolator::CalcPolygonWachspress(current_position, vpos, w, current_cell_vertices_number);

    if (actual_vertex_size <= 0 || cell_vertex_ztop_size <= 0 || (cell_vertex_ztop_size % actual_vertex_size) != 0) {
        MOPS::SetPixel(img0, width, height, y, x, vec3_nan);
        if (has_attr && img1 != nullptr) {
            MOPS::SetPixel(img1, width, height, y, x, vec3_nan);
        }
        return;
    }

    const int ztop_levels = cell_vertex_ztop_size / actual_vertex_size;
    if (ztop_levels <= 0 || ztop_levels > MAX_VERTICAL_LEVELS) {
        MOPS::SetPixel(img0, width, height, y, x, vec3_nan);
        if (has_attr && img1 != nullptr) {
            MOPS::SetPixel(img1, width, height, y, x, vec3_nan);
        }
        return;
    }

    double current_point_ztop_vec[MAX_ZTOP_LAYERS];
    for (int k = 0; k < ztop_levels; ++k) {
        double acc = 0.0;
        for (int v = 0; v < current_cell_vertices_number; ++v) {
            const int vid = static_cast<int>(current_cell_vertices_idx[v]);
            const double ztop = cell_vertex_ztop[vid * ztop_levels + k];
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

    double epsd = fmax(1e-6, 1e-8 * fabs(z_surf - z_bot));
    if (!(fixed_depth <= z_surf + epsd && fixed_depth >= z_bot - epsd)) {
        MOPS::SetPixel(img0, width, height, y, x, vec3_nan);
        if (has_attr && img1 != nullptr) {
            MOPS::SetPixel(img1, width, height, y, x, vec3_nan);
        }
        return;
    }

    int local_layer = -1;

    // Check if particle is above surface (shallower than layer 0)
    // Note: ztop is negative, so "above" means greater value
    if (fixed_depth >= current_point_ztop_vec[0]) {
        local_layer = 0;
    } else {
        // Search for the layer containing fixed_depth
        for (int k = 1; k < ztop_levels; ++k) {
            double top_i = current_point_ztop_vec[k - 1];
            double bot_i = current_point_ztop_vec[k];

            // Check if this layer is masked (zero thickness)
            double layer_thickness = top_i - bot_i;
            if (layer_thickness < 1e-6) {
                // Skip masked layers
                continue;
            }

            if (top_i < bot_i) {
                double t = top_i;
                top_i = bot_i;
                bot_i = t;
            }
            if (fixed_depth <= top_i + 1e-8 && fixed_depth >= bot_i - 1e-8) {
                local_layer = k;
                break;
            }
        }
    }

    if (local_layer < 0) {
        MOPS::SetPixel(img0, width, height, y, x, vec3_nan);
        if (has_attr && img1 != nullptr) {
            MOPS::SetPixel(img1, width, height, y, x, vec3_nan);
        }
        return;
    }

    // CRITICAL: Check if lower layer is masked (seafloor), move up one layer
    {
        double ztop_dn = current_point_ztop_vec[local_layer];
        double ztop_up = current_point_ztop_vec[local_layer - 1];
        double layer_thickness = ztop_up - ztop_dn;

        if (fabs(layer_thickness) < 1e-6 && local_layer > 1) {
            // Lower layer is masked (seafloor) - move particle up one layer
            local_layer = local_layer - 1;
        }
    }

    

    double top_i = current_point_ztop_vec[ClampInt(local_layer - 1, 0, ztop_levels - 1)];
    double bot_i = current_point_ztop_vec[ClampInt(local_layer, 0, ztop_levels - 1)];
    double denom = top_i - bot_i;
    double tparam = (denom > 1e-12) ? (fixed_depth - bot_i) / denom : 0.5;

    if (actual_vertex_size <= 0 || cell_vertex_velocity_size <= 0 || (cell_vertex_velocity_size % actual_vertex_size) != 0) {
        MOPS::SetPixel(img0, width, height, y, x, vec3_nan);
        if (has_attr && img1 != nullptr) {
            MOPS::SetPixel(img1, width, height, y, x, vec3_nan);
        }
        return;
    }

    const int vel_levels = cell_vertex_velocity_size / actual_vertex_size;
    int j = ClampInt(local_layer - 1, 0, vel_levels - 1);
    int j_bot = ClampInt(j + 1, 0, vel_levels - 1);
    int j_top = j;

    vec3 v_top = MOPS::CUDAKernel::CalcVelocity(
        current_cell_vertices_idx,
        w,
        MAX_VERTEX_NUM,
        current_cell_vertices_number,
        vel_levels,
        j_top,
        cell_vertex_velocity);
    vec3 v_bot = MOPS::CUDAKernel::CalcVelocity(
        current_cell_vertices_idx,
        w,
        MAX_VERTEX_NUM,
        current_cell_vertices_number,
        vel_levels,
        j_bot,
        cell_vertex_velocity);

    double mtop = MOPS_LENGTH(v_top);
    double mbot = MOPS_LENGTH(v_bot);
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

    double u_east = 0.0;
    double v_north = 0.0;
    MOPS::GeoConverter::convertXYZVelocityToENU(current_position, final_vel, u_east, v_north);
    double spd = sqrt(u_east * u_east + v_north * v_north);
    vec3 current_point_velocity_enu = {u_east, v_north, spd};
    MOPS::SetPixel(img0, width, height, y, x, current_point_velocity_enu);

    if (has_attr && img1 != nullptr && attr0 != nullptr && attr0_size > 0 && (attr0_size % actual_vertex_size) == 0) {
        int attr_levels0 = attr0_size / actual_vertex_size;
        int aj = ClampInt(local_layer - 1, 0, attr_levels0 - 1);
        double a0 = MOPS::CUDAKernel::CalcAttribute(
            current_cell_vertices_idx,
            w,
            MAX_VERTEX_NUM,
            current_cell_vertices_number,
            attr_levels0,
            aj,
            attr0);

        double a1 = 0.0;
        if (attr1 != nullptr && attr1_size > 0 && (attr1_size % actual_vertex_size) == 0) {
            int attr_levels1 = attr1_size / actual_vertex_size;
            int aj1 = ClampInt(local_layer - 1, 0, attr_levels1 - 1);
            a1 = MOPS::CUDAKernel::CalcAttribute(
                current_cell_vertices_idx,
                w,
                MAX_VERTEX_NUM,
                current_cell_vertices_number,
                attr_levels1,
                aj1,
                attr1);
        }

        vec3 current_point_attr_value = {a0, a1, 0.0};
        MOPS::SetPixel(img1, width, height, y, x, current_point_attr_value);
    }
}

__global__ void KernelVisualizeFixedLayer(
    int width,
    int height,
    double min_lat,
    double max_lat,
    double min_lon,
    double max_lon,
    int fixed_layer,
    const int* cell_id,
    int cell_size,
    int max_edge,
    const vec3* vertex_coord,
    const size_t* number_vertex_on_cell,
    const size_t* vertices_on_cell,
    const vec3* cell_vertex_velocity,
    int actual_vertex_size,
    int cell_vertex_velocity_size,
    double* img)
{
    const int x = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int y = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }


    auto double_nan = nan("");
    vec3 vec3_nan = {double_nan, double_nan, double_nan};

    const int global_id = y * width + x;
    const int cell = cell_id[global_id];
    if (cell < 0 || cell >= cell_size) {
        MOPS::SetPixel(img, width, height, y, x, vec3_nan);
        return;
    }

    if (actual_vertex_size <= 0 || cell_vertex_velocity_size <= 0 || (cell_vertex_velocity_size % actual_vertex_size) != 0) {
        MOPS::SetPixel(img, width, height, y, x, vec3_nan);
        return;
    }

    const int vel_levels = cell_vertex_velocity_size / actual_vertex_size;
    if (fixed_layer < 0 || fixed_layer >= vel_levels) {
        MOPS::SetPixel(img, width, height, y, x, vec3_nan);
        return;
    }

    vec2 current_pixel = {static_cast<double>(y), static_cast<double>(x)};
    CartesianCoord current_position;
    SphericalCoord current_latlon_r;
    MOPS::GeoConverter::convertPixelToLatLonToRadians(width, height, min_lat, max_lat, min_lon, max_lon, current_pixel, current_latlon_r);
    MOPS::GeoConverter::convertRadianLatLonToXYZ(current_latlon_r, current_position);

    bool is_in_mesh = MOPS::CUDAKernel::IsInMesh(
        cell,
        max_edge,
        current_position,
        number_vertex_on_cell,
        vertices_on_cell,
        vertex_coord);
    if (!is_in_mesh) {
        MOPS::SetPixel(img, width, height, y, x, vec3_nan);
        return;
    }

    const int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell]);
    if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
        MOPS::SetPixel(img, width, height, y, x, vec3_nan);
        return;
    }

    size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
    MOPS::CUDAKernel::GetCellVerticesIdx(
        cell,
        current_cell_vertices_number,
        current_cell_vertices_idx,
        MAX_VERTEX_NUM,
        max_edge,
        vertices_on_cell);

    vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
    double current_cell_vertex_weight[MAX_VERTEX_NUM];
    if (!MOPS::CUDAKernel::GetCellVertexPos(
            current_cell_vertex_pos,
            current_cell_vertices_idx,
            MAX_VERTEX_NUM,
            current_cell_vertices_number,
            vertex_coord)) {
        MOPS::SetPixel(img, width, height, y, x, vec3_nan);
        return;
    }

    for (int i = 0; i < MAX_VERTEX_NUM; ++i) {
        current_cell_vertex_weight[i] = 0.0;
    }
    MOPS::Interpolator::CalcPolygonWachspress(
        current_position,
        current_cell_vertex_pos,
        current_cell_vertex_weight,
        current_cell_vertices_number);

    vec3 current_point_velocity = MOPS::CUDAKernel::CalcVelocity(
        current_cell_vertices_idx,
        current_cell_vertex_weight,
        MAX_VERTEX_NUM,
        current_cell_vertices_number,
        vel_levels,
        fixed_layer,
        cell_vertex_velocity);

    double zional_velocity = 0.0;
    double merminoal_velicity = 0.0;
    MOPS::GeoConverter::convertXYZVelocityToENU(
        current_position,
        current_point_velocity,
        zional_velocity,
        merminoal_velicity);

    vec3 current_point_velocity_enu = {zional_velocity, merminoal_velicity, 0.0};
    MOPS::SetPixel(img, width, height, y, x, current_point_velocity_enu);
}

} // namespace

namespace MOPS::GPU::CUDABackend::Kernel {

void VisualizeFixedLayer(
    MPASOField* mpasoF,
    VisualizationSettings* config,
    ImageBuffer<double>* img)
{
    if (mpasoF == nullptr || config == nullptr || img == nullptr) {
        Error("[CUDABackend::Kernel]::VisualizeFixedLayer invalid input");
    }
    if (mpasoF->mGrid == nullptr || mpasoF->mSol_Front == nullptr) {
        Error("[CUDABackend::Kernel]::VisualizeFixedLayer field is not initialized");
    }

    const int width = static_cast<int>(config->imageSize.x());
    const int height = static_cast<int>(config->imageSize.y());
    const double min_lat = config->LatRange.x();
    const double max_lat = config->LatRange.y();
    const double min_lon = config->LonRange.x();
    const double max_lon = config->LonRange.y();
    const int fixed_layer = config->FixedLayer;

    std::vector<int> cell_id_vec(width * height, -1);
    MOPS::CUDAKernel::SearchKDTree(cell_id_vec.data(), mpasoF->mGrid.get(), width, height, min_lat, max_lat, min_lon, max_lon);

    int* d_cell_id = DeviceAllocAndCopy(cell_id_vec, "cudaMalloc/cudaMemcpy cellID");
    vec3* d_vertex_coord = DeviceAllocAndCopy(mpasoF->mGrid->vertexCoord_vec, "cudaMalloc/cudaMemcpy vertexCoord");
    size_t* d_number_vertex_on_cell = DeviceAllocAndCopy(mpasoF->mGrid->numberVertexOnCell_vec, "cudaMalloc/cudaMemcpy numberVertexOnCell");
    size_t* d_vertices_on_cell = DeviceAllocAndCopy(mpasoF->mGrid->verticesOnCell_vec, "cudaMalloc/cudaMemcpy verticesOnCell");
    vec3* d_cell_vertex_velocity = DeviceAllocAndCopy(mpasoF->mSol_Front->cellVertexVelocity_vec, "cudaMalloc/cudaMemcpy cellVertexVelocity");

    double* d_img = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_img), img->mPixels.size() * sizeof(double)), "cudaMalloc img");

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    KernelVisualizeFixedLayer<<<grid, block>>>(
        width,
        height,
        min_lat,
        max_lat,
        min_lon,
        max_lon,
        fixed_layer,
        d_cell_id,
        mpasoF->mGrid->mCellsSize,
        mpasoF->mGrid->mMaxEdgesSize,
        d_vertex_coord,
        d_number_vertex_on_cell,
        d_vertices_on_cell,
        d_cell_vertex_velocity,
        mpasoF->mGrid->mVertexSize,
        static_cast<int>(mpasoF->mSol_Front->cellVertexVelocity_vec.size()),
        d_img);

    CheckCuda(cudaGetLastError(), "KernelVisualizeFixedLayer launch");
    CheckCuda(cudaDeviceSynchronize(), "KernelVisualizeFixedLayer sync");

    CheckCuda(
        cudaMemcpy(
            img->mPixels.data(),
            d_img,
            img->mPixels.size() * sizeof(double),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy img");

    FreeDev(d_img, "cudaFree img");
    FreeDev(d_cell_vertex_velocity, "cudaFree cellVertexVelocity");
    FreeDev(d_vertices_on_cell, "cudaFree verticesOnCell");
    FreeDev(d_number_vertex_on_cell, "cudaFree numberVertexOnCell");
    FreeDev(d_vertex_coord, "cudaFree vertexCoord");
    FreeDev(d_cell_id, "cudaFree cellID");
}

void VisualizeFixedDepth(
    MPASOField* mpasoF,
    VisualizationSettings* config,
    std::vector<ImageBuffer<double>>& img_vec)
{
    if (mpasoF == nullptr || config == nullptr || img_vec.empty()) {
        Error("[CUDABackend::Kernel]::VisualizeFixedDepth invalid input");
    }
    if (mpasoF->mGrid == nullptr || mpasoF->mSol_Front == nullptr) {
        Error("[CUDABackend::Kernel]::VisualizeFixedDepth field is not initialized");
    }

    const int width = static_cast<int>(config->imageSize.x());
    const int height = static_cast<int>(config->imageSize.y());
    const double min_lat = config->LatRange.x();
    const double max_lat = config->LatRange.y();
    const double min_lon = config->LonRange.x();
    const double max_lon = config->LonRange.y();
    const double fixed_depth = -config->FixedDepth;

    std::vector<int> cell_id_vec(width * height, -1);
    MOPS::CUDAKernel::SearchKDTree(cell_id_vec.data(), mpasoF->mGrid.get(), width, height, min_lat, max_lat, min_lon, max_lon);

    // IMPORTANT: Collect in specific order (temperature, salinity)
    std::vector<const std::vector<double>*> attr_ptrs;
    auto temp_it = mpasoF->mSol_Front->mDoubleAttributes_CtoV.find("temperature");
    auto sal_it = mpasoF->mSol_Front->mDoubleAttributes_CtoV.find("salinity");
    if (temp_it != mpasoF->mSol_Front->mDoubleAttributes_CtoV.end()) {
        attr_ptrs.push_back(&temp_it->second);
    }
    if (sal_it != mpasoF->mSol_Front->mDoubleAttributes_CtoV.end()) {
        attr_ptrs.push_back(&sal_it->second);
    }

    const bool has_attr = (!attr_ptrs.empty()) && (img_vec.size() > 1);

    int* d_cell_id = DeviceAllocAndCopy(cell_id_vec, "cudaMalloc/cudaMemcpy cellID");
    vec3* d_vertex_coord = DeviceAllocAndCopy(mpasoF->mGrid->vertexCoord_vec, "cudaMalloc/cudaMemcpy vertexCoord");
    size_t* d_number_vertex_on_cell = DeviceAllocAndCopy(mpasoF->mGrid->numberVertexOnCell_vec, "cudaMalloc/cudaMemcpy numberVertexOnCell");
    size_t* d_vertices_on_cell = DeviceAllocAndCopy(mpasoF->mGrid->verticesOnCell_vec, "cudaMalloc/cudaMemcpy verticesOnCell");
    vec3* d_cell_vertex_velocity = DeviceAllocAndCopy(mpasoF->mSol_Front->cellVertexVelocity_vec, "cudaMalloc/cudaMemcpy cellVertexVelocity");
    double* d_cell_vertex_ztop = DeviceAllocAndCopy(mpasoF->mSol_Front->cellVertexZTop_vec, "cudaMalloc/cudaMemcpy cellVertexZTop");

    double* d_attr0 = nullptr;
    double* d_attr1 = nullptr;
    int attr0_size = 0;
    int attr1_size = 0;
    if (!attr_ptrs.empty()) {
        d_attr0 = DeviceAllocAndCopy(*attr_ptrs[0], "cudaMalloc/cudaMemcpy attr0");
        attr0_size = static_cast<int>(attr_ptrs[0]->size());
    }
    if (attr_ptrs.size() > 1) {
        d_attr1 = DeviceAllocAndCopy(*attr_ptrs[1], "cudaMalloc/cudaMemcpy attr1");
        attr1_size = static_cast<int>(attr_ptrs[1]->size());
    }

    double* d_img0 = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_img0), img_vec[0].mPixels.size() * sizeof(double)), "cudaMalloc img0");

    double* d_img1 = nullptr;
    if (has_attr) {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&d_img1), img_vec[1].mPixels.size() * sizeof(double)), "cudaMalloc img1");
    }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    KernelVisualizeFixedDepth<<<grid, block>>>(
        width,
        height,
        min_lat,
        max_lat,
        min_lon,
        max_lon,
        fixed_depth,
        d_cell_id,
        mpasoF->mGrid->mCellsSize,
        mpasoF->mGrid->mMaxEdgesSize,
        mpasoF->mGrid->mVertexSize,
        d_vertex_coord,
        d_number_vertex_on_cell,
        d_vertices_on_cell,
        d_cell_vertex_velocity,
        static_cast<int>(mpasoF->mSol_Front->cellVertexVelocity_vec.size()),
        d_cell_vertex_ztop,
        static_cast<int>(mpasoF->mSol_Front->cellVertexZTop_vec.size()),
        d_attr0,
        attr0_size,
        d_attr1,
        attr1_size,
        has_attr,
        d_img0,
        d_img1);

    CheckCuda(cudaGetLastError(), "KernelVisualizeFixedDepth launch");
    CheckCuda(cudaDeviceSynchronize(), "KernelVisualizeFixedDepth sync");

    CheckCuda(
        cudaMemcpy(
            img_vec[0].mPixels.data(),
            d_img0,
            img_vec[0].mPixels.size() * sizeof(double),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy img0");

    if (has_attr) {
        CheckCuda(
            cudaMemcpy(
                img_vec[1].mPixels.data(),
                d_img1,
                img_vec[1].mPixels.size() * sizeof(double),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy img1");
    }

    FreeDev(d_img1, "cudaFree img1");
    FreeDev(d_img0, "cudaFree img0");
    FreeDev(d_attr1, "cudaFree attr1");
    FreeDev(d_attr0, "cudaFree attr0");
    FreeDev(d_cell_vertex_ztop, "cudaFree cellVertexZTop");
    FreeDev(d_cell_vertex_velocity, "cudaFree cellVertexVelocity");
    FreeDev(d_vertices_on_cell, "cudaFree verticesOnCell");
    FreeDev(d_number_vertex_on_cell, "cudaFree numberVertexOnCell");
    FreeDev(d_vertex_coord, "cudaFree vertexCoord");
    FreeDev(d_cell_id, "cudaFree cellID");
}

void VisualizeFixedLatitude(
    MPASOField* mpasoF,
    VisualizationSettings* config,
    ImageBuffer<double>* img)
{
    if (mpasoF == nullptr || config == nullptr || img == nullptr) {
        Error("[CUDABackend::Kernel]::VisualizeFixedLatitude invalid input");
    }
    if (mpasoF->mGrid == nullptr || mpasoF->mSol_Front == nullptr) {
        Error("[CUDABackend::Kernel]::VisualizeFixedLatitude field is not initialized");
    }

    const int width = static_cast<int>(config->imageSize.x());
    const int height = static_cast<int>(config->imageSize.y());
    const double min_lon = config->LonRange.x();
    const double max_lon = config->LonRange.y();
    const double fixed_lat = config->FixedLatitude;

    const auto& ref_bottom_depth = mpasoF->mGrid->cellRefBottomDepth_vec;
    const double min_depth = ref_bottom_depth.front();
    const double max_depth = ref_bottom_depth.back();

    std::vector<int> cell_id_vec(width * height, -1);
    const double i_step = (height > 1) ? (max_depth - min_depth) / (height - 1) : 0.0;
    const double j_step = (width > 1) ? (max_lon - min_lon) / (width - 1) : 0.0;
    for (int ih = 0; ih < height; ++ih) {
        for (int jw = 0; jw < width; ++jw) {
            const double lon = min_lon + jw * j_step;

            SphericalCoord latlon_r = {fixed_lat * (M_PI / 180.0), lon * (M_PI / 180.0)};
            CartesianCoord current_position;
            GeoConverter::convertRadianLatLonToXYZ(latlon_r, current_position);

            int cell_id_value = -1;
            mpasoF->mGrid->searchKDT(current_position, cell_id_value);
            cell_id_vec[ih * width + jw] = cell_id_value;
        }
    }

    auto double_nan = nan("");
    vec3 vec3_nan = {double_nan, double_nan, double_nan};

    const int actual_vertex_size = mpasoF->mGrid->mVertexSize;
    const int ztop_size = static_cast<int>(mpasoF->mSol_Front->cellVertexZTop_vec.size());
    const int vel_size = static_cast<int>(mpasoF->mSol_Front->cellVertexVelocity_vec.size());
    if (actual_vertex_size <= 0 || ztop_size <= 0 || vel_size <= 0) {
        Error("[CUDABackend::Kernel]::VisualizeFixedLatitude invalid mesh/solution array sizes");
    }

    int n_vert = ztop_size / actual_vertex_size;
    if (n_vert <= 0 || n_vert > MAX_VERTICAL_LEVELS || (vel_size / actual_vertex_size) < n_vert) {
        Error("[CUDABackend::Kernel]::VisualizeFixedLatitude invalid vertical levels");
    }

    for (int ih = 0; ih < height; ++ih) {
        const double depth_positive = min_depth + ih * i_step;
        const double depth = -fabs(depth_positive);

        for (int jw = 0; jw < width; ++jw) {
            const double lon = min_lon + jw * j_step;
            SphericalCoord latlon_r = {fixed_lat * (M_PI / 180.0), lon * (M_PI / 180.0)};
            CartesianCoord position;
            GeoConverter::convertRadianLatLonToXYZ(latlon_r, position);

            const int cell_id = cell_id_vec[ih * width + jw];
            if (cell_id < 0 || cell_id >= mpasoF->mGrid->mCellsSize) {
                img->setPixel(ih, jw, vec3_nan);
                continue;
            }

            bool is_in_mesh = MOPS::CUDAKernel::IsInMesh(
                cell_id,
                mpasoF->mGrid->mMaxEdgesSize,
                position,
                mpasoF->mGrid->numberVertexOnCell_vec.data(),
                mpasoF->mGrid->verticesOnCell_vec.data(),
                mpasoF->mGrid->vertexCoord_vec.data());
            if (!is_in_mesh) {
                img->setPixel(ih, jw, vec3_nan);
                continue;
            }

            const int current_cell_vertices_number = static_cast<int>(mpasoF->mGrid->numberVertexOnCell_vec[cell_id]);
            if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
                img->setPixel(ih, jw, vec3_nan);
                continue;
            }

            size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
            MOPS::CUDAKernel::GetCellVerticesIdx(
                cell_id,
                current_cell_vertices_number,
                current_cell_vertices_idx,
                MAX_VERTEX_NUM,
                mpasoF->mGrid->mMaxEdgesSize,
                mpasoF->mGrid->verticesOnCell_vec.data());

            vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
            double current_cell_vertex_weight[MAX_VERTEX_NUM];
            if (!MOPS::CUDAKernel::GetCellVertexPos(
                    current_cell_vertex_pos,
                    current_cell_vertices_idx,
                    MAX_VERTEX_NUM,
                    current_cell_vertices_number,
                    mpasoF->mGrid->vertexCoord_vec.data())) {
                img->setPixel(ih, jw, vec3_nan);
                continue;
            }

            for (int i = 0; i < MAX_VERTEX_NUM; ++i) {
                current_cell_vertex_weight[i] = 0.0;
            }
            Interpolator::CalcPolygonWachspress(
                position,
                current_cell_vertex_pos,
                current_cell_vertex_weight,
                current_cell_vertices_number);

            double current_point_ztop_vec[MAX_ZTOP_LAYERS];
            for (int k = 0; k < n_vert; ++k) {
                double current_point_ztop_in_layer = 0.0;
                for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
                    auto vid = current_cell_vertices_idx[v_idx];
                    double ztop = mpasoF->mSol_Front->cellVertexZTop_vec[vid * n_vert + k];
                    current_point_ztop_in_layer += current_cell_vertex_weight[v_idx] * ztop;
                }
                current_point_ztop_vec[k] = current_point_ztop_in_layer;
            }

            int layer = -1;
            const double epsilon = 1e-6;
            if (depth > current_point_ztop_vec[0] + epsilon || depth < current_point_ztop_vec[n_vert - 1] - epsilon) {
                img->setPixel(ih, jw, vec3_nan);
                continue;
            }

            for (int k = 1; k < n_vert; ++k) {
                double z_up = current_point_ztop_vec[k - 1];
                double z_dn = current_point_ztop_vec[k];
                if (depth <= z_up + epsilon && depth >= z_dn - epsilon) {
                    layer = k;
                    break;
                }
            }

            if (layer == -1) {
                img->setPixel(ih, jw, vec3_nan);
                continue;
            }

            double ztop_layer_dn = current_point_ztop_vec[layer];
            double ztop_layer_up = current_point_ztop_vec[layer - 1];
            double denom = (ztop_layer_dn - ztop_layer_up);
            if (fabs(denom) < 1e-30) {
                img->setPixel(ih, jw, vec3_nan);
                continue;
            }
            double t = (depth - ztop_layer_up) / denom;

            vec3 current_point_vel_up = MOPS::CUDAKernel::CalcVelocity(
                current_cell_vertices_idx,
                current_cell_vertex_weight,
                MAX_VERTEX_NUM,
                current_cell_vertices_number,
                n_vert,
                layer - 1,
                mpasoF->mSol_Front->cellVertexVelocity_vec.data());
            vec3 current_point_vel_dn = MOPS::CUDAKernel::CalcVelocity(
                current_cell_vertices_idx,
                current_cell_vertex_weight,
                MAX_VERTEX_NUM,
                current_cell_vertices_number,
                n_vert,
                layer,
                mpasoF->mSol_Front->cellVertexVelocity_vec.data());

            vec3 final_vel;
            final_vel.x() = (1.0 - t) * current_point_vel_up.x() + t * current_point_vel_dn.x();
            final_vel.y() = (1.0 - t) * current_point_vel_up.y() + t * current_point_vel_dn.y();
            final_vel.z() = (1.0 - t) * current_point_vel_up.z() + t * current_point_vel_dn.z();

            double zional_velocity = 0.0;
            double merminoal_velicity = 0.0;
            GeoConverter::convertXYZVelocityToENU(position, final_vel, zional_velocity, merminoal_velicity);
            vec3 current_point_velocity_enu = {zional_velocity, merminoal_velicity, 0.0};

            img->setPixel(ih, jw, current_point_velocity_enu);
        }
    }
}


MOPS_DEVICE inline vec3 AdvectOnSphereCUDA(const vec3& pos, const vec3& vel, double dt_local)
{
    const double rr = MOPS_LENGTH(pos);
    const double speed_local = MOPS_LENGTH(vel);
    if (rr < 1e-12 || speed_local < 1e-12) {
        return pos;
    }

    vec3 axis = MOPS::CUDAKernel::CalcRotationAxis(pos, vel);
    const double theta = (speed_local * dt_local) / rr;
    return MOPS::CUDAKernel::CalcPositionAfterRotation(pos, axis, theta);
}



std::vector<TrajectoryLine> StreamLine(
    MPASOField* mpasoF,
    std::vector<CartesianCoord>& points,
    TrajectorySettings* config,
    std::vector<int>& default_cell_id)
{
    if (mpasoF == nullptr || config == nullptr) {
        Error("[CUDABackend::Kernel]::StreamLine invalid input");
    }
    if (mpasoF->mGrid == nullptr || mpasoF->mSol_Front == nullptr) {
        Error("[CUDABackend::Kernel]::StreamLine field is not initialized");
    }

    if (points.empty()) {
        return {};
    }
    if (config->deltaT == 0 || config->recordT == 0 || config->simulationDuration == 0) {
        Error("[CUDABackend::Kernel]::StreamLine invalid trajectory settings");
    }

    std::vector<vec3> stable_points = points;

    auto host_buffers = MOPS::Common::InitTrajectoryOutputBuffers(
        stable_points.size(),
        static_cast<int>(config->simulationDuration),
        static_cast<int>(config->recordT),
        false);
    std::vector<vec3>& update_points = host_buffers.points;
    std::vector<vec3>& update_vels = host_buffers.velocities;

    std::vector<float> effective_depths = MOPS::Common::BuildEffectiveDepths(stable_points, config, "StreamLine");
    std::vector<TrajectoryLine> trajectory_lines = MOPS::Common::InitTrajectoryLines(stable_points, effective_depths, config);

    if (default_cell_id.size() != stable_points.size()) {
        default_cell_id.assign(stable_points.size(), -1);
        for (size_t i = 0; i < stable_points.size(); ++i) {
            int cell_id = -1;
            mpasoF->mGrid->searchKDT(stable_points[i], cell_id);
            default_cell_id[i] = cell_id;
        }
    }

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(mpasoF->mGrid.get());
    if (grid_info_vec.size() < 6) {
        Error("[CUDABackend::Kernel]::StreamLine grid_info is incomplete");
    }

    const int actual_cell_size = static_cast<int>(grid_info_vec[0]);
    const int actual_max_edge_size = static_cast<int>(grid_info_vec[2]);
    const int actual_vertex_size = static_cast<int>(grid_info_vec[3]);
    const int actual_ztop_layer = static_cast<int>(grid_info_vec[4]);
    const int actual_ztop_layer_p1 = static_cast<int>(grid_info_vec[5]);

    const int particle_count = static_cast<int>(stable_points.size());
    const int each_points_size = static_cast<int>(config->simulationDuration / config->recordT);
    const int times = static_cast<int>(config->simulationDuration / config->deltaT);
    const int dt_sign = (config->directionType == MOPS::CalcDirection::kForward) ? 1 : -1;
    const int delta_t = dt_sign * static_cast<int>(config->deltaT);
    const bool use_euler = (config->methodType == MOPS::CalcMethodType::kEuler);

    vec3* d_sample_points = DeviceAllocAndCopy(stable_points, "cudaMalloc/cudaMemcpy sample_points");
    vec3* d_write_points = DeviceAllocAndCopy(update_points, "cudaMalloc/cudaMemcpy write_points");
    vec3* d_write_vels = DeviceAllocAndCopy(update_vels, "cudaMalloc/cudaMemcpy write_vels");
    float* d_particle_depths = DeviceAllocAndCopy(effective_depths, "cudaMalloc/cudaMemcpy particle_depths");
    int* d_default_cell_id = DeviceAllocAndCopy(default_cell_id, "cudaMalloc/cudaMemcpy default_cell_id");

    vec3* d_vertex_coord = DeviceAllocAndCopy(mpasoF->mGrid->vertexCoord_vec, "cudaMalloc/cudaMemcpy vertexCoord");
    vec3* d_cell_coord = DeviceAllocAndCopy(mpasoF->mGrid->cellCoord_vec, "cudaMalloc/cudaMemcpy cellCoord");
    size_t* d_number_vertex_on_cell = DeviceAllocAndCopy(mpasoF->mGrid->numberVertexOnCell_vec, "cudaMalloc/cudaMemcpy numberVertexOnCell");
    size_t* d_vertices_on_cell = DeviceAllocAndCopy(mpasoF->mGrid->verticesOnCell_vec, "cudaMalloc/cudaMemcpy verticesOnCell");
    size_t* d_cells_on_cell = DeviceAllocAndCopy(mpasoF->mGrid->cellsOnCell_vec, "cudaMalloc/cudaMemcpy cellsOnCell");

    vec3* d_cell_vertex_velocity = DeviceAllocAndCopy(mpasoF->mSol_Front->cellVertexVelocity_vec, "cudaMalloc/cudaMemcpy cellVertexVelocity");
    double* d_cell_vertex_ztop = DeviceAllocAndCopy(mpasoF->mSol_Front->cellVertexZTop_vec, "cudaMalloc/cudaMemcpy cellVertexZTop");
    double* d_cell_vertex_vert_velocity = DeviceAllocAndCopy(mpasoF->mSol_Front->cellVertexVertVelocity_vec, "cudaMalloc/cudaMemcpy cellVertexVertVelocity");

    dim3 block(128);
    dim3 grid((particle_count + block.x - 1) / block.x);

    KernelStreamLine<<<grid, block>>>(
        particle_count,
        each_points_size,
        times,
        static_cast<int>(config->recordT),
        delta_t,
        use_euler,
        actual_cell_size,
        actual_max_edge_size,
        actual_vertex_size,
        actual_ztop_layer,
        actual_ztop_layer_p1,
        d_default_cell_id,
        d_sample_points,
        d_write_points,
        d_write_vels,
        d_particle_depths,
        d_vertex_coord,
        d_cell_coord,
        d_number_vertex_on_cell,
        d_vertices_on_cell,
        d_cells_on_cell,
        d_cell_vertex_velocity,
        d_cell_vertex_ztop,
        d_cell_vertex_vert_velocity);

    CheckCuda(cudaGetLastError(), "KernelStreamLine launch");
    CheckCuda(cudaDeviceSynchronize(), "KernelStreamLine sync");

    CheckCuda(
        cudaMemcpy(
            update_points.data(),
            d_write_points,
            update_points.size() * sizeof(vec3),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy streamline points");

    CheckCuda(
        cudaMemcpy(
            update_vels.data(),
            d_write_vels,
            update_vels.size() * sizeof(vec3),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy streamline velocities");

    FreeDev(d_cell_vertex_vert_velocity, "cudaFree cellVertexVertVelocity");
    FreeDev(d_cell_vertex_ztop, "cudaFree cellVertexZTop");
    FreeDev(d_cell_vertex_velocity, "cudaFree cellVertexVelocity");
    FreeDev(d_cells_on_cell, "cudaFree cellsOnCell");
    FreeDev(d_vertices_on_cell, "cudaFree verticesOnCell");
    FreeDev(d_number_vertex_on_cell, "cudaFree numberVertexOnCell");
    FreeDev(d_cell_coord, "cudaFree cellCoord");
    FreeDev(d_vertex_coord, "cudaFree vertexCoord");
    FreeDev(d_default_cell_id, "cudaFree defaultCellID");
    FreeDev(d_particle_depths, "cudaFree particleDepths");
    FreeDev(d_write_vels, "cudaFree writeVels");
    FreeDev(d_write_points, "cudaFree writePoints");
    FreeDev(d_sample_points, "cudaFree samplePoints");

    const size_t total_points = update_points.size();
    auto clean_traj = MOPS::Common::FinalizeTrajectoryLines(
        trajectory_lines,
        update_points,
        update_vels,
        static_cast<size_t>(each_points_size),
        total_points);

    trajectory_lines.clear();
    return clean_traj;
}


// Helper function: Calculate zTop at surface level (level 0) at given position
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
        return 0.0;
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

    // Interpolate zTop at level 0
    double ztop_surface = 0.0;
    for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) {
        int vid = static_cast<int>(current_cell_vertices_idx[v_idx]);
        if (vid < 0 || vid >= actual_vertex_size) {
            continue;
        }
        double ztop_at_vertex = cell_vertex_ztop[vid * actual_ztop_layer + 0];  // level 0
        ztop_surface += current_cell_vertex_weight[v_idx] * ztop_at_vertex;
    }

    return ztop_surface;
}

// Helper function: Calculate zTop at bottom level (seafloor) at given position
// Helper function to interpolate zTop at a specific layer
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

// Helper function: Project velocity onto edge tangent direction
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

// Helper function: Project position to match a given depth
// Keeps horizontal direction but adjusts radius to match depth
// This is used after surface/bottom boundary adjustments
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

// Helper function: Find which cell (current or neighbors) contains the position
// Returns cell_id if found, -1 if position is outside all cells (true boundary)
MOPS_DEVICE inline int FindContainingCellInCurrentOrNeighborsCUDA(
    const vec3& pos,
    int current_cell_id,
    int actual_max_edge_size,
    const size_t* number_vertex_on_cell,
    const size_t* cells_on_cell,
    const size_t* vertices_on_cell,
    const vec3* vertex_coord)
{

    // First check current cell (fast path)
    bool in_current = MOPS::CUDAKernel::IsInMesh(
        current_cell_id,
        actual_max_edge_size,
        pos,
        number_vertex_on_cell,
        vertices_on_cell,
        vertex_coord);

    if (in_current) {
        return current_cell_id;
    }

    // Not in current cell - check neighbor cells
    int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[current_cell_id]);
    if (current_cell_vertices_number <= 0 || current_cell_vertices_number > MAX_VERTEX_NUM) {
        return -1;
    }

    // Get neighbor cells (use int array to match GetCellNeighborsIdx signature)
    int cell_neig_vec[MAX_VERTEX_NUM + 1];
    for (int i = 0; i < MAX_VERTEX_NUM + 1; ++i) {
        cell_neig_vec[i] = -1;
    }
    MOPS::CUDAKernel::GetCellNeighborsIdx(
        current_cell_id,
        current_cell_vertices_number,
        cell_neig_vec,
        MAX_VERTEX_NUM,
        actual_max_edge_size,
        cells_on_cell);

    // Check each neighbor cell
    for (int i = 0; i < current_cell_vertices_number; ++i) {
        int neighbor_cell_id = static_cast<int>(cell_neig_vec[i]);
        if (neighbor_cell_id < 0) {
            continue;
        }

        bool in_neighbor = MOPS::CUDAKernel::IsInMesh(
            neighbor_cell_id,
            actual_max_edge_size,
            pos,
            number_vertex_on_cell,
            vertices_on_cell,
            vertex_coord);

        if (in_neighbor) {
            return neighbor_cell_id;
        }
    }

    // Not in current cell or any neighbor - truly outside mesh
    return -1;
}

// Helper function: Hash function for deterministic pseudo-random number generation
MOPS_DEVICE inline unsigned int hash(unsigned int x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x);
    return x;
}

// Helper function: Generate random float in [0, 1] using hash-based PRNG
MOPS_DEVICE inline float random_float(int particle_id, int timestep, int component)
{
    unsigned int seed = hash(static_cast<unsigned int>(particle_id * 73856093 + timestep * 19349663 + component * 83492791));
    return (seed & 0xFFFFFF) / 16777216.0f;
}

// Helper function: Generate random tangent velocity on sphere surface
// Used to prevent particles from getting stuck at boundaries
MOPS_DEVICE inline vec3 GenerateRandomTangentVelocityCUDA(
    const vec3& pos,
    int particle_id,
    int timestep,
    double magnitude,
    int attempt = 0)  // Added attempt parameter for multiple tries
{
    // Normalize position to get radial direction
    vec3 radial = pos;
    double radial_len = MOPS_LENGTH(radial);
    if (radial_len < 1e-12) {
        return vec3{0.0, 0.0, 0.0};
    }
    radial = radial / radial_len;

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

// Helper function: Try multiple random directions to find valid escape
// Returns true if valid position found, false if all attempts failed


// Wrapper function: Calculate velocity with edge projection constraint for stuck particles


std::vector<TrajectoryLine> PathLine(
    MPASOField* mpasoF,
    std::vector<CartesianCoord>& points,
    TrajectorySettings* config,
    std::vector<int>& default_cell_id)
{
    if (mpasoF == nullptr || config == nullptr) {
        Error("[CUDABackend::Kernel]::PathLine invalid input");
    }
    if (mpasoF->mGrid == nullptr || mpasoF->mSol_Front == nullptr || mpasoF->mSol_Back == nullptr) {
        Error("[CUDABackend::Kernel]::PathLine field is not initialized");
    }

    if (points.empty()) {
        return {};
    }
    if (config->deltaT == 0 || config->recordT == 0 || config->simulationDuration == 0) {
        Error("[CUDABackend::Kernel]::PathLine invalid trajectory settings");
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

    std::vector<float> effective_depths = MOPS::Common::BuildEffectiveDepths(stable_points, config, "PathLine");
    std::vector<TrajectoryLine> trajectory_lines = MOPS::Common::InitTrajectoryLines(stable_points, effective_depths, config);

    if (default_cell_id.size() != stable_points.size()) {
        default_cell_id.assign(stable_points.size(), -1);
        for (size_t i = 0; i < stable_points.size(); ++i) {
            int cell_id = -1;
            mpasoF->mGrid->searchKDT(stable_points[i], cell_id);
            default_cell_id[i] = cell_id;
        }
    }

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(mpasoF->mGrid.get());
    if (grid_info_vec.size() < 6) {
        Error("[CUDABackend::Kernel]::PathLine grid_info is incomplete");
    }

    const int actual_cell_size = static_cast<int>(grid_info_vec[0]);
    const int actual_max_edge_size = static_cast<int>(grid_info_vec[2]);
    const int actual_vertex_size = static_cast<int>(grid_info_vec[3]);
    const int actual_ztop_layer = static_cast<int>(grid_info_vec[4]);
    const int actual_ztop_layer_p1 = static_cast<int>(grid_info_vec[5]);

    // Collect attribute pointers from CtoV (vertex-interpolated) data
    // IMPORTANT: Collect in specific order (temperature, salinity) to match TrajectoryCommon.h expectations
    // where a.x() is temperature and a.y() is salinity
    std::vector<const std::vector<double>*> front_attr_ptrs;
    std::vector<const std::vector<double>*> back_attr_ptrs;

    // attr0 = temperature, attr1 = salinity (explicit order)
    auto front_temp_it = mpasoF->mSol_Front->mDoubleAttributes_CtoV.find("temperature");
    auto front_sal_it = mpasoF->mSol_Front->mDoubleAttributes_CtoV.find("salinity");
    auto back_temp_it = mpasoF->mSol_Back->mDoubleAttributes_CtoV.find("temperature");
    auto back_sal_it = mpasoF->mSol_Back->mDoubleAttributes_CtoV.find("salinity");

    if (front_temp_it != mpasoF->mSol_Front->mDoubleAttributes_CtoV.end()) {
        front_attr_ptrs.push_back(&front_temp_it->second);
    }
    if (front_sal_it != mpasoF->mSol_Front->mDoubleAttributes_CtoV.end()) {
        front_attr_ptrs.push_back(&front_sal_it->second);
    }
    if (back_temp_it != mpasoF->mSol_Back->mDoubleAttributes_CtoV.end()) {
        back_attr_ptrs.push_back(&back_temp_it->second);
    }
    if (back_sal_it != mpasoF->mSol_Back->mDoubleAttributes_CtoV.end()) {
        back_attr_ptrs.push_back(&back_sal_it->second);
    }

    int attr_count = static_cast<int>(front_attr_ptrs.size());
    if (static_cast<int>(back_attr_ptrs.size()) < attr_count) {
        attr_count = static_cast<int>(back_attr_ptrs.size());
    }

    const bool has_double_attributes = (attr_count > 0);

    const int particle_count = static_cast<int>(stable_points.size());
    const int each_points_size = static_cast<int>(config->simulationDuration / config->recordT);
    const int n_steps = static_cast<int>(config->simulationDuration / config->deltaT);
    const int dt_sign = (config->directionType == MOPS::CalcDirection::kForward) ? 1 : -1;
    const int delta_t = dt_sign * static_cast<int>(config->deltaT);
    const bool use_euler = (config->methodType == MOPS::CalcMethodType::kEuler);

    vec3* d_sample_points = DeviceAllocAndCopy(stable_points, "cudaMalloc/cudaMemcpy path sample_points");
    vec3* d_write_points = DeviceAllocAndCopy(update_points, "cudaMalloc/cudaMemcpy path write_points");
    vec3* d_write_vels = DeviceAllocAndCopy(update_vels, "cudaMalloc/cudaMemcpy path write_vels");
    vec3* d_write_attrs = DeviceAllocAndCopy(update_attrs, "cudaMalloc/cudaMemcpy path write_attrs");
    float* d_particle_depths = DeviceAllocAndCopy(effective_depths, "cudaMalloc/cudaMemcpy path particle_depths");
    int* d_default_cell_id = DeviceAllocAndCopy(default_cell_id, "cudaMalloc/cudaMemcpy path default_cell_id");

    vec3* d_vertex_coord = DeviceAllocAndCopy(mpasoF->mGrid->vertexCoord_vec, "cudaMalloc/cudaMemcpy path vertexCoord");
    vec3* d_cell_coord = DeviceAllocAndCopy(mpasoF->mGrid->cellCoord_vec, "cudaMalloc/cudaMemcpy path cellCoord");
    size_t* d_number_vertex_on_cell = DeviceAllocAndCopy(mpasoF->mGrid->numberVertexOnCell_vec, "cudaMalloc/cudaMemcpy path numberVertexOnCell");
    size_t* d_vertices_on_cell = DeviceAllocAndCopy(mpasoF->mGrid->verticesOnCell_vec, "cudaMalloc/cudaMemcpy path verticesOnCell");
    size_t* d_cells_on_cell = DeviceAllocAndCopy(mpasoF->mGrid->cellsOnCell_vec, "cudaMalloc/cudaMemcpy path cellsOnCell");

    int* d_max_level_cell = nullptr;
    if (!mpasoF->mGrid->maxLevelCell_vec.empty()) {
        d_max_level_cell = DeviceAllocAndCopy(mpasoF->mGrid->maxLevelCell_vec, "cudaMalloc/cudaMemcpy path maxLevelCell");
    } else {
        printf("[WARNING] maxLevelCell_vec is empty! Using fallback.\n");
        // Allocate a dummy array filled with max layer value
        std::vector<int> dummy_max_level(actual_cell_size, actual_ztop_layer);
        d_max_level_cell = DeviceAllocAndCopy(dummy_max_level, "cudaMalloc/cudaMemcpy path maxLevelCell (dummy)");
    }

    vec3* d_cell_vertex_velocity_front = DeviceAllocAndCopy(mpasoF->mSol_Front->cellVertexVelocity_vec, "cudaMalloc/cudaMemcpy path cellVertexVelocityFront");
    vec3* d_cell_vertex_velocity_back = DeviceAllocAndCopy(mpasoF->mSol_Back->cellVertexVelocity_vec, "cudaMalloc/cudaMemcpy path cellVertexVelocityBack");
    double* d_cell_vertex_ztop_front = DeviceAllocAndCopy(mpasoF->mSol_Front->cellVertexZTop_vec, "cudaMalloc/cudaMemcpy path cellVertexZTopFront");
    double* d_cell_vertex_ztop_back = DeviceAllocAndCopy(mpasoF->mSol_Back->cellVertexZTop_vec, "cudaMalloc/cudaMemcpy path cellVertexZTopBack");
    double* d_cell_vertex_vert_velocity_front = DeviceAllocAndCopy(mpasoF->mSol_Front->cellVertexVertVelocity_vec, "cudaMalloc/cudaMemcpy path cellVertexVertVelocityFront");
    double* d_cell_vertex_vert_velocity_back = DeviceAllocAndCopy(mpasoF->mSol_Back->cellVertexVertVelocity_vec, "cudaMalloc/cudaMemcpy path cellVertexVertVelocityBack");

    double* d_attr0_front = nullptr;
    double* d_attr1_front = nullptr;
    double* d_attr0_back = nullptr;
    double* d_attr1_back = nullptr;
    if (attr_count >= 1) {
        d_attr0_front = DeviceAllocAndCopy(*front_attr_ptrs[0], "cudaMalloc/cudaMemcpy path attr0Front");
        d_attr0_back = DeviceAllocAndCopy(*back_attr_ptrs[0], "cudaMalloc/cudaMemcpy path attr0Back");
    }
    if (attr_count >= 2) {
        d_attr1_front = DeviceAllocAndCopy(*front_attr_ptrs[1], "cudaMalloc/cudaMemcpy path attr1Front");
        d_attr1_back = DeviceAllocAndCopy(*back_attr_ptrs[1], "cudaMalloc/cudaMemcpy path attr1Back");
    }

    // Allocate boundary hit counter array
    std::vector<int> boundary_hit_count_host(particle_count, 0);
    int* d_boundary_hit_count = DeviceAllocAndCopy(boundary_hit_count_host, "cudaMalloc/cudaMemcpy path boundary_hit_count");

    dim3 block(128);
    dim3 grid((particle_count + block.x - 1) / block.x);

    KernelPathLine<<<grid, block>>>(
        particle_count,
        each_points_size,
        n_steps,
        static_cast<int>(config->simulationDuration),
        static_cast<int>(config->recordT),
        delta_t,
        use_euler,
        actual_cell_size,
        actual_max_edge_size,
        actual_vertex_size,
        actual_ztop_layer,
        actual_ztop_layer_p1,
        has_double_attributes,
        attr_count,
        d_default_cell_id,
        d_sample_points,
        d_write_points,
        d_write_vels,
        d_write_attrs,
        d_particle_depths,
        d_vertex_coord,
        d_cell_coord,
        d_number_vertex_on_cell,
        d_vertices_on_cell,
        d_cells_on_cell,
        d_max_level_cell,
        d_cell_vertex_velocity_front,
        d_cell_vertex_velocity_back,
        d_cell_vertex_ztop_front,
        d_cell_vertex_ztop_back,
        d_cell_vertex_vert_velocity_front,
        d_cell_vertex_vert_velocity_back,
        d_attr0_front,
        d_attr1_front,
        d_attr0_back,
        d_attr1_back,
        d_boundary_hit_count,
        config->perturbation_sigma_h,
        config->perturbation_sigma_v,
        config->masked_attr_value);

    CheckCuda(cudaGetLastError(), "KernelPathLine launch");
    CheckCuda(cudaDeviceSynchronize(), "KernelPathLine sync");

    CopyDeviceToHost(update_points, d_write_points, "cudaMemcpy path points");
    CopyDeviceToHost(update_vels, d_write_vels, "cudaMemcpy path velocities");

    // DEBUG: Check for zero velocities
    {
        int zero_count = 0;
        std::vector<int> zero_indices;
        for (size_t i = 0; i < update_vels.size(); ++i) {
            double vel_mag = std::sqrt(
                update_vels[i].x() * update_vels[i].x() +
                update_vels[i].y() * update_vels[i].y() +
                update_vels[i].z() * update_vels[i].z()
            );
            if (vel_mag < 1e-10) {
                zero_count++;
                if (zero_indices.size() < 50) {
                    zero_indices.push_back(i);
                }
            }
        }

        
    }

    CopyDeviceToHost(update_attrs, d_write_attrs, "cudaMemcpy path attrs");
    CopyDeviceToHost(boundary_hit_count_host, d_boundary_hit_count, "cudaMemcpy path boundary_hit_count");

    FreeDev(d_boundary_hit_count, "cudaFree path boundary_hit_count");
    FreeDev(d_attr1_back, "cudaFree path attr1Back");
    FreeDev(d_attr1_front, "cudaFree path attr1Front");
    FreeDev(d_attr0_back, "cudaFree path attr0Back");
    FreeDev(d_attr0_front, "cudaFree path attr0Front");
    FreeDev(d_cell_vertex_vert_velocity_back, "cudaFree path cellVertexVertVelocityBack");
    FreeDev(d_cell_vertex_vert_velocity_front, "cudaFree path cellVertexVertVelocityFront");
    FreeDev(d_cell_vertex_ztop_back, "cudaFree path cellVertexZTopBack");
    FreeDev(d_cell_vertex_ztop_front, "cudaFree path cellVertexZTopFront");
    FreeDev(d_cell_vertex_velocity_back, "cudaFree path cellVertexVelocityBack");
    FreeDev(d_cell_vertex_velocity_front, "cudaFree path cellVertexVelocityFront");
    FreeDev(d_cells_on_cell, "cudaFree path cellsOnCell");
    FreeDev(d_max_level_cell, "cudaFree path maxLevelCell");
    FreeDev(d_vertices_on_cell, "cudaFree path verticesOnCell");
    FreeDev(d_number_vertex_on_cell, "cudaFree path numberVertexOnCell");
    FreeDev(d_cell_coord, "cudaFree path cellCoord");
    FreeDev(d_vertex_coord, "cudaFree path vertexCoord");
    FreeDev(d_default_cell_id, "cudaFree path defaultCellID");
    FreeDev(d_particle_depths, "cudaFree path particleDepths");
    FreeDev(d_write_attrs, "cudaFree path writeAttrs");
    FreeDev(d_write_vels, "cudaFree path writeVels");
    FreeDev(d_write_points, "cudaFree path writePoints");
    FreeDev(d_sample_points, "cudaFree path samplePoints");

    const size_t total_points = update_points.size();
    auto clean_traj = MOPS::Common::FinalizeTrajectoryLinesWithAttrs(
        trajectory_lines,
        update_points,
        update_vels,
        update_attrs,
        static_cast<size_t>(each_points_size),
        total_points);

    trajectory_lines.clear();
    return clean_traj;
}

} // namespace MOPS::GPU::CUDABackend::Kernel
