#include "GPU/CUDA/Kernel/MPASOVisualizerKernels.h"

#include <cuda_runtime.h>

#include "GPU/CUDA/Kernel/CUDAKernel.h"
#include "Common/ImageBuffer.hpp"
#include "Common/TrajectoryCommon.h"
#include "Utils/Interpolation.hpp"

namespace {

inline void CheckCuda(cudaError_t code, const char* what)
{
    if (code != cudaSuccess) {
        Error("[CUDA]::%s failed: %s", what, cudaGetErrorString(code));
    }
}

template <typename T>
T* DeviceAllocAndCopy(const std::vector<T>& src, const char* what)
{
    if (src.empty()) {
        return nullptr;
    }

    T* dst = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&dst), src.size() * sizeof(T)), what);
    CheckCuda(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), what);
    return dst;
}

template <typename T>
void FreeDev(T* dev_ptr, const char* what)
{
    if (dev_ptr != nullptr) {
        CheckCuda(cudaFree(dev_ptr), what);
    }
}

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

    constexpr int MAX_VERTEX_NUM = 20;
    constexpr int MAX_VERTLEVELS = 100;

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
    if (ztop_levels <= 0 || ztop_levels > MAX_VERTLEVELS) {
        MOPS::SetPixel(img0, width, height, y, x, vec3_nan);
        if (has_attr && img1 != nullptr) {
            MOPS::SetPixel(img1, width, height, y, x, vec3_nan);
        }
        return;
    }

    double current_point_ztop_vec[MAX_VERTLEVELS];
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
    for (int k = 1; k < ztop_levels; ++k) {
        double top_i = current_point_ztop_vec[k - 1];
        double bot_i = current_point_ztop_vec[k];
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
    if (fixed_depth <= current_point_ztop_vec[0]) {
        local_layer = 0;
    }
    if (local_layer < 0) {
        MOPS::SetPixel(img0, width, height, y, x, vec3_nan);
        if (has_attr && img1 != nullptr) {
            MOPS::SetPixel(img1, width, height, y, x, vec3_nan);
        }
        return;
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

    constexpr int MAX_VERTEX_NUM = 20;

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

    std::vector<const std::vector<double>*> attr_ptrs;
    for (const auto& kv : mpasoF->mSol_Front->mDoubleAttributes_CtoV) {
        attr_ptrs.push_back(&kv.second);
        if (attr_ptrs.size() == 2) {
            break;
        }
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

    constexpr int MAX_VERTEX_NUM = 20;
    constexpr int MAX_VERTLEVELS = 100;
    auto double_nan = nan("");
    vec3 vec3_nan = {double_nan, double_nan, double_nan};

    const int actual_vertex_size = mpasoF->mGrid->mVertexSize;
    const int ztop_size = static_cast<int>(mpasoF->mSol_Front->cellVertexZTop_vec.size());
    const int vel_size = static_cast<int>(mpasoF->mSol_Front->cellVertexVelocity_vec.size());
    if (actual_vertex_size <= 0 || ztop_size <= 0 || vel_size <= 0) {
        Error("[CUDABackend::Kernel]::VisualizeFixedLatitude invalid mesh/solution array sizes");
    }

    int n_vert = ztop_size / actual_vertex_size;
    if (n_vert <= 0 || n_vert > MAX_VERTLEVELS || (vel_size / actual_vertex_size) < n_vert) {
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

            double current_point_ztop_vec[MAX_VERTLEVELS];
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

struct StreamlineVelocityState {
    vec3 h_vel;
    double v_vel;
    bool ok;
};

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
    constexpr int MAX_VERTEX_NUM = 20;
    constexpr int MAX_VERTICAL_LEVEL_NUM = 100;

    if (cell_id < 0) {
        return {vec3{0.0, 0.0, 0.0}, 0.0, false};
    }
    if (actual_ztop_layer <= 1 || actual_ztop_layer > MAX_VERTICAL_LEVEL_NUM) {
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

    double current_point_ztop_vec[MAX_VERTICAL_LEVEL_NUM];
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

    constexpr int MAX_CELL_NEIGHBOR_NUM = 21;

    int run_time = 0;
    bool first_loop = true;
    bool first_vel = true;

    const int base_idx = global_id * each_points_size;
    int update_points_idx = 0;

    vec3 sample_point_position;
    vec3 new_position;
    int cell_id = -1;
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

            int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
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
            }
            update_points_idx += 1;
        }
    }
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

struct PathlineVelocityState {
    vec3 h_vel;
    double v_vel;
    vec3 attr;
    bool ok;
};

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
    const double* attr1_back)
{
    constexpr int MAX_VERTEX_NUM = 10;
    constexpr int MAX_VERTICAL_LEVEL_NUM = 100;

    auto ret0 = []() -> PathlineVelocityState {
        return {vec3{0.0, 0.0, 0.0}, 0.0, vec3{0.0, 0.0, 0.0}, false};
    };

    if (cell_id < 0) {
        return ret0();
    }
    if (actual_ztop_layer <= 1 || actual_ztop_layer > MAX_VERTICAL_LEVEL_NUM) {
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

    double current_point_ztop_front_vec[MAX_VERTICAL_LEVEL_NUM];
    double current_point_ztop_back_vec[MAX_VERTICAL_LEVEL_NUM];
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
            if (current_depth <= top_i + eps && current_depth >= bot_i - eps) {
                local_layer_back = k;
                break;
            }
        }
    }

    if (local_layer_front < 0 || local_layer_back < 0) {
        return ret0();
    }

    double ztop_layer_front_dn = current_point_ztop_front_vec[local_layer_front];
    double ztop_layer_front_up = current_point_ztop_front_vec[local_layer_front - 1];
    double ztop_layer_back_dn = current_point_ztop_back_vec[local_layer_back];
    double ztop_layer_back_up = current_point_ztop_back_vec[local_layer_back - 1];

    double x_front = current_depth;
    double x_back = current_depth;

    x_front = fmax(ztop_layer_front_dn, fmin(x_front, ztop_layer_front_up));
    double denom = ztop_layer_front_up - ztop_layer_front_dn;
    if (fabs(denom) < 1e-12) {
        return ret0();
    }
    double t_front = (x_front - ztop_layer_front_dn) / denom;

    x_back = fmax(ztop_layer_back_dn, fmin(x_back, ztop_layer_back_up));
    denom = ztop_layer_back_up - ztop_layer_back_dn;
    if (fabs(denom) < 1e-12) {
        return ret0();
    }
    double t_back = (x_back - ztop_layer_back_dn) / denom;

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

    return {current_horizontal_velocity, current_vertical_velocity, current_point_attr_value, true};
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
    const vec3* cell_vertex_velocity_front,
    const vec3* cell_vertex_velocity_back,
    const double* cell_vertex_ztop_front,
    const double* cell_vertex_ztop_back,
    const double* cell_vertex_vert_velocity_front,
    const double* cell_vertex_vert_velocity_back,
    const double* attr0_front,
    const double* attr1_front,
    const double* attr0_back,
    const double* attr1_back)
{
    const int global_id = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (global_id >= particle_count || each_points_size <= 0 || n_steps <= 0) {
        return;
    }

    constexpr int MAX_VERTEX_NUM = 10;
    constexpr int MAX_CELL_NEIGHBOR_NUM = 11;

    double run_time = 0.0;
    bool first_loop = true;
    bool first_vel = true;
    bool first_attr = true;

    int base_idx = global_id * each_points_size;
    int update_points_idx = 0;

    vec3 sample_point_position;
    vec3 new_position;
    int cell_id = -1;
    int cell_neig_vec[MAX_CELL_NEIGHBOR_NUM];

    for (int i = 0; i < MAX_CELL_NEIGHBOR_NUM; ++i) {
        cell_neig_vec[i] = -1;
    }

    for (int i_step = 0; i_step < n_steps; ++i_step) {
        double alpha = static_cast<double>(i_step) / static_cast<double>(n_steps);
        run_time += delta_t;
        (void)run_time;

        sample_point_position = sample_points[global_id];
        int first_cell_id = default_cell_id[global_id];
        double current_depth = -1.0 * static_cast<double>(particle_depths[global_id]);

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
                MAX_VERTEX_NUM,
                actual_max_edge_size,
                cells_on_cell);
            write_points[base_idx] = sample_point_position;
        } else {
            if (cell_id < 0 || cell_id >= actual_cell_size) {
                return;
            }

            int current_cell_vertices_number = static_cast<int>(number_vertex_on_cell[cell_id]);
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
            PathlineVelocityState s = CalcVelocityAtPathlineCUDA(
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
                attr1_back);

            current_horizontal_velocity = s.h_vel;
            current_vertical_velocity = s.v_vel;
            current_attrs = s.attr;
        } else {
            const double dt = static_cast<double>(delta_t);
            const double dalpha = dt / static_cast<double>(simulation_duration);

            double a1 = alpha;
            PathlineVelocityState s1 = CalcVelocityAtPathlineCUDA(
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
                attr1_back);

            vec3 p2 = AdvectOnSphereCUDA(current_position, s1.h_vel, dt * 0.5);
            double a2 = a1 + 0.5 * dalpha;
            if (a2 > 1.0) a2 = 1.0;
            if (a2 < 0.0) a2 = 0.0;
            PathlineVelocityState s2 = CalcVelocityAtPathlineCUDA(
                p2,
                cell_id,
                current_depth,
                a2,
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
                attr1_back);

            vec3 p3 = AdvectOnSphereCUDA(current_position, s2.h_vel, dt * 0.5);
            double a3 = a1 + 0.5 * dalpha;
            if (a3 > 1.0) a3 = 1.0;
            if (a3 < 0.0) a3 = 0.0;
            PathlineVelocityState s3 = CalcVelocityAtPathlineCUDA(
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
                attr1_back);

            vec3 p4 = AdvectOnSphereCUDA(current_position, s3.h_vel, dt);
            double a4 = a1 + dalpha;
            if (a4 > 1.0) a4 = 1.0;
            if (a4 < 0.0) a4 = 0.0;
            PathlineVelocityState s4 = CalcVelocityAtPathlineCUDA(
                p4,
                cell_id,
                current_depth,
                a4,
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
                attr1_back);

            current_horizontal_velocity = (s1.h_vel + 2.0 * s2.h_vel + 2.0 * s3.h_vel + s4.h_vel) / 6.0;
            current_attrs = (s1.attr + 2.0 * s2.attr + 2.0 * s3.attr + s4.attr) / 6.0;
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

        if (first_vel) {
            first_vel = false;
            write_vels[base_idx] = current_horizontal_velocity;
        }
        if (first_attr && has_double_attributes) {
            first_attr = false;
            write_attrs[base_idx] = current_attrs;
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
            }
            update_points_idx += 1;
        }
    }
}

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

    const bool has_double_attributes = (mpasoF->mSol_Front->mDoubleAttributes.size() > 1);

    std::vector<const std::vector<double>*> front_attr_ptrs;
    std::vector<const std::vector<double>*> back_attr_ptrs;
    if (has_double_attributes) {
        for (const auto& kv : mpasoF->mSol_Front->mDoubleAttributes_CtoV) {
            front_attr_ptrs.push_back(&kv.second);
            if (front_attr_ptrs.size() == 2) {
                break;
            }
        }
        for (const auto& kv : mpasoF->mSol_Back->mDoubleAttributes_CtoV) {
            back_attr_ptrs.push_back(&kv.second);
            if (back_attr_ptrs.size() == 2) {
                break;
            }
        }
    }
    int attr_count = static_cast<int>(front_attr_ptrs.size());
    if (static_cast<int>(back_attr_ptrs.size()) < attr_count) {
        attr_count = static_cast<int>(back_attr_ptrs.size());
    }

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
        d_cell_vertex_velocity_front,
        d_cell_vertex_velocity_back,
        d_cell_vertex_ztop_front,
        d_cell_vertex_ztop_back,
        d_cell_vertex_vert_velocity_front,
        d_cell_vertex_vert_velocity_back,
        d_attr0_front,
        d_attr1_front,
        d_attr0_back,
        d_attr1_back);

    CheckCuda(cudaGetLastError(), "KernelPathLine launch");
    CheckCuda(cudaDeviceSynchronize(), "KernelPathLine sync");

    CheckCuda(
        cudaMemcpy(
            update_points.data(),
            d_write_points,
            update_points.size() * sizeof(vec3),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy path points");

    CheckCuda(
        cudaMemcpy(
            update_vels.data(),
            d_write_vels,
            update_vels.size() * sizeof(vec3),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy path velocities");

    CheckCuda(
        cudaMemcpy(
            update_attrs.data(),
            d_write_attrs,
            update_attrs.size() * sizeof(vec3),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy path attrs");

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
