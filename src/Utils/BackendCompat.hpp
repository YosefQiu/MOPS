#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#if defined(MOPS_USE_SYCL) && (MOPS_USE_SYCL == 1)
#define USE_SYCL 1
#else
#define USE_SYCL 0
#endif

#if defined(MOPS_USE_CUDA) && (MOPS_USE_CUDA == 1)
#define USE_CUDA 1
#else
#define USE_CUDA 0
#endif

#if defined(MOPS_USE_HIP) && (MOPS_USE_HIP == 1)
#define USE_HIP 1
#else
#define USE_HIP 0
#endif

#if defined(MOPS_USE_TBB) && (MOPS_USE_TBB == 1)
#define USE_TBB 1
#else
#define USE_TBB 0
#endif

#if defined(MOPS_USE_CPU) && (MOPS_USE_CPU == 1)
#define USE_CPU 1
#else
#define USE_CPU 0
#endif

#if defined(MOPS_USE_GPU) && (MOPS_USE_GPU == 1)
#define USE_GPU 1
#else
#define USE_GPU 0
#endif

#if ((USE_SYCL + USE_CUDA + USE_HIP + USE_TBB) > 1)
#error "BackendCompat: conflicting backend flags detected. Enable only one of MOPS_USE_SYCL, MOPS_USE_CUDA, MOPS_USE_HIP, or MOPS_USE_TBB."
#endif

#if ((USE_CPU + USE_GPU) > 1)
#error "BackendCompat: conflicting runtime-domain flags detected. Enable only one of MOPS_USE_CPU or MOPS_USE_GPU."
#endif

#if (USE_TBB == 1) && (USE_CPU == 0)
#error "BackendCompat: MOPS_USE_TBB=1 requires MOPS_USE_CPU=1."
#endif

#if ((USE_SYCL == 1) || (USE_CUDA == 1) || (USE_HIP == 1)) && (USE_GPU == 0)
#error "BackendCompat: MOPS_USE_SYCL/MOPS_USE_CUDA/MOPS_USE_HIP requires MOPS_USE_GPU=1."
#endif

#if USE_SYCL
#include <sycl/sycl.hpp>
#else
namespace sycl {
class queue {};
} // namespace sycl
#endif

#if USE_CUDA
#include <cuda_runtime.h>
#include "Utils/CUDACommon/helper_math.h"

using vec2 = double2;
using vec3 = double3;
using vec4 = double4;
using vec2i = int2;
using vec3i = int3;
using vec4i = int4;

inline __host__ __device__ double MOPS_CUDA_DOT(const vec3& a, const vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ vec3 MOPS_CUDA_CROSS(const vec3& a, const vec3& b)
{
    return make_double3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

inline __host__ __device__ vec2 operator+(const vec2& a, const vec2& b) { return make_double2(a.x + b.x, a.y + b.y); }
inline __host__ __device__ vec2 operator-(const vec2& a, const vec2& b) { return make_double2(a.x - b.x, a.y - b.y); }
inline __host__ __device__ vec2 operator*(const vec2& a, double s) { return make_double2(a.x * s, a.y * s); }
inline __host__ __device__ vec2 operator*(double s, const vec2& a) { return make_double2(a.x * s, a.y * s); }
inline __host__ __device__ vec2 operator/(const vec2& a, double s) { return make_double2(a.x / s, a.y / s); }

inline __host__ __device__ vec3 operator+(const vec3& a, const vec3& b) { return make_double3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline __host__ __device__ vec3 operator-(const vec3& a, const vec3& b) { return make_double3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline __host__ __device__ vec3 operator*(const vec3& a, double s) { return make_double3(a.x * s, a.y * s, a.z * s); }
inline __host__ __device__ vec3 operator*(double s, const vec3& a) { return make_double3(a.x * s, a.y * s, a.z * s); }
inline __host__ __device__ vec3 operator/(const vec3& a, double s) { return make_double3(a.x / s, a.y / s, a.z / s); }

inline __host__ __device__ vec3& operator+=(vec3& a, const vec3& b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
inline __host__ __device__ vec3& operator-=(vec3& a, const vec3& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }

#elif USE_HIP
#include <hip/hip_runtime.h>

using vec2 = double2;
using vec3 = double3;
using vec4 = double4;
using vec2i = int2;
using vec3i = int3;
using vec4i = int4;

inline __host__ __device__ double MOPS_HIP_DOT(const vec3& a, const vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ vec3 MOPS_HIP_CROSS(const vec3& a, const vec3& b)
{
    return make_double3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

inline __host__ __device__ vec2 operator+(const vec2& a, const vec2& b) { return make_double2(a.x + b.x, a.y + b.y); }
inline __host__ __device__ vec2 operator-(const vec2& a, const vec2& b) { return make_double2(a.x - b.x, a.y - b.y); }
inline __host__ __device__ vec2 operator*(const vec2& a, double s) { return make_double2(a.x * s, a.y * s); }
inline __host__ __device__ vec2 operator*(double s, const vec2& a) { return make_double2(a.x * s, a.y * s); }
inline __host__ __device__ vec2 operator/(const vec2& a, double s) { return make_double2(a.x / s, a.y / s); }

inline __host__ __device__ vec3 operator+(const vec3& a, const vec3& b) { return make_double3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline __host__ __device__ vec3 operator-(const vec3& a, const vec3& b) { return make_double3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline __host__ __device__ vec3 operator*(const vec3& a, double s) { return make_double3(a.x * s, a.y * s, a.z * s); }
inline __host__ __device__ vec3 operator*(double s, const vec3& a) { return make_double3(a.x * s, a.y * s, a.z * s); }
inline __host__ __device__ vec3 operator/(const vec3& a, double s) { return make_double3(a.x / s, a.y / s, a.z / s); }

inline __host__ __device__ vec3& operator+=(vec3& a, const vec3& b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
inline __host__ __device__ vec3& operator-=(vec3& a, const vec3& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }

#elif USE_TBB
#include "Utils/CPUCommon/cyVector.h"

using vec2 = cy::Vec2d;
using vec3 = cy::Vec3d;
using vec4 = cy::Vec4d;
using vec2i = cy::Vec2<int>;
using vec3i = cy::Vec3<int>;
using vec4i = cy::Vec4<int>;

#else
using vec2 = sycl::double2;
using vec3 = sycl::double3;
using vec4 = sycl::double4;
using vec2i = sycl::int2;
using vec3i = sycl::int3;
using vec4i = sycl::int4;
#endif

using SphericalCoord = vec2;
using CartesianCoord = vec3;

#if USE_SYCL == 1
#define MOPS_DOT(A, B) sycl::dot(A, B)
#define MOPS_CROSS(A, B) sycl::cross(A, B)
#define MOPS_VEC3_MIN(a, b) sycl::float3(std::min((a).x(), (b).x()), std::min((a).y(), (b).y()), std::min((a).z(), (b).z()))
#define MOPS_VEC3_MAX(a, b) sycl::float3(std::max((a).x(), (b).x()), std::max((a).y(), (b).y()), std::max((a).z(), (b).z()))
#define MOPS_LENGTH(v) std::sqrt((v).x() * (v).x() + (v).y() * (v).y() + (v).z() * (v).z())
#define MOPS_VEC3_NORMALIZE(v) \
    do { \
        double _len = MOPS_LENGTH(v); \
        (v).x() /= _len; \
        (v).y() /= _len; \
        (v).z() /= _len; \
    } while (0)
#define PRINT_VEC3(v) std::cout << #v << ": (" << (v).x() << ", " << (v).y() << ", " << (v).z() << ")" << std::endl
#define PRINT_VEC2(v) std::cout << #v << ": (" << (v).x() << ", " << (v).y() << ")" << std::endl
#define PRINT_DOUBLE(v) std::cout << #v << ": " << v << std::endl
#define PRINT_INT(v) std::cout << #v << ": " << v << std::endl
#define PRINT_FLOAT(v) std::cout << #v << ": " << v << std::endl
#define NaN std::numeric_limits<float>::quiet_NaN()
#define PRINT_GAP std::cout << "=====================================" << std::endl
#endif

#if USE_CUDA == 1
/* Keep existing vec.x()/vec.y()/vec.z() call sites source-compatible with CUDA vector fields. */
#define x() x
#define y() y
#define z() z
#define w() w

#define MOPS_DOT(A, B) MOPS_CUDA_DOT((A), (B))
#define MOPS_CROSS(A, B) MOPS_CUDA_CROSS((A), (B))
#define MOPS_VEC3_MIN(a, b) make_double3(std::min((a).x, (b).x), std::min((a).y, (b).y), std::min((a).z, (b).z))
#define MOPS_VEC3_MAX(a, b) make_double3(std::max((a).x, (b).x), std::max((a).y, (b).y), std::max((a).z, (b).z))
#define MOPS_LENGTH(v) std::sqrt((v).x * (v).x + (v).y * (v).y + (v).z * (v).z)
#define MOPS_VEC3_NORMALIZE(v) \
    do { \
        double _len = MOPS_LENGTH(v); \
        (v).x /= _len; \
        (v).y /= _len; \
        (v).z /= _len; \
    } while (0)
#define PRINT_VEC3(v) std::cout << #v << ": (" << (v).x << ", " << (v).y << ", " << (v).z << ")" << std::endl
#define PRINT_VEC2(v) std::cout << #v << ": (" << (v).x << ", " << (v).y << ")" << std::endl
#define PRINT_DOUBLE(v) std::cout << #v << ": " << v << std::endl
#define PRINT_INT(v) std::cout << #v << ": " << v << std::endl
#define PRINT_FLOAT(v) std::cout << #v << ": " << v << std::endl
#define NaN std::numeric_limits<double>::quiet_NaN()
#define PRINT_GAP std::cout << "=====================================" << std::endl
#endif

#if USE_HIP == 1
/* Keep existing vec.x()/vec.y()/vec.z() call sites source-compatible with HIP vector fields. */
#define x() x
#define y() y
#define z() z
#define w() w

#define MOPS_DOT(A, B) MOPS_HIP_DOT((A), (B))
#define MOPS_CROSS(A, B) MOPS_HIP_CROSS((A), (B))
#define MOPS_VEC3_MIN(a, b) make_double3(std::min((a).x, (b).x), std::min((a).y, (b).y), std::min((a).z, (b).z))
#define MOPS_VEC3_MAX(a, b) make_double3(std::max((a).x, (b).x), std::max((a).y, (b).y), std::max((a).z, (b).z))
#define MOPS_LENGTH(v) std::sqrt((v).x * (v).x + (v).y * (v).y + (v).z * (v).z)
#define MOPS_VEC3_NORMALIZE(v) \
    do { \
        double _len = MOPS_LENGTH(v); \
        (v).x /= _len; \
        (v).y /= _len; \
        (v).z /= _len; \
    } while (0)
#define PRINT_VEC3(v) std::cout << #v << ": (" << (v).x << ", " << (v).y << ", " << (v).z << ")" << std::endl
#define PRINT_VEC2(v) std::cout << #v << ": (" << (v).x << ", " << (v).y << ")" << std::endl
#define PRINT_DOUBLE(v) std::cout << #v << ": " << v << std::endl
#define PRINT_INT(v) std::cout << #v << ": " << v << std::endl
#define PRINT_FLOAT(v) std::cout << #v << ": " << v << std::endl
#define NaN std::numeric_limits<double>::quiet_NaN()
#define PRINT_GAP std::cout << "=====================================" << std::endl
#endif

#if USE_TBB == 1
/* Keep existing vec.x()/vec.y()/vec.z() call sites source-compatible with cy::Vec fields. */
#define x() x
#define y() y
#define z() z
#define w() w

#define MOPS_DOT(A, B) ((A).Dot((B)))
#define MOPS_CROSS(A, B) ((A).Cross((B)))
#define MOPS_VEC3_MIN(a, b) vec3(std::min((a).x, (b).x), std::min((a).y, (b).y), std::min((a).z, (b).z))
#define MOPS_VEC3_MAX(a, b) vec3(std::max((a).x, (b).x), std::max((a).y, (b).y), std::max((a).z, (b).z))
#define MOPS_LENGTH(v) std::sqrt((v).x * (v).x + (v).y * (v).y + (v).z * (v).z)
#define MOPS_VEC3_NORMALIZE(v) \
    do { \
        double _len = MOPS_LENGTH(v); \
        (v).x /= _len; \
        (v).y /= _len; \
        (v).z /= _len; \
    } while (0)
#define PRINT_VEC3(v) std::cout << #v << ": (" << (v).x << ", " << (v).y << ", " << (v).z << ")" << std::endl
#define PRINT_VEC2(v) std::cout << #v << ": (" << (v).x << ", " << (v).y << ")" << std::endl
#define PRINT_DOUBLE(v) std::cout << #v << ": " << v << std::endl
#define PRINT_INT(v) std::cout << #v << ": " << v << std::endl
#define PRINT_FLOAT(v) std::cout << #v << ": " << v << std::endl
#define NaN std::numeric_limits<double>::quiet_NaN()
#define PRINT_GAP std::cout << "=====================================" << std::endl
#endif

#if USE_CUDA == 1
#define MOPS_HOST_DEVICE __host__ __device__
#define MOPS_DEVICE __device__
#define MOPS_HOST __host__
#elif USE_HIP == 1
/*
 * In HIP mode, only HIP device translation units should carry device qualifiers.
 * Host-only translation units keep empty qualifiers to avoid hipcc/nvcc side effects.
 */
#if defined(MOPS_HIP_DEVICE_BUILD) && (MOPS_HIP_DEVICE_BUILD == 1)
#define MOPS_HOST_DEVICE __host__ __device__
#define MOPS_DEVICE __device__
#define MOPS_HOST __host__
#else
#define MOPS_HOST_DEVICE
#define MOPS_DEVICE
#define MOPS_HOST
#endif
#else
#define MOPS_HOST_DEVICE
#define MOPS_DEVICE
#define MOPS_HOST
#endif

// Backward-compatibility aliases for older call sites.
#ifndef MOPS_DISABLE_YOSEF_COMPAT
#if USE_HIP == 1
#define YOSEF_HIP_DOT MOPS_HIP_DOT
#define YOSEF_HIP_CROSS MOPS_HIP_CROSS
/* In HIP mode, legacy CUDA-named aliases are redirected to HIP implementations. */
#define YOSEF_CUDA_DOT MOPS_HIP_DOT
#define YOSEF_CUDA_CROSS MOPS_HIP_CROSS
#elif USE_CUDA == 1
#define YOSEF_CUDA_DOT MOPS_CUDA_DOT
#define YOSEF_CUDA_CROSS MOPS_CUDA_CROSS
#endif
#define YOSEF_DOT MOPS_DOT
#define YOSEF_CROSS MOPS_CROSS
#define YOSEF_VEC3_MIN MOPS_VEC3_MIN
#define YOSEF_VEC3_MAX MOPS_VEC3_MAX
#define YOSEF_LENGTH MOPS_LENGTH
#define YOSEF_VEC3_NORMALIZE MOPS_VEC3_NORMALIZE
#endif

namespace MOPS::math {

template <typename T>
MOPS_HOST_DEVICE inline bool isnan(T v)
{
#if USE_SYCL
    return sycl::isnan(v);
#else
    return std::isnan(v);
#endif
}

template <typename T>
MOPS_HOST_DEVICE inline bool isfinite(T v)
{
#if USE_SYCL
    return sycl::isfinite(v);
#else
    return std::isfinite(v);
#endif
}

template <typename T>
MOPS_HOST_DEVICE inline T sqrt(T v)
{
#if USE_SYCL
    return sycl::sqrt(v);
#else
    return std::sqrt(v);
#endif
}

template <typename T>
MOPS_HOST_DEVICE inline T sin(T v)
{
#if USE_SYCL
    return sycl::sin(v);
#else
    return std::sin(v);
#endif
}

template <typename T>
MOPS_HOST_DEVICE inline T cos(T v)
{
#if USE_SYCL
    return sycl::cos(v);
#else
    return std::cos(v);
#endif
}

template <typename T>
MOPS_HOST_DEVICE inline T asin(T v)
{
#if USE_SYCL
    return sycl::asin(v);
#else
    return std::asin(v);
#endif
}

template <typename T>
MOPS_HOST_DEVICE inline T fabs(T v)
{
#if USE_SYCL
    return sycl::fabs(v);
#else
    return std::fabs(v);
#endif
}

template <typename T>
MOPS_HOST_DEVICE inline T atan2(T y, T x)
{
#if USE_SYCL
    return sycl::atan2(y, x);
#else
    return std::atan2(y, x);
#endif
}

} // namespace MOPS::math
