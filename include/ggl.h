#pragma once
//c++ header file
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <stack>
#include <queue>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <iomanip>
#include <chrono>
#include <limits>
#include <numbers>
#include <cmath>
#include <filesystem>
#include <cstddef>



//c header file
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <atomic>

#define LOADING_CYCODE_BASE     0
#define LOADING_NETCDF_CXX      1
#define LOADING_SYCL            1

#ifndef MOPS_VTK
#define MOPS_VTK 0
#endif

#if MOPS_VTK
    //VTK Readers/Writers
    #include <vtkUnstructuredGridReader.h>
    #include <vtkXMLUnstructuredGridReader.h>
    #include <vtkXMLPolyDataReader.h>
    #include <vtkXMLImageDataWriter.h>
    #include <vtkXMLPolyDataWriter.h>

    //VTK Data Structures and Sources
    #include <vtkAppendFilter.h>
    #include <vtkSphereSource.h>
    #include <vtkUnstructuredGrid.h>
    #include <vtkImageData.h>
    #include <vtkPoints.h>
    #include <vtkPointData.h>
    #include <vtkPolyLine.h>
    #include <vtkCellArray.h>
    #include <vtkTetra.h>
    #include <vtkLine.h>

    //VTK Rendering and Visualization
    #include <vtkAppendPolyData.h>
    #include <vtkActor.h>
    #include <vtkCamera.h>
    #include <vtkDataSetMapper.h>
    #include <vtkNamedColors.h>
    #include <vtkNew.h>
    #include <vtkProperty.h>
    #include <vtkDoubleArray.h>
    #include <vtkRenderWindow.h>
    #include <vtkRenderWindowInteractor.h>
    #include <vtkRenderer.h>
    #include <vtkSmartPointer.h>
#endif

#if LOADING_CYCODE_BASE == 1
    #include "cyCodeBase/cyCore.h"
    #include "cyCodeBase/cyVector.h"
    #include "cyCodeBase/cyColor.h"
    #include "cyCodeBase/cyIVector.h"
    #include "cyCodeBase/cyMatrix.h"
    #include "cyCodeBase/cyMatrix.h"
    #include "cyCodeBase/cyQuat.h"

    using vec2i = cy::IVec2i;
    using vec2f = cy::Vec2f;
    using vec3f = cy::Vec3f;
    using vec3i = cy::IVec3i;

#endif

#if LOADING_NETCDF_CXX == 1
    #include "netcdf.h"
#endif

#if LOADING_SYCL == 1
    #include <sycl/sycl.hpp>
    // #include <sycl/ext/intel/math.hpp>

    using vec2 = sycl::double2;
    using vec3 = sycl::double3;
    using vec4 = sycl::double4;
    using vec2i = sycl::int2;
    using vec3i = sycl::int3;
    using vec4i = sycl::int4;
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif // !M_PI


using SphericalCoord = vec2;
using CartesianCoord = vec3;


#if LOADING_SYCL == 0 && LOADING_CYCODE_BASE == 1
#define USE_SYCL 0
    #define YOSEF_DOT(A,B)          A.Dot(B)
    #define YOSEF_CROSS(A,B)        A.Cross(B)
    #define YOSEF_VEC3_MIN(a, b)    vec3f(std::min((a).x, (b).x), std::min((a).y, (b).y), std::min((a).z, (b).z))
    #define YOSEF_VEC3_MAX(a, b)    vec3f(std::max((a).x, (b).x), std::max((a).y, (b).y), std::max((a).z, (b).z))
    #define PRINT_VEC3(v)   std::cout << #v << ": (" << (v).x << ", " << (v).y << ", " << (v).z << ")" << std::endl
    #define PRINT_VEC2(v)   std::cout << #v << ": (" << (v).x << ", " << (v).y << ")" << std::endl
    #define PRINT_DOUBLE(v) std::cout << #v << ": " << v << std::endl
    #define PRINT_INT(v)    std::cout << #v << ": " << v << std::endl
    #define PRINT_FLOAT(v)  std::cout << #v << ": " << v << std::endl
    #define NaN 			std::numeric_limits<float>::quiet_NaN()
    #define PRINT_GAP       std::cout << "=====================================" << std::endl#endif
#endif

#if LOADING_SYCL == 1 && LOADING_CYCODE_BASE == 0
#define USE_SYCL 1
    #define YOSEF_DOT(A,B)          sycl::dot(A, B)
    #define YOSEF_CROSS(A,B)        sycl::cross(A, B)
    #define YOSEF_VEC3_MIN(a, b)    sycl::float3(std::min((a).x(), (b).x()), std::min((a).y(), (b).y()), std::min((a).z(), (b).z()))
    #define YOSEF_VEC3_MAX(a, b)    sycl::float3(std::max((a).x(), (b).x()), std::max((a).y(), (b).y()), std::max((a).z(), (b).z()))
    #define YOSEF_LENGTH(v)         std::sqrt((v).x() * (v).x() + (v).y() * (v).y() + (v).z() * (v).z())
    #define YOSEF_VEC3_NORMALIZE(v)     \
    do {                                \
        double _len = YOSEF_LENGTH(v);  \
        (v).x() /= _len;         \
        (v).y() /= _len;         \
        (v).z() /= _len;         \
    } while(0) 
    #define PRINT_VEC3(v)           std::cout << #v << ": (" << (v).x() << ", " << (v).y() << ", " << (v).z() << ")" << std::endl
    #define PRINT_VEC2(v)           std::cout << #v << ": (" << (v).x() << ", " << (v).y() << ")" << std::endl
    #define PRINT_DOUBLE(v)         std::cout << #v << ": " << v << std::endl
    #define PRINT_INT(v)            std::cout << #v << ": " << v << std::endl
    #define PRINT_FLOAT(v)          std::cout << #v << ": " << v << std::endl
    #define NaN 			        std::numeric_limits<float>::quiet_NaN()
    #define PRINT_GAP               std::cout << "=====================================" << std::endl
#endif
