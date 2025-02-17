#pragma once
#include "ggl.h"
#include "ImageBuffer.hpp"
#include "GeoConverter.hpp"
#include "MPASOField.h"

enum class CalcPositionType : int { kCenter, kVertx, kPoint, kCount };
enum class CalcAttributeType : int { kZonalMerimoal, kVelocity, kZTop, kCount };
enum class VisualizeType : int {kFixedLayer, kFixedDepth};

struct VisualizationSettings
{
    vec2 imageSize;
    vec2 LonRange;

    vec2 LatRange;
    
    vec2 DepthRange;
    double FixedLatitude;
    union
    {
        double FixedDepth;
        double FixedLayer;
    };
    CalcAttributeType CalcType = CalcAttributeType::kVelocity;
    CalcPositionType PositionType = CalcPositionType::kPoint;
    VisualizeType VisType;

    double TimeStep;
    VisualizationSettings() = default;

};

struct SamplingSettings
{
    vec2i sampleNumer;
    vec2 sampleLatitudeRange;
    vec2 sampleLongitudeRange;
    double sampleDepth;
};

#define ONE_SECOND  1
#define ONE_MINUTE  60
#define ONE_HOUR    60 * 60
#define ONE_DAY     60 * 60 * 24
#define ONE_MONTH   60 * 60 * 24 * 30
#define ONE_YEAR    60 * 60 * 24 * 30 * 12 

struct TrajectorySettings
{
    size_t deltaT;   // 相隔多少秒计算一次 新的位置
    size_t simulationDuration; //要模拟的总时长
    size_t recordT; // 相隔多少秒存储一次 新的位置
    float depth;
    std::string fileName;
};

class MPASOVisualizer
{
public:
    static void VisualizeFixedLayer(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, sycl::queue& sycl_Q);
    static void VisualizeFixedDepth(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, sycl::queue& sycl_Q);
    [[deprecated]] static void VisualizeFixedLatitude(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, sycl::queue& sycl_Q);
    static void GenerateSamplePoint(std::vector<CartesianCoord>& points, SamplingSettings* config);
    static void GenerateGaussianSpherePoints(std::vector<CartesianCoord>& points, SamplingSettings* config, int numPoints, double meanLat, double meanLon, double stdDev);
    static std::vector<CartesianCoord>  VisualizeTrajectory(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, sycl::queue& sycl_Q);

    static void VisualizeFixedLayer_TimeVarying(int width, int height, ImageBuffer<double>* img1, ImageBuffer<double>* img2, float time1, float time2, float time, sycl::queue& sycl_Q);
  
    static std::vector<CartesianCoord>  TEST_VisualizeTrajectory(std::vector<CartesianCoord>& points, TrajectorySettings* config, sycl::queue& sycl_Q);

};

