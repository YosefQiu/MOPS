#pragma once
#include "ggl.h"
#include "SYCL/ImageBuffer.hpp"
#include "Utils/GeoConverter.hpp"
#include "Core/MPASOField.h"
#include <vector>

namespace MOPS
{
    enum class CalcPositionType : int { kCenter, kVertx, kPoint, kCount };
    enum class CalcAttributeType : int { kZonalMerimoal, kVelocity, kZTop, kTemperature, kSalinity, kAll, kCount };
    enum class CalcDirection:int {kForward, kBackward, kCount};
    enum class CalcMethodType:int {kRK4, kEuler, kCount};
    enum class VisualizeType : int {kFixedLayer, kFixedDepth};
    
    enum class SaveType : int {kVTI, kPNG, kNone, kCount};

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
        int tile_index;
        CalcAttributeType CalcType = CalcAttributeType::kZonalMerimoal;
        CalcPositionType PositionType = CalcPositionType::kPoint;
        VisualizeType VisType = VisualizeType::kFixedDepth;
        SaveType SaveType = SaveType::kNone;
        double TimeStep;
        VisualizationSettings() = default;

    };

    struct SamplingSettings 
    {
    public:
        SamplingSettings() = default;

        void setSampleRange(const vec2i& number) { sampleRange = number; }
        void setGeoBox(const vec2& latRange, const vec2& lonRange) { sampleLatitudeRange = latRange; sampleLongitudeRange = lonRange; }
        void setDepth(double depth) { sampleDepth = depth; }
        void setSamplingRegion(const vec2i& number, const vec2& latRange, const vec2& lonRange, double depth) { sampleRange = number; sampleLatitudeRange = latRange; sampleLongitudeRange = lonRange; sampleDepth = depth; }
        void atCellCenter(bool bIsAtCellCenter) { bAtCellCenter = bIsAtCellCenter;}
        vec2i getSampleRange() const { return sampleRange; }
        vec2 getLatitudeRange() const { return sampleLatitudeRange; }
        vec2 getLongitudeRange() const { return sampleLongitudeRange; }
        bool isAtCellCenter() const { return bAtCellCenter; }
        double getDepth() const { return sampleDepth; }

        

    private:
        vec2i sampleRange;
        vec2 sampleLatitudeRange;
        vec2 sampleLongitudeRange;
        double sampleDepth = 0.0;
        bool bAtCellCenter = false;
    };

    struct TrajectoryLine
    {
        int lineID;
        std::vector<CartesianCoord> points;
        std::vector<CartesianCoord> velocity;
        std::vector<double> temperature;
        std::vector<double> salinity;
        CartesianCoord lastPoint;
        double duration;
        double timestamp;
        double depth;
    };

    #define ONE_SECOND  1
    #define ONE_MINUTE  60
    #define ONE_HOUR    60 * 60
    #define ONE_DAY     60 * 60 * 24
    #define ONE_MONTH   60 * 60 * 24 * 30
    #define ONE_YEAR    60 * 60 * 24 * 30 * 12 

    struct TrajectorySettings
    {
        size_t deltaT;              // How many seconds to calculate the new position?
        size_t simulationDuration;  // Total simulation duration in seconds
        size_t recordT;             // How many seconds between recording new positions
        float depth;                // Default depth if particle_depths is empty
        std::vector<float> particle_depths;  // NEW: Per-particle depth (meters, positive downward)
        std::string fileName;
        CalcDirection directionType = CalcDirection::kForward;
        CalcMethodType methodType = CalcMethodType::kEuler;
        
        // Helper: check if per-particle depths are enabled
        bool hasPerParticleDepths() const { return !particle_depths.empty(); }
    };

    class MPASOVisualizer
    {
    public:
        static void VisualizeFixedLayer(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, sycl::queue& sycl_Q);
        static void VisualizeFixedDepth(MPASOField* mpasoF, VisualizationSettings* config, std::vector<ImageBuffer<double>>& img_vec, sycl::queue& sycl_Q);
        static void VisualizeFixedDepth(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, sycl::queue& sycl_Q);
        static void VisualizeFixedLatitude(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, sycl::queue& sycl_Q);
        static void GenerateSamplePoint(std::vector<CartesianCoord>& points, SamplingSettings* config);
        static void GenerateGaussianSpherePoints(std::vector<CartesianCoord>& points, SamplingSettings* config, int numPoints, double meanLat, double meanLon, double stdDev);
        static void GenerateSamplePointAtCenter(std::vector<CartesianCoord>& points, SamplingSettings* config);
        
        
        static std::vector<TrajectoryLine> StreamLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, sycl::queue& sycl_Q);
        static std::vector<TrajectoryLine> PathLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, sycl::queue& sycl_Q);



        static void VisualizeFixedLayer_TimeVarying(int width, int height, ImageBuffer<double>* img1, ImageBuffer<double>* img2, float time1, float time2, float time, sycl::queue& sycl_Q);
    
        static std::vector<CartesianCoord>  TEST_VisualizeTrajectory(std::vector<CartesianCoord>& points, TrajectorySettings* config, sycl::queue& sycl_Q);

        static std::vector<TrajectoryLine> removeNaNTrajectoriesAndReindex(std::vector<TrajectoryLine>& trajectory_lines);
    };


}
