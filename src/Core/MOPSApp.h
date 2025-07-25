#pragma once
#include "ggl.h"
#include "Core/MPASOGrid.h"
#include "Core/MPASOSolution.h"
#include "Core/MPASOField.h"
#include "SYCL/ImageBuffer.hpp"
#include "Core/MPASOVisualizer.h"



namespace MOPS
{

    struct MOPSConfig 
    {
        std::string inputYaml;
        std::string dataPrefix;
        int month = 1;
        int timestep = 0;
        std::string dataDir;
    };

    enum class MOPSState { Uninitialized, Configuring, Ready};

    class MOPSApp
    {
    public:
        MOPSApp(){}
        ~MOPSApp(){}
        void init(const char* device);
        void addGrid(std::shared_ptr<MPASOGrid> grid);
        void addSol(int timestep, std::shared_ptr<MPASOSolution> sol);
        void addField();
        void activeAttribute(int timestep);
        std::vector<TrajectoryLine> runStreamLine(TrajectorySettings* config, std::vector<CartesianCoord>& sample_points);
        std::vector<TrajectoryLine> runPathLine(TrajectorySettings* config, std::vector<CartesianCoord>& sample_points, std::vector<int>& timestep_vec);
        std::vector<ImageBuffer<double>> runRemapping(VisualizationSettings* config);
        void generateSamplePoints(SamplingSettings* config, std::vector<CartesianCoord>& sample_points);
        void generateSamplePointsAtCenter(SamplingSettings* config, std::vector<CartesianCoord>& sample_points);
    public:
        MOPSState getState() const { return mState; }
        void setState(MOPSState state) { mState = state; }
        bool checkAttribute() const;
    private:
        MOPSState mState = MOPSState::Uninitialized;
        std::string mDataDir;
        sycl::queue mSYCLQueue;
        std::shared_ptr<MPASOGrid> mpasoGrid = nullptr;
        std::map<int, std::shared_ptr<MPASOSolution>> mpasoAttributeMap;
        std::shared_ptr<MPASOField> mpasoField = nullptr;

       
        
    };
}
