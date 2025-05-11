#include "api/MOPS.h"
#include "MOPSApp.h"
#include <vector>


namespace MOPS 
{
    MOPSApp app;
    void MOPS_Init(const char* device) 
    {
        app.init(device);
    }

    void MOPS_AddGridMesh(std::shared_ptr<MPASOGrid> grid)
    {
        app.addGrid(grid);
    }

    void MOPS_AddAttribute(int timestep, std::shared_ptr<MPASOSolution> sol)
    {
        app.addSol(timestep, sol);
    }

    void MOPS_Begin()
    {
        app.setState(MOPSState::Configuring);
    }

    void MOPS_End()
    {
        if (app.getState() != MOPSState::Configuring)
        {
            std::cerr << " [ MOPS is not configuring ]\n";
            exit(1);
        }
        if (!app.checkAttribute())
        {
            std::cerr << " [ MOPS is not configured ]\n";
            exit(1);
        }
        app.setState(MOPSState::Ready);

        app.addField();
    }

    void MOPS_ActiveAttribute(int timestep)
    {
        app.activeAttribute(timestep);
    }

    std::vector<ImageBuffer<double>> MOPS_RunRemapping(VisualizationSettings* config)
    {
        auto img_vec = app.runRemapping(config);
        

        return img_vec;
    }

    std::vector<TrajectoryLine> MOPS_RunStreamLine(TrajectorySettings* config, std::vector<CartesianCoord>& sample_points)
    {
        auto lines = app.runStreamLine(config, sample_points);
        return lines;
    }

    std::vector<TrajectoryLine> MOPS_RunPathLine(TrajectorySettings* config, std::vector<CartesianCoord>& sample_points, std::vector<int>& timestep_vec)
    {
        auto lines = app.runPathLine(config, sample_points, timestep_vec);
        return lines;
    }

    void MOPS_GenerateSamplePoints(SamplingSettings* config, std::vector<CartesianCoord>& sample_points)
    {
        app.generateSamplePoints(config, sample_points);
    }
    

    
   

}
