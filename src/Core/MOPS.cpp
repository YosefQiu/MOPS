#include "api/MOPS.h"
#include "Core/MOPSApp.h"
#include <memory>


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

    void MOPS_AddAttribute(int solID, std::shared_ptr<MPASOSolution> sol)
    {
        app.addSol(solID, sol);
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

    void MOPS_ActiveAttribute(int t1, std::optional<int> t2)
    {
        app.activeAttribute(t1, t2);
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

    std::vector<TrajectoryLine> MOPS_RunPathLine(TrajectorySettings* config, std::vector<CartesianCoord>& sample_points)
    {
        auto lines = app.runPathLine(config, sample_points);
        return lines;
    }

    void MOPS_GenerateSamplePoints(SamplingSettings* config, std::vector<CartesianCoord>& sample_points)
    {
        if (config->isAtCellCenter() == false)
            app.generateSamplePoints(config, sample_points);
        else if (config->isAtCellCenter() == true)
            app.generateSamplePointsAtCenter(config, sample_points);
    }
    

    std::shared_ptr<MPASOField> MOPS_GetFieldSnapshots()
    {
        return app.getField();
    }
   

}
