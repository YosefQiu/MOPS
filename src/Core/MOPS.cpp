#include "api/MOPS.h"
#include "Core/MOPSApp.h"
#include "Utils/Timer.hpp"
#include <memory>
#include <cstring>


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

    // ========================================================================
    // Timing API implementation
    // ========================================================================

    void MOPS_ResetTiming()
    {
        TimerManager::instance().reset();
    }

    void MOPS_PrintTimingSummary()
    {
        TimerManager::instance().printSummary();
    }

    void MOPS_PrintTimingDetailed()
    {
        TimerManager::instance().printDetailed();
    }

    double MOPS_GetCategoryTime(const char* category)
    {
        if (strcmp(category, "IO_Read") == 0)
            return TimerManager::instance().getCategoryTime(TimerCategory::IO_Read);
        else if (strcmp(category, "IO_Write") == 0)
            return TimerManager::instance().getCategoryTime(TimerCategory::IO_Write);
        else if (strcmp(category, "Preprocessing") == 0)
            return TimerManager::instance().getCategoryTime(TimerCategory::Preprocessing);
        else if (strcmp(category, "MemoryCopy") == 0)
            return TimerManager::instance().getCategoryTime(TimerCategory::MemoryCopy);
        else if (strcmp(category, "GPUKernel") == 0)
            return TimerManager::instance().getCategoryTime(TimerCategory::GPUKernel);
        else if (strcmp(category, "CPUCompute") == 0)
            return TimerManager::instance().getCategoryTime(TimerCategory::CPUCompute);
        else
            return TimerManager::instance().getCategoryTime(TimerCategory::Other);
    }

    double MOPS_GetTotalTime()
    {
        return TimerManager::instance().getTotalTime();
    }

}
