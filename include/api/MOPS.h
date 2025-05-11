#pragma once
#include "ggl.h"
#include "MOPSApp.h"
#include <vector>


namespace MOPS
{
    extern MOPSApp app;
    
    /**
     * @brief Initialize the MOPS environment using provided configuration.
     *
     * This function sets up core internal states (e.g., SYCL queue, stream parser)
     * and prepares for subsequent data ingestion (grid, solution).
     *
     * @param config Configuration structure specifying input YAML path, data prefix, and timestep.
     *        - `config.inputYaml`: Path to YAML metadata describing the dataset.
     *        - `config.dataPrefix`: Prefix path for locating NetCDF files.
     *        - `config.timestep`: Integer timestep to be visualized.
     */
	void MOPS_Init(const char* device = "gpu");

    /**
     * @brief Begin the configuration stage for adding simulation data.
     *
     * This marks the beginning of the data ingestion phase. You must call this
     * before adding any grid or solution data. It resets internal buffers.
     *
     * Typical usage:
     * @code
     * MOPS::MOPS_Begin();
     * MOPS::MOPS_AddGrid();
     * MOPS::MOPS_AddSolution(...);
     * MOPS::MOPS_End();
     * @endcode
     */
    void MOPS_Begin();

    /**
     * @brief Load and initialize MPAS-O grid data.
     *
     * This function constructs the spatial grid layout from metadata defined
     * in the YAML or inferred via internal stream. It builds the KD-tree and
     * prepares geometric structures.
     *
     * Must be called between `MOPS_Begin()` and `MOPS_End()`.
     */
    void MOPS_AddGridMesh(std::shared_ptr<MPASOGrid> grid);
    
    /**
     * @brief Load MPAS-O velocity solution data for the current timestep.
     *
     * This function reads the velocity fields from NetCDF files and initializes
     * associated cell-centered and vertex-based quantities.
     *
     * @param config The configuration containing current timestep (in-place mutable).
     *        - `config.timestep`: The timestep for which the solution is extracted.
     *
     * Must be called after `MOPS_AddGrid()` and before `MOPS_End()`.
     */
    void MOPS_AddAttribute(int timestep, std::shared_ptr<MPASOSolution> sol);

    /**
     * @brief Finalize the ingestion stage and build simulation fields.
     *
     * This function synthesizes the field data by combining grid and solution states.
     * After calling this, the system is ready for rendering or sampling.
     */
    void MOPS_End();

    /**
     * @brief Activate an attribute for the given timestep.
     *
     * This function sets the active attribute for the specified timestep.
     * It allows you to switch between different attributes at runtime.
     *
     * @param timestep The timestep to activate.
     */
    void MOPS_ActiveAttribute(int timestep);

    /**
     * @brief Perform remapping-based rendering for fixed-depth or fixed-layer visualizations.
     *
     * This function creates VTI output using field remapping for a user-defined
     * depth slice. The output image is written to disk automatically.
     *
     * @param config Visualization settings (e.g., lat/lon range, depth).
     */
    std::vector<ImageBuffer<double>> MOPS_RunRemapping(VisualizationSettings* config);

    /**
     * @brief Execute trajectory tracing based on previously defined sample points.
     *
     * This function integrates particles through the velocity field using
     * a fixed step or time-varying update.
     *
     * @param config Simulation duration, time step, and output parameters.
     */
    std::vector<TrajectoryLine> MOPS_RunStreamLine(TrajectorySettings* config, std::vector<CartesianCoord>& sample_points);
    

    std::vector<TrajectoryLine> MOPS_RunPathLine(TrajectorySettings* config, std::vector<CartesianCoord>& sample_points, std::vector<int>& timestep_vec);

    /**
     * @brief Generate sample points used for trajectory or probing tasks.
     *
     * This function samples points uniformly or by Gaussian distribution within
     * a lat/lon/depth region. These points will be used for `MOPS_RunStreamLine()`.
     *
     * @param config Sampling domain settings (region, resolution, etc.).
     */
    void MOPS_GenerateSamplePoints(SamplingSettings* config, std::vector<CartesianCoord>& sample_points);

}
