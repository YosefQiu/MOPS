/**
 * ============================================================================
 * reMapping.cpp - MPAS-O Remapping Tutorial
 * ============================================================================
 *
 * CONFIGURATION GUIDE - TWO MODES
 * ============================================================================
 *
 * ┌─── Mode 1: Single Depth [DEFAULT] ──────────────────────────────────────┐
 * │  Remap one depth level for quick testing                               │
 * │                                                                         │
 * │  CONFIG:                                                                │
 * │    Keep commented: // #define PROCESS_MULTIPLE_DEPTHS (line 48)        │
 * │                                                                         │
 * │  PARAMETERS (line 57):                                                  │
 * │    SINGLE_DEPTH: Depth to process (meters, positive downward)          │
 * │                                                                         │
 * │  CURRENT: Process depth 10m                                            │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * ┌─── Mode 2: Multiple Depths ──────────────────────────────────────────────┐
 * │  Process multiple depth levels in a loop                               │
 * │                                                                         │
 * │  CONFIG:                                                                │
 * │    Uncomment: #define PROCESS_MULTIPLE_DEPTHS (line 48)                │
 * │                                                                         │
 * │  PARAMETERS (lines 60-62):                                              │
 * │    DEPTH_START: Starting depth (meters)                                │
 * │    DEPTH_END:   Ending depth (meters)                                  │
 * │    DEPTH_STEP:  Depth increment (meters)                               │
 * │                                                                         │
 * │  EXAMPLE: Process from 0m to 100m with 10m increments                  │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * ============================================================================
 */

#include "ggl.h"
#include "api/MOPS.h"
#include "Utils/cxxopts.hpp"
#include "Utils/Utils.hpp"
#include "Utils/YamlGen.hpp"
#include "IO/VTKFileManager.hpp"
#include "IO/MPASOReader.h"
#include "Common/ImageBuffer.hpp"
#include <string>

// ============================================================================
// Processing Mode Configuration
// ============================================================================
// Define PROCESS_MULTIPLE_DEPTHS to process a range of depths
// #define PROCESS_MULTIPLE_DEPTHS  // COMMENTED OUT: Using single depth mode

// ============================================================================
// Single Depth Configuration
// ============================================================================
const float SINGLE_DEPTH = 10.0f;  // Depth to process (meters)

// ============================================================================
// Multiple Depths Configuration
// ============================================================================
const float DEPTH_START = 0.0f;    // Starting depth (meters)
const float DEPTH_END = 100.0f;    // Ending depth (meters)
const float DEPTH_STEP = 10.0f;    // Depth increment (meters)

// ============================================================================
// Simulation Configuration
// ============================================================================
const char* YAML_CONFIG_PATH = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/test.yaml";
const int YEAR = 15;
const int MONTH = 1;
const int DAY = 1;
const int TIMESTEP = 0;

// ============================================================================
// Grid Configuration
// ============================================================================
const int GRID_WIDTH = 3601;   // Longitude resolution (pixels, 0.1° resolution)
const int GRID_HEIGHT = 1801;  // Latitude resolution (pixels, 0.1° resolution)

// ============================================================================
// Output Configuration
// ============================================================================
#if MOPS_VTK
const MOPS::SaveType OUTPUT_FORMAT = MOPS::SaveType::kVTI;  // VTK Image format
#else
const MOPS::SaveType OUTPUT_FORMAT = MOPS::SaveType::kPNG;  // PNG images
#endif

void tutorial_reMapping(float fixed_depth)
{
	MOPS::VisualizationSettings* config = new MOPS::VisualizationSettings();
	config->imageSize = vec2{static_cast<double>(GRID_WIDTH), static_cast<double>(GRID_HEIGHT)};
	config->LatRange = vec2{-90.0, 90.0};
	config->LonRange = vec2{-180.0, 180.0};
	config->FixedDepth = fixed_depth;
	config->TimeStep = TIMESTEP;
	config->saveType = OUTPUT_FORMAT;

	std::cout << "== Remapping at depth " << fixed_depth << "m ==" << std::endl;

	auto img_vec = MOPS::MOPS_RunRemapping(config);

#if MOPS_VTK
	std::string str = "";
#if MOPS_MPI == 1
	str += "rank_" + std::to_string(rank_id) + "_";
#endif
	str += "timestep_" + std::to_string(config->TimeStep) + "_";
	str += "depth_" + std::to_string(static_cast<int>(fixed_depth)) + "m_";
	str += "tile_" + std::to_string(config->tile_index);
	std::vector<std::string> names = {
		"E: Zonal Velocity", "N: Meridional Velocity", "Velocity Magnitude",
		"Temperature", "Salinity", "None"
	};
#endif

	if (config->saveType == MOPS::SaveType::kVTI)
	{
#if MOPS_VTK
		MOPS::VTKFileManager::SaveVTI(img_vec, config, names, str);
		std::cout << "Saved VTI: " << str << std::endl;
#endif
	}
	else if (config->saveType == MOPS::SaveType::kPNG)
	{
		for (int i = 0; i < img_vec.size(); ++i)
		{
			for (int ch = 0; ch < 3; ++ch)  // channel 0,1,2
			{
#if MOPS_MPI
				std::string filename = "rank_" + std::to_string(rank_id) +
				                       "_depth_" + std::to_string(static_cast<int>(fixed_depth)) + "m" +
				                       "_output_" + std::to_string(i) + "_ch" + std::to_string(ch) + ".png";
#else
				std::string filename = "depth_" + std::to_string(static_cast<int>(fixed_depth)) + "m" +
				                       "_output_" + std::to_string(i) + "_ch" + std::to_string(ch) + ".png";
#endif
				MOPS::SaveToPNG(img_vec[i], filename, ch);
			}
		}
		std::cout << "Saved " << (img_vec.size() * 3) << " PNG files" << std::endl;
	}

	delete config;
}

void IO()
{
	std::string timeStamp = MOPS_IO::make_date_tag(YEAR, MONTH, DAY);

	auto mpasoGrid = std::make_shared<MOPS::MPASOGrid>();
	auto solFront = std::make_shared<MOPS::MPASOSolution>();

	solFront->initSolution(MOPS::MPASOReader::readSolData(YAML_CONFIG_PATH, timeStamp, TIMESTEP).get());
	solFront->addAttribute("temperature", MOPS::AttributeFormat::kFloat);
	solFront->addAttribute("salinity", MOPS::AttributeFormat::kFloat);

	mpasoGrid->initGrid(MOPS::MPASOReader::readGridData(YAML_CONFIG_PATH).get());

	#if defined(MOPS_USE_TBB) && (MOPS_USE_TBB == 1)
	MOPS::MOPS_Init("cpu");
	#else
	MOPS::MOPS_Init("gpu");
	#endif

	MOPS::MOPS_Begin();
	MOPS::MOPS_AddGridMesh(mpasoGrid);
	MOPS::MOPS_AddAttribute(solFront->getID(), solFront);
	MOPS::MOPS_End();

	MOPS::MOPS_ActiveAttribute(solFront->getID());

#ifdef PROCESS_MULTIPLE_DEPTHS
	{
		std::cout << "== Start remapping across multiple depths ==" << std::endl;
		for (float depth = DEPTH_START; depth <= DEPTH_END; depth += DEPTH_STEP) {
			tutorial_reMapping(depth);
		}
		std::cout << "== All remapping complete! ==" << std::endl;
	}
#else
	{
		std::cout << "== Processing single depth: " << SINGLE_DEPTH << "m ==" << std::endl;
		tutorial_reMapping(SINGLE_DEPTH);
		std::cout << "== Remapping complete! ==" << std::endl;
	}
#endif
}

int main()
{
	// Reset timing before starting (optional)
	MOPS::MOPS_ResetTiming();

	IO();

	// Print timing summary
	MOPS::MOPS_PrintTimingSummary();

	return 0;
}
