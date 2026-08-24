/**
 * ============================================================================
 * streamLine.cpp - MOPS Streamline Simulation Tutorial
 * ============================================================================
 *
 * CONFIGURATION GUIDE - TWO MODES
 * ============================================================================
 *
 * ┌─── Mode 1: Manual Points [DEFAULT] ─────────────────────────────────────┐
 * │  Manually specify starting points for streamlines                      │
 * │                                                                         │
 * │  CONFIG:                                                                │
 * │    Keep commented: // #define USE_GRID_SAMPLING (line 53)              │
 * │                                                                         │
 * │  PARAMETERS (lines 62-64):                                              │
 * │    MANUAL_POINT_X: X coordinate (meters)                               │
 * │    MANUAL_POINT_Y: Y coordinate (meters)                               │
 * │    MANUAL_POINT_Z: Z coordinate (meters)                               │
 * │                                                                         │
 * │  CURRENT: 1 point at (1908930, -5174124, 3189701)                      │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * ┌─── Mode 2: Grid Sampling ────────────────────────────────────────────────┐
 * │  Generate points by sampling from a geographic box                     │
 * │                                                                         │
 * │  CONFIG:                                                                │
 * │    Uncomment: #define USE_GRID_SAMPLING (line 53)                      │
 * │                                                                         │
 * │  PARAMETERS (lines 67-72):                                              │
 * │    SAMPLE_LAT_MIN/MAX:  Latitude range (degrees)                       │
 * │    SAMPLE_LON_MIN/MAX:  Longitude range (degrees)                      │
 * │    SAMPLE_GRID_ROWS:    Number of rows in sampling grid               │
 * │    SAMPLE_GRID_COLS:    Number of columns in sampling grid            │
 * │    SAMPLE_AT_CENTER:    Sample at cell center (true) or vertex (false)│
 * │                                                                         │
 * │  EXAMPLE: 2x2 grid in box [35°N-45°N, 60°W-15°W]                      │
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
// Sampling Mode Configuration
// ============================================================================
// Define USE_GRID_SAMPLING to generate points from a geographic grid
// #define USE_GRID_SAMPLING  // COMMENTED OUT: Using manual points instead

// ============================================================================
// Manual Points Configuration
// ============================================================================
const double MANUAL_POINT_X = 1908930.101867;
const double MANUAL_POINT_Y = -5174124.236251;
const double MANUAL_POINT_Z = 3189701.032088;

// ============================================================================
// Grid Sampling Configuration
// ============================================================================
const double SAMPLE_LAT_MIN = 35.0;   // Minimum latitude (degrees)
const double SAMPLE_LAT_MAX = 45.0;   // Maximum latitude (degrees)
const double SAMPLE_LON_MIN = -60.0;  // Minimum longitude (degrees)
const double SAMPLE_LON_MAX = -15.0;  // Maximum longitude (degrees)
const int SAMPLE_GRID_ROWS = 2;       // Number of rows in sampling grid
const int SAMPLE_GRID_COLS = 2;       // Number of columns in sampling grid
const bool SAMPLE_AT_CENTER = false;  // Sample at cell center (true) or vertex (false)

// ============================================================================
// Simulation Configuration
// ============================================================================
const char* YAML_CONFIG_PATH = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/test.yaml";
const int YEAR = 15;
const int MONTH = 1;
const int DAY = 1;
const int TIMESTEP = 0;

// ============================================================================
// Depth Configuration
// ============================================================================
const float FIXED_DEPTH = 10.0f;  // Depth (meters, positive downward)

// ============================================================================
// Trajectory Settings
// ============================================================================
const MOPS::CalcDirection TRAJECTORY_DIRECTION = MOPS::CalcDirection::kForward;
const int DELTA_T_MINUTES = 1;          // Time step in minutes
const int SIMULATION_YEARS = 2;         // Simulation duration in years
const int RECORD_INTERVAL_MINUTES = 6;  // Output interval in minutes

void tutorial_streamLine(const std::string name_prefix, float fixed_depth, MOPS_IO::YMD simulation)
{
	std::vector<CartesianCoord> sample_points;

#ifdef USE_GRID_SAMPLING
	{
		std::cout << "== Generating sample points from grid ==" << std::endl;
		MOPS::SamplingSettings* sampling_conf = new MOPS::SamplingSettings();
		sampling_conf->setSampleRange(vec2i{SAMPLE_GRID_ROWS, SAMPLE_GRID_COLS});
		sampling_conf->setGeoBox(vec2{SAMPLE_LAT_MIN, SAMPLE_LAT_MAX}, vec2{SAMPLE_LON_MIN, SAMPLE_LON_MAX});
		sampling_conf->atCellCenter(SAMPLE_AT_CENTER);
		sampling_conf->setDepth(fixed_depth);
		MOPS::MOPS_GenerateSamplePoints(sampling_conf, sample_points);
		delete sampling_conf;
		std::cout << "Generated " << sample_points.size() << " sample points" << std::endl;
	}
#else
	{
		std::cout << "== Using manual sample points ==" << std::endl;
		sample_points.resize(1);
		sample_points[0] = CartesianCoord{MANUAL_POINT_X, MANUAL_POINT_Y, MANUAL_POINT_Z};
		std::cout << "Using 1 manual point: (" << MANUAL_POINT_X << ", "
		          << MANUAL_POINT_Y << ", " << MANUAL_POINT_Z << ")" << std::endl;
	}
#endif

	MOPS::TrajectorySettings* traj_conf = new MOPS::TrajectorySettings;
	traj_conf->directionType = TRAJECTORY_DIRECTION;
	traj_conf->depth = fixed_depth;
	traj_conf->deltaT = ONE_MINUTE * DELTA_T_MINUTES;
	traj_conf->simulationDuration = ONE_YEAR * SIMULATION_YEARS;
	traj_conf->recordT = ONE_MINUTE * RECORD_INTERVAL_MINUTES;
	auto direction_str = (traj_conf->directionType == MOPS::CalcDirection::kForward) ? "FORWARD" : "BACKWARD";

	auto title = name_prefix + "_";
	traj_conf->fileName = title + direction_str;

	std::cout << "== Running streamline simulation ==" << std::endl;

	auto lines = MOPS::MOPS_RunStreamLine(traj_conf, sample_points);
	std::cout << "First streamline length: " << lines[0].points.size() << " points" << std::endl;

	MOPS::VTKFileManager::SaveTrajectoryLinesAsVTP(lines, traj_conf->fileName);
	std::cout << "Saved streamlines to: " << traj_conf->fileName << ".vtp" << std::endl;

	delete traj_conf;
}

void IO()
{
	std::string timeStamp = MOPS_IO::make_date_tag(YEAR, MONTH, DAY);
	MOPS_IO::YMD simulation = {SIMULATION_YEARS, 0, 0};  // YMD data structure

	auto fileNamePrefix = [&]() {
		auto end = MOPS_IO::fromStringYMD(std::to_string(toIntYMD(timeStamp))) + simulation;
		return std::string("StreamLine_") +
		       std::to_string(toIntYMD(timeStamp)) +
		       "_to_" +
		       std::to_string(toIntYMD(MOPS_IO::make_date_tag(end.year, end.month, end.day)));
	}();

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

	tutorial_streamLine(fileNamePrefix, FIXED_DEPTH, simulation);
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
