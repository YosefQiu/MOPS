/**
 * ============================================================================
 * testAbPts.cpp - MOPS Pathline Simulation Tutorial
 * ============================================================================
 *
 * CONFIGURATION GUIDE - TWO MODES
 * ============================================================================
 *
 * ┌─── Mode 1: Manual Particle [DEFAULT] ──────────────────────────────────┐
 * │  Manually specify single particle position and depth                   │
 * │                                                                         │
 * │  CONFIG:                                                                │
 * │    Keep commented: // #define USE_PARTICLE_FILE (line 62)              │
 * │                                                                         │
 * │  PARAMETERS (lines 70-72):                                              │
 * │    MANUAL_PARTICLE_LAT:   Latitude (degrees)                           │
 * │    MANUAL_PARTICLE_LON:   Longitude (degrees)                          │
 * │    MANUAL_PARTICLE_DEPTH: Depth (meters, positive downward)            │
 * │                                                                         │
 * │  CURRENT: 1 particle at 45°N, 160°W, 2m depth                          │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * ┌─── Mode 2: Load from NPY File ──────────────────────────────────────────┐
 * │  Load particles from 2D .npy file with lat/lon/depth                   │
 * │                                                                         │
 * │  CONFIG:                                                                │
 * │    Uncomment: #define USE_PARTICLE_FILE (line 62)                      │
 * │                                                                         │
 * │  PARAMETERS:                                                            │
 * │    PARTICLE_FILE_PATH:     .npy file path (line 63)                    │
 * │    MAX_PARTICLES_TO_LOAD:  Limit particle count (line 64)              │
 * │                                                                         │
 * │  NPY FORMAT: 2D array (num_particles, 3)                               │
 * │              Each row: [latitude, longitude, depth]                    │
 * │              Depth values from file are used directly                  │
 * │                                                                         │
 * │  EXAMPLE: Load up to 5000 particles from seeds file                    │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * ============================================================================
 */

#include "ggl.h"
#include "api/MOPS.h"
#include "Utils/cxxopts.hpp"
#include "Utils/Utils.hpp"
#include "Utils/YamlGen.hpp"
#include "Utils/GeoConverter.hpp"
#include "IO/VTKFileManager.hpp"
#include "IO/MPASOReader.h"


using namespace MOPS;

std::vector<CartesianCoord> lastPts_vec;
std::vector<float> lastDepths_vec;

// ============================================================================
// Particle Input Configuration
// ============================================================================
// Define USE_PARTICLE_FILE to load particles from a .npy file instead of generating them
// #define USE_PARTICLE_FILE  // COMMENTED OUT: Using manual particle setup instead
const char* PARTICLE_FILE_PATH = "/pscratch/sd/q/qiuyf/MOPS/TestData/seeds_45N_negative_random_0_005Sv_0.npy";
const int MAX_PARTICLES_TO_LOAD = 10;  // Start with 1000 particles for debugging


// ============================================================================
// Manual Particle Configuration
// ============================================================================
const double MANUAL_PARTICLE_LAT = 45.0;  // Latitude (degrees)
const double MANUAL_PARTICLE_LON = -160.0;  // Longitude (degrees)
const float MANUAL_PARTICLE_DEPTH = 2.0f;   // Depth (meters)

// ============================================================================
// Simulation Configuration
// ============================================================================
const char* YAML_CONFIG_PATH = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/test_ab_climatology.yaml";
const int TIMESTEP = 0;

// Time range configuration
const int START_YEAR = 10;
const int START_MONTH = 12;
const int END_YEAR = 10;
const int END_MONTH = 10;

// Trajectory settings
const MOPS::CalcDirection TRAJECTORY_DIRECTION = MOPS::CalcDirection::kBackward;
const MOPS::CalcMethodType INTEGRATION_METHOD = MOPS::CalcMethodType::kRK4;
const int DELTA_T_MINUTES = 10;        // Time step in minutes
const int RECORD_INTERVAL_HOURS = 6;   // Output interval in hours

void tutorial_pathLine(const std::string name_prefix, bool isFirstPts, int day_gap)
{
	std::vector<CartesianCoord> sample_points;
	std::vector<float> sample_depths;  
	
	if (isFirstPts)
	{
#ifdef USE_PARTICLE_FILE
		{
			// Load particle seeds from NPY file
			size_t num_particles = 0;
			auto particles = MPASOReader::loadParticleSeeds(PARTICLE_FILE_PATH, num_particles);
			if (particles.empty()) {
				std::cerr << "[ERROR] No particles loaded from file!" << std::endl;
				exit(-1);
			}

			// Limit number of particles to avoid memory issues
			size_t particles_to_load = std::min(num_particles, static_cast<size_t>(MAX_PARTICLES_TO_LOAD));

			sample_points.resize(particles_to_load);
			sample_depths.resize(particles_to_load);

			// Use lat/lon/depth from NPY file directly
			for (size_t p = 0; p < particles_to_load; p++) {
				auto& lld = particles[p];
				sample_points[p] = GeoConverter::convertLatLonDepthToXYZ(lld.lat, lld.lon, lld.depth);
				sample_depths[p] = static_cast<float>(lld.depth);
			}
		}
#else
		{
			// Create manual particle with specified position and depth
			sample_points.resize(1);
			sample_depths.resize(1);

			sample_points[0] = GeoConverter::convertLatLonDepthToXYZ(MANUAL_PARTICLE_LAT, MANUAL_PARTICLE_LON, MANUAL_PARTICLE_DEPTH);
			sample_depths[0] = MANUAL_PARTICLE_DEPTH;
		}
#endif
	}
	else
	{
		{
			if (lastPts_vec.size() != 0)
			{
				sample_points.resize(lastPts_vec.size());
				sample_depths.resize(lastPts_vec.size());
				for (auto idx = 0; idx < lastPts_vec.size(); idx++)
				{
					sample_points[idx] = CartesianCoord{lastPts_vec[idx].x(), lastPts_vec[idx].y(), lastPts_vec[idx].z()};
					sample_depths[idx] = lastDepths_vec[idx];
				}
			}
			else
			{
				std::cerr << "[ERROR]::No sample points in memory!" << std::endl;
				exit(-1);
			}
		}
	}
	
	MOPS::TrajectorySettings* traj_conf = new MOPS::TrajectorySettings;
	traj_conf->directionType = TRAJECTORY_DIRECTION;
	traj_conf->methodType = INTEGRATION_METHOD;
	traj_conf->particle_depths = sample_depths;  // Per-particle depths
	traj_conf->deltaT = ONE_MINUTE * DELTA_T_MINUTES;
	traj_conf->simulationDuration = std::abs(day_gap);
	traj_conf->recordT = ONE_HOUR * RECORD_INTERVAL_HOURS;
    auto direction_str = (traj_conf->directionType == MOPS::CalcDirection::kForward) ? "FORWARD" : "BACKWARD";

	auto tiltle = name_prefix + "_";
	traj_conf->fileName = tiltle + direction_str;

	// GPU Kernel
	std::vector<MOPS::TrajectoryLine> lines;
	{
		lines = MOPS::MOPS_RunPathLine(traj_conf, sample_points);
	}


	// save to vtp
	{
		MOPS::VTKFileManager::SaveTrajectoryLinesAsVTP(lines, traj_conf->fileName);
	}
	
	// save last pts to memory (and depths for per-particle mode)
	{
		lastPts_vec.clear();
		lastDepths_vec.clear();
		std::vector<vec3> last_pts;
		for (auto idx = 0; idx < lines.size(); idx++) 
		{
			const auto& pts = lines[idx].points;
			bool found = false;

			for (int i = static_cast<int>(pts.size()) - 1; i >= 0; --i) {
				const auto& p = pts[i];
				if (!(p.x() == 0.0 && p.y() == 0.0 && p.z() == 0.0)) {
					last_pts.push_back(p);
					found = true;
					break;
				}
			}

			if (!found) 
			{
				last_pts.push_back(pts.back());  // or CartesianCoord{0, 0, 0};
			}
		}
		for (size_t i = 0; i < last_pts.size(); i++) {
			const auto& p = last_pts[i];
			lastPts_vec.push_back(CartesianCoord{p.x(), p.y(), p.z()});

			// Save evolved depth from the last valid XYZ point (depth positive downward)
			const double earthRadius = 6371010.0;
			double r = std::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
			lastDepths_vec.push_back(static_cast<float>(earthRadius - r));
		}
	}
	
    delete traj_conf;
	traj_conf = nullptr;
}


void IO()
{
	auto mpasoGrid = std::make_shared<MOPS::MPASOGrid>();
    auto solFront = std::make_shared<MOPS::MPASOSolution>();
	auto solBack = std::make_shared<MOPS::MPASOSolution>();

    // Generate month pairs for simulation period
	auto year_pairs = MOPS_IO::make_backward_month_pairs(START_YEAR, START_MONTH, END_YEAR, END_MONTH);
    
	{
		#if defined(MOPS_USE_TBB) && (MOPS_USE_TBB == 1)
		MOPS::MOPS_Init("cpu");
		#else
		MOPS::MOPS_Init("gpu");
		#endif
    }

	// Use new MOPS timing system
	{
		mpasoGrid->initGrid(MOPS::MPASOReader::readGridData(YAML_CONFIG_PATH).get());
	}

	bool isFirst = true;

    

    for (const auto& p : year_pairs)
    {
		
        {
            solFront->initSolution(MOPS::MPASOReader::readSolData(YAML_CONFIG_PATH, p.first, TIMESTEP).get());
            solBack->initSolution(MOPS::MPASOReader::readSolData(YAML_CONFIG_PATH, p.second, TIMESTEP).get());
        }

        {
            solFront->addAttribute("temperature", MOPS::AttributeFormat::kFloat);
            solFront->addAttribute("salinity", MOPS::AttributeFormat::kFloat);
            solBack->addAttribute("temperature", MOPS::AttributeFormat::kFloat);
            solBack->addAttribute("salinity", MOPS::AttributeFormat::kFloat);
        }

        auto t1 =solFront->getTimeStamp();
        auto t2 = solBack->getTimeStamp();

		auto fileNamePrefix = "PathLine_" + std::to_string(toIntYMD(p.first)) + "_to_" + std::to_string(toIntYMD(p.second));

        {
            MOPS::MOPS_Begin();
            MOPS::MOPS_AddGridMesh(mpasoGrid);
            MOPS::MOPS_AddAttribute(solFront->getID(), solFront);
            MOPS::MOPS_AddAttribute(solBack->getID(), solBack);
            MOPS::MOPS_End();
        }
        
        {
            MOPS::MOPS_ActiveAttribute(solFront->getID(), solBack->getID());
        }

        tutorial_pathLine(fileNamePrefix, isFirst, getTimeGapinSecond(t2.c_str(), t1.c_str()));
        isFirst = false;
    }

}

int main()
{
	// Reset timing before starting (optional, starts fresh)
	MOPS::MOPS_ResetTiming();

    IO();
	
	// Print timing summary - shows time spent in each category
	// (IO Read, IO Write, Preprocessing, GPU Kernel, etc.)
	MOPS::MOPS_PrintTimingSummary();
	
	// Optionally print detailed breakdown of each operation
	// MOPS::MOPS_PrintTimingDetailed();

	return 0;
}
