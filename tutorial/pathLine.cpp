#include "ggl.h"
#include "api/MOPS.h"
#include "Utils/cxxopts.hpp"
#include "Utils/Utils.hpp"
#include "Utils/YamlGen.hpp"
#include "IO/VTKFileManager.hpp"
#include "IO/MPASOReader.h"

using namespace MOPS;

// ============================================================================
// Depth Mode Configuration
// ============================================================================
// Define USE_PER_PARTICLE_DEPTH to enable per-particle depth tracking.
// When defined: each particle uses its own depth extracted from XYZ coordinates.
// When NOT defined: all particles use the same fixed_depth value.
// ============================================================================
#define USE_PER_PARTICLE_DEPTH  // Comment out this line for uniform depth mode



struct LatLonDepth {
    double lat;    // degrees
    double lon;    // degrees
    double depth;  // meters (positive downward)
};


// helper functions
LatLonDepth xyz_to_lat_lon_depth(double x, double y, double z)
{
	constexpr double EARTH_RADIUS2 = 6371000.0; // meters
    LatLonDepth out;
    out.lon = std::atan2(y, x) * 180.0 / M_PI;
    double r = std::sqrt(x*x + y*y + z*z);
    out.lat = std::asin(z / r) * 180.0 / M_PI;
    out.depth = EARTH_RADIUS2 - r;

    return out;
}

CartesianCoord lat_lon_depth_to_xyz(double lat_deg,
                                    double lon_deg,
                                    double depth)
{
	constexpr double EARTH_RADIUS2 = 6371000.0; // meters
    CartesianCoord out;
    double r = EARTH_RADIUS2 - depth;
    double lat = lat_deg * M_PI / 180.0;
    double lon = lon_deg * M_PI / 180.0;
    out.x() = r * std::cos(lat) * std::cos(lon);
    out.y() = r * std::cos(lat) * std::sin(lon);
    out.z() = r * std::sin(lat);
    return out;
}

CartesianCoord make_same_lat_depth_diff_lon(const CartesianCoord& p,
                                            double delta_lon_deg)
{
    // 1. XYZ → lat/lon/depth
    auto lld = xyz_to_lat_lon_depth(p.x(), p.y(), p.z());

    // 2. change lon
    lld.lon += delta_lon_deg;

    if (lld.lon > 180.0)  lld.lon -= 360.0;
    if (lld.lon < -180.0) lld.lon += 360.0;

    // 3. to xyz
    return lat_lon_depth_to_xyz(lld.lat, lld.lon, lld.depth);
}

int total_particles = 0;
std::vector<CartesianCoord> lastPts_vec;
std::vector<float> lastDepths_vec;  
float fixed_depth = 10.261f;

// ============================================================================
// Per-Particle Depth Configuration (only used when USE_PER_PARTICLE_DEPTH is defined)
// ============================================================================
// Define the depth range for particles. Depths will be linearly distributed.
float depth_min = 10.0f;    // Minimum depth (meters)
float depth_max = 500.0f;   // Maximum depth (meters)
// ============================================================================

void tutorial_pathLine(const std::string name_prefix, float fixed_depth, bool isFirstPts, int day_gap)
{
	std::vector<CartesianCoord> sample_points;
	std::vector<float> sample_depths;  
	
	if (isFirstPts)
	{
		Debug("== generate sample points ==");
		{
			MOPS::SamplingSettings* sampling_conf = new MOPS::SamplingSettings();
			sampling_conf->setSampleRange(vec2i(10, 10));
			sampling_conf->setGeoBox(vec2(35.0, 45.0),  vec2(-60.0, -15.0));
			sampling_conf->atCellCenter(false);
			sampling_conf->setDepth(fixed_depth);
			MOPS::MOPS_GenerateSamplePoints(sampling_conf, sample_points);
			delete sampling_conf;
			sampling_conf = nullptr;
		}
		total_particles = static_cast<int>(sample_points.size());
		
#ifdef USE_PER_PARTICLE_DEPTH
		// Per-particle depth mode: assign different depths to particles
		// The sample_points are generated at fixed_depth, but we will override
		// the depth used in GPU kernel with linearly distributed depths.
		sample_depths.resize(sample_points.size());
		for (size_t i = 0; i < sample_points.size(); i++) {
			// Linearly interpolate depth from depth_min to depth_max
			float t = (sample_points.size() > 1) 
			        ? static_cast<float>(i) / static_cast<float>(sample_points.size() - 1)
			        : 0.0f;
			sample_depths[i] = depth_min + t * (depth_max - depth_min);
			
			// Also update the XYZ coordinates to match the new depth
			auto lld = xyz_to_lat_lon_depth(sample_points[i].x(), sample_points[i].y(), sample_points[i].z());
			sample_points[i] = lat_lon_depth_to_xyz(lld.lat, lld.lon, sample_depths[i]);
		}
		Debug("[PER-PARTICLE DEPTH] Assigned depths from %.2f to %.2f meters (%zu particles)", 
			depth_min, depth_max, sample_points.size());
#else
		// Uniform depth mode: all particles use fixed_depth
		Debug("[UNIFORM DEPTH] All particles using depth: %.2f meters", fixed_depth);
#endif
		// sample_points.clear();
		// sample_points.resize(1);
        // sample_points[0] = CartesianCoord{ 4472513.01895255, -293143.99074839, 4521395.00861939}; 
		// for (auto i = 1; i < 15; i++)
		// {
		// 	sample_points[i] = make_same_lat_depth_diff_lon(sample_points[0], 2.0 * i);
		// }
	}
	else
	{
		Debug("== load sample points from memory ==");
		{
			if (lastPts_vec.size() != 0)
			{
				sample_points.resize(lastPts_vec.size());
#ifdef USE_PER_PARTICLE_DEPTH
				sample_depths.resize(lastPts_vec.size());
#endif
				for (auto idx = 0; idx < lastPts_vec.size(); idx++)
				{
					sample_points[idx] = CartesianCoord{lastPts_vec[idx].x(), lastPts_vec[idx].y(), lastPts_vec[idx].z()};
#ifdef USE_PER_PARTICLE_DEPTH
					// Per-particle depth: restore from saved depths
					sample_depths[idx] = lastDepths_vec[idx];
#endif
				}
#ifdef USE_PER_PARTICLE_DEPTH
				Debug("[PER-PARTICLE DEPTH] Restored %zu particles with saved depths", sample_points.size());
#endif
			}
			else
			{
				std::cerr << "[ERROR]::No sample points in memory!" << std::endl;
				exit(-1);
			}
		}
	}
	
	MOPS::TrajectorySettings* traj_conf = new MOPS::TrajectorySettings;
	traj_conf->directionType = MOPS::CalcDirection::kForward;
	traj_conf->methodType = MOPS::CalcMethodType::kEuler;
	traj_conf->depth = fixed_depth;  // Used in uniform depth mode or as fallback
#ifdef USE_PER_PARTICLE_DEPTH
	traj_conf->particle_depths = sample_depths;  // Per-particle depths
#endif
	traj_conf->deltaT = ONE_MINUTE * 10;			
	traj_conf->simulationDuration = std::abs(day_gap);
	traj_conf->recordT = ONE_DAY * 10;
    auto direction_str = (traj_conf->directionType == MOPS::CalcDirection::kForward) ? "FORWARD" : "BACKWARD";

	auto tiltle = name_prefix + "_";
	traj_conf->fileName = tiltle + direction_str;
	
	Debug("== multiple timesteps [pathline] ==");
#ifdef USE_PER_PARTICLE_DEPTH
	Debug("[MODE: PER-PARTICLE DEPTH] particles: %zu, depths: %zu", sample_points.size(), sample_depths.size());
#else
	Debug("[MODE: UNIFORM DEPTH] particles: %zu, depth: %.2f meters", sample_points.size(), fixed_depth);
#endif
    

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
#ifdef USE_PER_PARTICLE_DEPTH
		lastDepths_vec.clear();
#endif
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
#ifdef USE_PER_PARTICLE_DEPTH
			// Save the depth for this particle (from the trajectory line)
			lastDepths_vec.push_back(lines[i].depth);
#endif
		}
	}
	
    delete traj_conf;
	traj_conf = nullptr;
}


void IO()
{
    const char* yaml_path = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/test_ab_climatology.yaml";
	int timestep = 0;
	auto mpasoGrid = std::make_shared<MOPS::MPASOGrid>();
    auto solFront = std::make_shared<MOPS::MPASOSolution>();
	auto solBack = std::make_shared<MOPS::MPASOSolution>();

    auto pairs = MOPS_IO::make_forward_month_pairs(01, 1, 02, 12);
    
	{
        MOPS::MOPS_Init("gpu");
    }

	// Use new MOPS timing system
	{
		mpasoGrid->initGrid(MOPS::MPASOReader::readGridData(yaml_path).get());
	}

	bool isFirst = true;

    

    for (const auto& p : pairs)
    {
        Debug("pair: %s to %s", p.first.c_str(), p.second.c_str());
		
        {
            solFront->initSolution(MOPS::MPASOReader::readSolData(yaml_path, p.first, timestep).get());
            solBack->initSolution(MOPS::MPASOReader::readSolData(yaml_path, p.second, timestep).get());
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
        Debug("%s %s", p.first.c_str(), p.second.c_str());
        
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
        
        tutorial_pathLine(fileNamePrefix, fixed_depth, isFirst, getTimeGapinSecond(t2.c_str(), t1.c_str()));
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