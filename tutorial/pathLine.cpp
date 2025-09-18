#include "ggl.h"
#include "api/MOPS.h"
#include "Utils/cxxopts.hpp"
#include "Utils/Utils.hpp"
#include "Utils/YamlGen.hpp"
#include "IO/VTKFileManager.hpp"

#include "IO/MPASOReader.h"
#include <vector>


std::vector<CartesianCoord> lastPts_vec;
float fixed_depth = 10.0;

void testPathLine(const std::string name_prefix, float fixed_depth, bool isFirstPts, int day_gap, std::vector<int> timestep_vec)
{
	std::vector<CartesianCoord> sample_points; 
	if (isFirstPts)
	{
		std::cout << "== generate sample points ==" << std::endl;
		MOPS::SamplingSettings* sampling_conf = new MOPS::SamplingSettings();
		sampling_conf->setSampleRange(vec2i(2, 2));
		sampling_conf->setGeoBox(vec2(35.0, 45.0),  vec2(-60.0, -15.0));
		sampling_conf->atCellCenter(false);
		sampling_conf->setDepth(fixed_depth);
		MOPS::MOPS_GenerateSamplePoints(sampling_conf, sample_points);
		delete sampling_conf;
		sampling_conf = nullptr;

        sample_points[0] = CartesianCoord{ 1908930.101867, -5174124.236251, 3189701.032088}; 
	}
	else
	{
		std::cout << "== load sample points from memory ==" << std::endl;
		if (lastPts_vec.size() != 0)
		{
			sample_points.resize(lastPts_vec.size());
            for (auto idx = 0; idx < lastPts_vec.size(); idx++)
            {
                sample_points[idx] = CartesianCoord{lastPts_vec[idx].x(), lastPts_vec[idx].y(), lastPts_vec[idx].z()};
            }
        }
        else
        {
            std::cerr << "[ERROR]::No sample points in memory!" << std::endl;
            exit(-1);
		}
	}
	
	MOPS::TrajectorySettings* traj_conf = new MOPS::TrajectorySettings;
	traj_conf->directionType = MOPS::CalcDirection::kForward;
	traj_conf->methodType = MOPS::CalcMethodType::kRK4;
	traj_conf->depth = fixed_depth;
	traj_conf->deltaT = ONE_MINUTE * 1;			
	traj_conf->simulationDuration = std::abs(day_gap);
	traj_conf->recordT = ONE_MINUTE * 6;
    auto direction_str = (traj_conf->directionType == MOPS::CalcDirection::kForward) ? "FORWARD" : "BACKWARD";

	auto tiltle = name_prefix + "_";
	traj_conf->fileName = tiltle + direction_str;
	
	std::cout << "== multiple timesteps [pathline] ==" << std::endl;
    
	auto lines = MOPS::MOPS_RunPathLine(traj_conf, sample_points, timestep_vec);
	std::cout << "length " << lines[0].points.size() << std::endl;
	MOPS::VTKFileManager::SaveTrajectoryLinesAsVTP(lines, traj_conf->fileName);
	// save last pts to memory
    lastPts_vec.clear();
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
			// std::cerr << "[Warn]::line[" << idx << "] all points are (0,0,0), inserting dummy zero.\n";
			last_pts.push_back(pts.back());  // or CartesianCoord{0, 0, 0};
		}
	}
	for (const auto& p : last_pts) {
        lastPts_vec.push_back(CartesianCoord{p.x(), p.y(), p.z()});
    }
	std::cout << "== save last pts to memory: " << lastPts_vec.size() << std::endl;
	
    delete traj_conf;
	traj_conf = nullptr;
}


void testIO()
{
    const char* yaml_path = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/test.yaml";

	auto mpasoGrid = std::make_shared<MOPS::MPASOGrid>();
    auto solFront = std::make_shared<MOPS::MPASOSolution>();
	auto solBack = std::make_shared<MOPS::MPASOSolution>();

    auto pairs1 = MOPS_IO::make_forward_month_pairs(18, 1, 20, 12);
    std::cout << "Total pairs: " << pairs1.size() << std::endl;

	mpasoGrid->initGrid(MOPS::MPASOReader::readGridData(yaml_path).get());
	bool isFirst = true;

    MOPS::MOPS_Init("gpu");

    for (const auto& p : pairs1)
    {
        std::cout << "pair: " << p.first << " to " << p.second << std::endl;

        solFront->initSolution(MOPS::MPASOReader::readSolData(yaml_path, p.first, 0).get());
        solBack->initSolution(MOPS::MPASOReader::readSolData(yaml_path, p.second, 0).get());
        solFront->addAttribute("temperature", MOPS::AttributeFormat::kFloat);
        solFront->addAttribute("salinity", MOPS::AttributeFormat::kFloat);
        solBack->addAttribute("temperature", MOPS::AttributeFormat::kFloat);
        solBack->addAttribute("salinity", MOPS::AttributeFormat::kFloat);
        
        auto t1 =solFront->getCurrentTime();
        auto t2 = solBack->getCurrentTime();
        auto fileNamePreix = "PathLine_" + std::to_string(toIntYMD(p.first)) + "_to_" + std::to_string(toIntYMD(p.second));
        std::cout << toIntYMD(p.first) << " " << toIntYMD(p.second) << std::endl;
        MOPS::MOPS_Begin();
        MOPS::MOPS_AddGridMesh(mpasoGrid);
        MOPS::MOPS_AddAttribute(toIntYMD(p.first), solFront);
        MOPS::MOPS_AddAttribute(toIntYMD(p.second), solBack);
        MOPS::MOPS_End();
        MOPS::MOPS_ActiveAttribute(toIntYMD(p.first), toIntYMD(p.second));
        std::vector<int> timestep_vec = {toIntYMD(p.first), toIntYMD(p.second)};
        testPathLine(fileNamePreix, fixed_depth, isFirst, getTimeGapinSecond(t2.c_str(), t1.c_str()), timestep_vec);
        isFirst = false;
    }

}

int main()
{
	
    testIO();

	return 0;
}
