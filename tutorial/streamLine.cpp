#include "ggl.h"
#include "api/MOPS.h"
#include "Utils/cxxopts.hpp"
#include "Utils/Utils.hpp"
#include "Utils/YamlGen.hpp"
#include "IO/VTKFileManager.hpp"
#include "IO/MPASOReader.h"
#include "SYCL/ImageBuffer.hpp"
#include <string>

float fixed_depth = 10.0;

void tutoral_streamLine(const std::string name_prefix, float fixed_depth, MOPS_IO::YMD simulation)
{
	std::vector<CartesianCoord> sample_points; 
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
	
	MOPS::TrajectorySettings* traj_conf = new MOPS::TrajectorySettings;
	traj_conf->directionType = MOPS::CalcDirection::kForward;
	traj_conf->depth = fixed_depth;
	traj_conf->deltaT = ONE_MINUTE * 1;			
	traj_conf->simulationDuration = ONE_YEAR * 2;
	traj_conf->recordT = ONE_MINUTE * 6;
    auto direction_str = (traj_conf->directionType == MOPS::CalcDirection::kForward) ? "FORWARD" : "BACKWARD";

	auto tiltle = name_prefix + "_";
	traj_conf->fileName = tiltle + direction_str;
	
	std::cout << "== single timesteps [streamline] ==" << std::endl;
    
	auto lines = MOPS::MOPS_RunStreamLine(traj_conf, sample_points);
	std::cout << "length " << lines[0].points.size() << std::endl;
	MOPS::VTKFileManager::SaveTrajectoryLinesAsVTP(lines, traj_conf->fileName);
    delete traj_conf;
	traj_conf = nullptr;
}






void IO()
{
    const char* yaml_path = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/test.yaml";
	std::string timeStamp = MOPS_IO::make_date_tag(15, 1, 1); // YYYY-MM-DD
	int timeStep = 0;
	MOPS_IO::YMD simulation = {2, 0, 0}; // YMD data structure
	auto fileNamePrefix = [&]() {
		auto end = MOPS_IO::fromStringYMD(std::to_string(toIntYMD(timeStamp))) + simulation;
		return std::string("StreamLine_") +
			std::to_string(toIntYMD(timeStamp)) +
			"_to_" +
			std::to_string(toIntYMD(MOPS_IO::make_date_tag(end.year, end.month, end.day)));
	}();

	

	auto mpasoGrid = std::make_shared<MOPS::MPASOGrid>();
    auto solFront = std::make_shared<MOPS::MPASOSolution>();

	solFront->initSolution(MOPS::MPASOReader::readSolData(yaml_path, timeStamp, timeStep).get());
	solFront->addAttribute("temperature", MOPS::AttributeFormat::kFloat);
    solFront->addAttribute("salinity", MOPS::AttributeFormat::kFloat);
	
	mpasoGrid->initGrid(MOPS::MPASOReader::readGridData(yaml_path).get());

    MOPS::MOPS_Init("gpu");

	MOPS::MOPS_Begin();
    MOPS::MOPS_AddGridMesh(mpasoGrid);
    MOPS::MOPS_AddAttribute(solFront->getID(), solFront); // for each Solution set a uniqe ID
    MOPS::MOPS_End();

	MOPS::MOPS_ActiveAttribute(solFront->getID());

	tutoral_streamLine(fileNamePrefix, fixed_depth, simulation);


}




int main()
{
	
    IO();
	return 0;
}