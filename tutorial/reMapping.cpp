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

void tutoral_reMapping(float fixed_depth)
{
	constexpr int width = 3601; constexpr int height = 1801;
	MOPS::VisualizationSettings* config = new MOPS::VisualizationSettings();
	config->imageSize = vec2(width, height);
	config->LatRange = vec2(-90.0, 90.0);
	config->LonRange = vec2(-180.0, 180.0);
	config->FixedDepth = fixed_depth;
	config->TimeStep = 0;
	
#if MOPS_VTK
	config->SaveType = MOPS::SaveType::kVTI;
#else
	config->SaveType = MOPS::SaveType::kPNG;
#endif
	
	auto img_vec = MOPS::MOPS_RunRemapping(config);
#if MOPS_VTK
	std::string str = "";
#if MOPS_MPI == 1
	str += "rank_" + std::to_string(rank_id) + "_";
#endif
	str += "timestep_" + std::to_string(config->TimeStep) + "_";
	str += "tile_" + std::to_string(config->tile_index);
	std::vector<std::string> names = {
			"E: Zonal Velocity", "N: Meridional Velocity", "Velocity Magnitude",
			"Temperature", "Salinity", "None"
		};
#endif

	if (config->SaveType == MOPS::SaveType::kVTI)
	{
		#if MOPS_VTK
		MOPS::VTKFileManager::SaveVTI(img_vec, config, names, str);
		#endif
	}
	else if (config->SaveType == MOPS::SaveType::kPNG)
	{
		for (int i = 0; i < img_vec.size(); ++i)
		{
			for (int ch = 0; ch < 3; ++ch)  // channel 0,1,2
			{
				#if MOPS_MPI
				std::string filename = "rank_" + std::to_string(rank_id) + "_output_" + std::to_string(i) + "_ch" + std::to_string(ch) + ".png";
				#else
				std::string filename = "output_" + std::to_string(i) + "_ch" + std::to_string(ch) + ".png";
				#endif
				MOPS::SaveToPNG(img_vec[i], filename, ch);
			}
		}
	}
	
	delete config;
	config = nullptr;
}






void IO()
{
    const char* yaml_path = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/test.yaml";
	std::string timeStamp = MOPS_IO::make_date_tag(15, 1, 1); // YYYY-MM-DD
	int timeStep = 0;
	
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

	tutoral_reMapping(fixed_depth);


}




int main()
{
	
    IO();
	return 0;
}