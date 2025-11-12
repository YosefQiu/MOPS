#include "ggl.h"
#include "api/MOPS.h"
#include "Utils/cxxopts.hpp"
#include "Utils/Utils.hpp"
#include "Utils/YamlGen.hpp"
#include "IO/VTKFileManager.hpp"
#include "IO/MPASOReader.h"
#include "SYCL/ImageBuffer.hpp"


float fixed_depth = 10.0;


void tutoral_regrid(MOPS::MPASOGrid* mpasoGrid, MOPS::MPASOSolution* solFront, MOPS::MPASOField* mpasoF)
{

	const int re_grid_w = 360 * 2;
	const int re_grid_h = mpasoGrid->cellRefBottomDepth_vec.size();

	MOPS::VisualizationSettings* vis_config = new MOPS::VisualizationSettings();
	vis_config->imageSize = vec2(re_grid_w, re_grid_h);
	vis_config->DepthRange = vec2(mpasoGrid->cellRefBottomDepth_vec[0], mpasoGrid->cellRefBottomDepth_vec.back()); 
	vis_config->LonRange = vec2(-180.0, 180.0);
	vis_config->FixedLatitude = 40.0f;

	sycl::queue q(sycl::default_selector_v);
	MOPS::ImageBuffer<double>* img = new MOPS::ImageBuffer<double>(re_grid_w, re_grid_h);
	std::cout << "== regrid and visualize at fixed latitude ==" << std::endl;
	MOPS::MPASOVisualizer::VisualizeFixedLatitude(mpasoF, vis_config, img, q);
	std::cout << "== save regridded image ==" << std::endl;
	// save as image use stb_image
	MOPS::SaveToPNG<double>(*img, "E.png", 0);
	MOPS::SaveToPNG<double>(*img, "N.png", 1);
	// save as binary
	std::string filename = "regrid_fixed_latitude.bin";
	std::ofstream binFile(filename, std::ios::binary);
	if (!binFile)
    {
        std::cerr << "Error: cannot open " << filename << " for writing\n";
        return;
    }
	binFile.write(reinterpret_cast<const char*>(img->mPixels.data()),
                  img->mPixels.size() * sizeof(double));

	binFile.close();
    std::cout << "Saved raw binary file: " << "regrid_fixed_latitude.bin"
              << " (" << img->getWidth() << " x " << img->getHeight()
              << " x 4 channels)\n";

}



void IO()
{
    const char* yaml_path = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/bmoorema.yaml";
	std::string timeStamp = MOPS_IO::make_date_tag(01, 1, 1); // YYYY-MM-DD
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
	auto mpasoF = MOPS::MOPS_GetFieldSnapshots();
	tutoral_regrid(mpasoGrid.get(), solFront.get(), mpasoF.get());


}




int main()
{
	
    IO();
	return 0;
}