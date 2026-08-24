/**
 * ============================================================================
 * reGrid.cpp - MPAS-O Regridding Tutorial
 * ============================================================================
 *
 * CONFIGURATION GUIDE - TWO MODES
 * ============================================================================
 *
 * ┌─── Mode 1: Single Latitude [DEFAULT] ──────────────────────────────────┐
 * │  Process one latitude for quick testing                                │
 * │                                                                         │
 * │  CONFIG:                                                                │
 * │    Keep commented: // #define PROCESS_MULTIPLE_LATITUDES (line 49)     │
 * │                                                                         │
 * │  PARAMETERS (line 58):                                                  │
 * │    SINGLE_LATITUDE: Latitude to process (degrees, -90 to 90)           │
 * │                                                                         │
 * │  CURRENT: Process latitude 0° (equator)                                │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * ┌─── Mode 2: Multiple Latitudes ──────────────────────────────────────────┐
 * │  Process multiple latitudes in a loop                                  │
 * │                                                                         │
 * │  CONFIG:                                                                │
 * │    Uncomment: #define PROCESS_MULTIPLE_LATITUDES (line 49)             │
 * │                                                                         │
 * │  PARAMETERS (lines 61-63):                                              │
 * │    LAT_START:  Starting latitude (degrees)                             │
 * │    LAT_END:    Ending latitude (degrees)                               │
 * │    LAT_STEP:   Latitude increment (degrees)                            │
 * │                                                                         │
 * │  EXAMPLE: Process from -90° to 90° with 2° increments                  │
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
#include "Core/GPUContext.h"
#include <filesystem>

// ============================================================================
// Processing Mode Configuration
// ============================================================================
// Define PROCESS_MULTIPLE_LATITUDES to process a range of latitudes
// #define PROCESS_MULTIPLE_LATITUDES  // COMMENTED OUT: Using single latitude mode

// ============================================================================
// Single Latitude Configuration
// ============================================================================
const float SINGLE_LATITUDE = 0.0f;  // Latitude to process (degrees)

// ============================================================================
// Multiple Latitudes Configuration
// ============================================================================
const float LAT_START = -90.0f;  // Starting latitude (degrees)
const float LAT_END = 90.0f;     // Ending latitude (degrees)
const float LAT_STEP = 2.0f;     // Latitude increment (degrees)

// ============================================================================
// Simulation Configuration
// ============================================================================
const char* YAML_CONFIG_PATH = "/pscratch/sd/q/qiuyf/MOPS_Tutorial/yaml_dir/bmoorema.yaml";
const int YEAR = 1;
const int MONTH = 1;
const int DAY = 1;
const int TIMESTEP = 0;

// ============================================================================
// Grid Configuration
// ============================================================================
const int GRID_WIDTH = 360 * 2;   // Longitude resolution (pixels)

// ============================================================================
// Output Configuration
// ============================================================================
const char* OUTPUT_DIRECTORY = "regrid_outputs";
const bool SAVE_PNG = true;   // Save as PNG images
const bool SAVE_BIN = true;   // Save as binary files

namespace fs = std::filesystem;

void tutorial_regrid(MOPS::MPASOGrid* mpasoGrid, MOPS::MPASOSolution* solFront, MOPS::MPASOField* mpasoF,
                     float latitude, const std::string& output_dir, MOPS::GPUContext& gpu_ctx)
{
	const int re_grid_w = GRID_WIDTH;
	const int re_grid_h = mpasoGrid->cellRefBottomDepth_vec.size();

	MOPS::VisualizationSettings* vis_config = new MOPS::VisualizationSettings();
	vis_config->imageSize = vec2{static_cast<double>(re_grid_w), static_cast<double>(re_grid_h)};
	vis_config->DepthRange = vec2{mpasoGrid->cellRefBottomDepth_vec[0], mpasoGrid->cellRefBottomDepth_vec.back()};
	vis_config->LonRange = vec2{-180.0, 180.0};
	vis_config->FixedLatitude = latitude;

	MOPS::ImageBuffer<double>* img = new MOPS::ImageBuffer<double>(re_grid_w, re_grid_h);

	std::cout << "== regrid and visualize at latitude " << latitude << " ==" << std::endl;
	MOPS::MPASOVisualizer::VisualizeFixedLatitude(mpasoF, vis_config, img, gpu_ctx);

	// Create filenames with latitude
	std::string lat_str = (latitude >= 0) ? "N" + std::to_string(static_cast<int>(latitude))
	                                      : "S" + std::to_string(static_cast<int>(-latitude));

	// Save as PNG images
	if (SAVE_PNG)
	{
		std::string png_E = output_dir + "/E_" + lat_str + ".png";
		std::string png_N = output_dir + "/N_" + lat_str + ".png";
		MOPS::SaveToPNG<double>(*img, png_E, 0);
		MOPS::SaveToPNG<double>(*img, png_N, 1);
	}

	// Save as binary
	if (SAVE_BIN)
	{
		std::string bin_filename = output_dir + "/regrid_" + lat_str + ".bin";
		std::ofstream binFile(bin_filename, std::ios::binary);
		if (!binFile)
		{
			std::cerr << "[ERROR] Cannot open " << bin_filename << " for writing" << std::endl;
		}
		else
		{
			binFile.write(reinterpret_cast<const char*>(img->mPixels.data()),
			              img->mPixels.size() * sizeof(double));
			binFile.close();
		}
	}

	std::cout << "Saved latitude " << latitude << " to " << output_dir
	          << " (" << img->getWidth() << " x " << img->getHeight() << " x 4 channels)" << std::endl;

	delete img;
	delete vis_config;
}

void IO()
{
	std::string timeStamp = MOPS_IO::make_date_tag(YEAR, MONTH, DAY);

	// Create output directory
	std::string output_dir = OUTPUT_DIRECTORY;
	if (!fs::exists(output_dir)) {
		fs::create_directory(output_dir);
		std::cout << "Created output directory: " << output_dir << std::endl;
	}

	auto mpasoGrid = std::make_shared<MOPS::MPASOGrid>();
	auto solFront = std::make_shared<MOPS::MPASOSolution>();

	solFront->initSolution(MOPS::MPASOReader::readSolData(YAML_CONFIG_PATH, timeStamp, TIMESTEP).get());
	mpasoGrid->initGrid(MOPS::MPASOReader::readGridData(YAML_CONFIG_PATH).get());

	std::cout << "refBottomDepth: " << mpasoGrid->cellRefBottomDepth_vec[0]
	          << " to " << mpasoGrid->cellRefBottomDepth_vec.back() << std::endl;

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
	auto mpasoF = MOPS::MOPS_GetFieldSnapshots();

	// Initialize GPU context once
	MOPS::GPUContext gpu_ctx;
	#if defined(MOPS_USE_CUDA) && (MOPS_USE_CUDA == 1)
	gpu_ctx = MOPS::GPUContext::FromCUDA(nullptr);
	#elif defined(MOPS_USE_SYCL) && (MOPS_USE_SYCL == 1)
	sycl::queue q(sycl::default_selector_v);
	gpu_ctx = MOPS::GPUContext::FromSYCL(q);
	#endif

#ifdef PROCESS_MULTIPLE_LATITUDES
	{
		std::cout << "== start regridding across latitudes ==" << std::endl;
		for (float lat = LAT_START; lat <= LAT_END; lat += LAT_STEP) {
			tutorial_regrid(mpasoGrid.get(), solFront.get(), mpasoF.get(), lat, output_dir, gpu_ctx);
		}
		std::cout << "== All regridding complete! Files saved to " << output_dir << " ==" << std::endl;
	}
#else
	{
		std::cout << "== Processing single latitude: " << SINGLE_LATITUDE << " ==" << std::endl;
		tutorial_regrid(mpasoGrid.get(), solFront.get(), mpasoF.get(), SINGLE_LATITUDE, output_dir, gpu_ctx);
		std::cout << "== Regridding complete! Files saved to " << output_dir << " ==" << std::endl;
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
