
#include "ggl.h"
#include "MPASOReader.h"
#include "MPASOGrid.h"
#include "MPASOSolution.h"
#include "MPASOField.h"
#include "ImageBuffer.hpp"
#include "MPASOVisualizer.h"
#include "VTKFileManager.hpp"
#include "Command.hpp"	

#include "ndarray/ndarray_group_stream.hh"

// time fixed
std::shared_ptr<MPASOGrid> mpasoGrid = nullptr;
std::shared_ptr<MPASOSolution> mpasoSol = nullptr;
std::shared_ptr<MPASOField> mpasoField = nullptr;

// time varying
std::shared_ptr<MPASOGrid> tv_mpasoGrid = nullptr;
std::shared_ptr<MPASOSolution> tv_mpasoSol1 = nullptr;
std::shared_ptr<MPASOSolution> tv_mpasoSol2 = nullptr;
std::shared_ptr<MPASOField> tv_mpasoField1 = nullptr;
std::shared_ptr<MPASOField> tv_mpasoField2 = nullptr;

const char* path;

std::string input_yaml_filename;
std::string data_path_prefix;

std::string vectorToString(const std::vector<std::string>& vec, const std::string& delimiter = ", ") {
	std::ostringstream oss;
	for (size_t i = 0; i < vec.size(); ++i) {
		if (i != 0) {
			oss << delimiter;
		}
		oss << vec[i];
	}
	return oss.str();
}

enum class FileType : int {kDaily, kMonthly, kCount};

void calcIterator(FileType fileType, int numberOfFiles, int gap, int& iter)
{
	switch (fileType)
	{
	case FileType::kDaily:
	{
		if (numberOfFiles == 1)
		{
			iter = 31 / gap;
		}
		else
		{
			Debug("TODO FUNC calcIterator");
			exit(1);
		}
	}
		break;
	case FileType::kMonthly:
		break;
	default:
		break;
	}
}

void Remapping_Test(sycl::queue& sycl_Q, ftk::stream* stream, std::optional<Command> cmd, int gap = 1)
{
	if (!cmd.has_value())
	{
		Debug("Error: No command line arguments provided.");
		return;
	}

    const int num_files = 1;
    std::string data_path_prefix[num_files];

	if (num_files > 1)
	{
		for (int i = 0; i < num_files; ++i) 
		{
			int month = i + 1;
			char month_str[3];
			sprintf(month_str, "%02d", month);

			// Constructing file paths
			data_path_prefix[i] = "archive/ocn/hist/20231012.GMPAS-JRA1p5.TL319_oRRS18to6.pm-cpu.mpaso.hist.am.timeSeriesStatsMonthly.0001-" 
			                      + std::string(month_str) + "-01.nc";
		}
	}
    
	int iter;
	calcIterator(FileType::kDaily, 1, gap, iter);
	Debug("iter ======================================== [ %d ]", iter);
	
	int timestep = 0;
	std::string fileName = removeFileExtension(stream->substreams[0]->filenames[0]);
	std::string dataDir = createDataPath(".data", fileName);
	std::vector<CartesianCoord> sample_points; std::vector<int> cell_id_vec;
	for (auto file_i = 0; file_i < iter; file_i++)
	{
		if (num_files > 1)
			stream->set_path_prefix(data_path_prefix[file_i]);

		auto grid_path = stream->substreams[0]->filenames[0];
		auto vel_path = stream->substreams[1]->filenames[0];
		
		auto gs = stream->read_static();
		if (file_i == 0)
		{
			std::cout << " [ files attribute information" << " ]\n";
			mpasoGrid = std::make_shared<MPASOGrid>();
			mpasoGrid->initGrid(gs.get(), MPASOReader::readGridInfo(grid_path).get());
			mpasoGrid->createKDTree((dataDir + "/" + "KDTree.bin").c_str(), sycl_Q);
			std::cout << " [\u2713]finished loading grid information" << std::endl;
		}
		
		auto gt = stream->read(timestep);
		mpasoSol = std::make_shared<MPASOSolution>();
		mpasoSol->initSolution(gt.get(), MPASOReader::readSolInfo(vel_path, timestep).get());

		mpasoSol->mCellsSize = mpasoGrid->mCellsSize;
		mpasoSol->mEdgesSize = mpasoGrid->mEdgesSize;
		mpasoSol->mMaxEdgesSize = mpasoGrid->mMaxEdgesSize;
		mpasoSol->mVertexSize = mpasoGrid->mVertexSize;
		mpasoGrid->mVertLevels = mpasoSol->mVertLevels;
		mpasoGrid->mVertLevelsP1 = mpasoSol->mVertLevelsP1;

		mpasoSol->calcCellCenterZtop();
		std::cout << " [ Run calcCellVertexZtop" << " ]\n";
		mpasoSol->calcCellVertexZtop(mpasoGrid.get(), dataDir, sycl_Q);
		std::cout << " [ Run calcCellCenterVelocity" << " ]\n";
		mpasoSol->calcCellCenterVelocityByZM(mpasoGrid.get(), dataDir, sycl_Q);
		std::cout << " [ Run calcCellVertexVelocity" << " ]\n";
		mpasoSol->calcCellVertexVelocity(mpasoGrid.get(), dataDir, sycl_Q);
		std::cout << " [\u2713]finished loading sol information at timestep [ " << mpasoSol->mCurrentTime << " ]" << std::endl;

		mpasoField = std::make_shared<MPASOField>();
		mpasoField->initField(mpasoGrid, mpasoSol);

		const int width = cmd->image_width; const int height = cmd->image_height;
		VisualizationSettings* config_fixed_layer = new VisualizationSettings();
		config_fixed_layer->imageSize = vec2(width, height);
		config_fixed_layer->LatRange = vec2(cmd->latitude_min, cmd->latitude_max);
		config_fixed_layer->LonRange = vec2(cmd->longitude_min, cmd->longitude_max);
		config_fixed_layer->FixedLayer = cmd->fixed_layer;
		config_fixed_layer->TimeStep = 0.0;
		config_fixed_layer->CalcType = CalcAttributeType::kZonalMerimoal;
		config_fixed_layer->VisType = VisualizeType::kFixedLayer;
		config_fixed_layer->PositionType = CalcPositionType::kPoint;

		VisualizationSettings* config_fixed_depth = new VisualizationSettings();
		config_fixed_depth->imageSize = vec2(width, height);
		config_fixed_depth->LatRange = vec2(cmd->latitude_min, cmd->latitude_max);
		config_fixed_depth->LonRange = vec2(cmd->longitude_min, cmd->longitude_max);
		config_fixed_depth->FixedDepth = cmd->fixed_depth;
		config_fixed_depth->TimeStep = 0.0;
		config_fixed_depth->CalcType = CalcAttributeType::kZonalMerimoal;
		config_fixed_depth->VisType = VisualizeType::kFixedDepth;
		config_fixed_depth->PositionType = CalcPositionType::kPoint;

		std::cout << "==========================================\n";

		ImageBuffer<double>* img1 = new ImageBuffer<double>(config_fixed_layer->imageSize.x(), config_fixed_layer->imageSize.y());
		ImageBuffer<double>* img2 = new ImageBuffer<double>(config_fixed_depth->imageSize.x(), config_fixed_depth->imageSize.y());
		
		// remapping
		std::string outPutBase = "output_" + std::to_string(file_i + 1) + "_";
		std::cout << " [ Run remapping test" << " ]\n";
		std::cout << " == [ Run remapping VisualizeFixedLayer(fixed time)" << " ] " << std::flush;
		MPASOVisualizer::VisualizeFixedLayer(mpasoField.get(), config_fixed_layer, img1, sycl_Q);
		std:: cout << " ==== [\u2713] done..." << std::endl;
		std::cout << " ==== [ saving ]" << std::flush;
		VTKFileManager::SaveVTI(img1, config_fixed_layer, outPutBase);
		std:: cout << " ==== [\u2713] done..." << std::endl;

		std::cout << " == [ Run remapping VisualizeFixedDepth(fixed time)" << " ] " << std::flush;
		MPASOVisualizer::VisualizeFixedDepth(mpasoField.get(), config_fixed_depth, img2, sycl_Q);
		std:: cout << " ==== [\u2713] done..." << std::endl;
		std::cout << " ==== [ saving ]" << std::flush;
		VTKFileManager::SaveVTI(img2, config_fixed_depth, outPutBase);
		std:: cout << " ==== [\u2713] done..." << std::endl;
		
		gap += iter;
	}
}


int main(int argc, char* argv[])
{
	std::optional<Command> cmd;
	try {
        cmd = Command::parse(argc, argv);
        cmd->print();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
	

	sycl::queue sycl_Q;
#if __linux__
	sycl_Q = sycl::queue(sycl::gpu_selector_v);
#elif _WIN32
	sycl_Q = sycl::queue(sycl::cpu_selector_v);
#elif __APPLE__
	sycl_Q = nullptr;
#endif
	std::cout << " [ system information" << " ]\n";
	std::cout << "Device selected : " << sycl_Q.get_device().get_info<sycl::info::device::name>() << "\n";
	std::cout << "Device vendor   : " << sycl_Q.get_device().get_info<sycl::info::device::vendor>() << "\n";
	std::cout << "Device version  : " << sycl_Q.get_device().get_info<sycl::info::device::version>() << "\n";

	if (cmd)
    	path = cmd->input_yaml_path.c_str();
	std::shared_ptr<ftk::stream> stream(new ftk::stream);
	if (!data_path_prefix.empty()) stream->set_path_prefix(data_path_prefix);
	std::cout << " [ files information" << " ]\n";
	stream->parse_yaml(cmd->input_yaml_path.c_str());
	

	Remapping_Test(sycl_Q, stream.get(), cmd, 31);	

	return 0;
}
