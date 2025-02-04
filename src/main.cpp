
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


void TimeFixed(sycl::queue& sycl_Q, ftk::stream* stream)
{
	auto grid_path = stream->substreams[0]->filenames[0];
	auto vel_path = stream->substreams[1]->filenames[0];
	
	auto gs = stream->read_static();
	std::cout << " [ files attribute information" << " ]\n";
	mpasoGrid = std::make_shared<MPASOGrid>();
	mpasoGrid->initGrid(gs.get(), MPASOReader::readGridInfo(grid_path).get());
	mpasoGrid->createKDTree("./index.bin", sycl_Q);

	int timestep = 0;
	auto gt = stream->read(timestep);
	mpasoSol = std::make_shared<MPASOSolution>();
	mpasoSol->initSolution(gt.get(), MPASOReader::readSolInfo(vel_path, timestep).get());
	mpasoSol->calcCellVertexZtop(mpasoGrid.get(), sycl_Q);
	mpasoSol->calcCellCenterVelocity(mpasoGrid.get(), sycl_Q);
	mpasoSol->calcCellVertexVelocity(mpasoGrid.get(), sycl_Q);

	mpasoField = std::make_shared<MPASOField>();
	mpasoField->initField(mpasoGrid, mpasoSol);


	constexpr int width = 361; constexpr int height = 181;
	VisualizationSettings* config = new VisualizationSettings();
	config->imageSize = vec2(width, height);
	config->LatRange = vec2(20.0, 50.0);
	config->LonRange = vec2(-17.0, 17.0);
	config->FixedLayer = 0.0;
	config->TimeStep = 0.0;
	config->CalcType = CalcAttributeType::kZonalMerimoal;
	config->VisType = VisualizeType::kFixedLayer;
	config->PositionType = CalcPositionType::kPoint;


	VisualizationSettings* config_fixed_depth = new VisualizationSettings();
	config_fixed_depth->imageSize = vec2(width, height);
	config_fixed_depth->LatRange = vec2(20.0, 50.0);
	config_fixed_depth->LonRange = vec2(-17.0, 17.0);
	config_fixed_depth->FixedDepth = 700.0;
	config_fixed_depth->TimeStep = 0.0;
	config_fixed_depth->CalcType = CalcAttributeType::kZonalMerimoal;
	config_fixed_depth->VisType = VisualizeType::kFixedDepth;
	config_fixed_depth->PositionType = CalcPositionType::kPoint;

	VisualizationSettings* config_fixed_lat = new VisualizationSettings();
	config_fixed_lat->imageSize = vec2(width, height);
	config_fixed_lat->LonRange = vec2(-17.0, 17.0);
	config_fixed_lat->DepthRange = vec2(0.0, 5000.0);
	config_fixed_lat->FixedLatitude = 35.0;
	config_fixed_lat->TimeStep = 0.0;
	config_fixed_lat->CalcType = CalcAttributeType::kZonalMerimoal;
	config_fixed_lat->VisType = VisualizeType::kFixedDepth;
	config_fixed_lat->PositionType = CalcPositionType::kPoint;

	std::cout << "==========================================\n";

	ImageBuffer<double>* img1 = new ImageBuffer<double>(config->imageSize.x(), config->imageSize.y());
	ImageBuffer<double>* img2 = new ImageBuffer<double>(config_fixed_depth->imageSize.x(), config_fixed_depth->imageSize.y());
	ImageBuffer<double>* img3 = new ImageBuffer<double>(config_fixed_lat->imageSize.x(), config_fixed_lat->imageSize.y());

	// remapping
	std::cout << " [ Run remapping test(fixed time)" << " ]\n";
	std::cout << " 	[ Run remapping VisualizeFixedLayer(fixed time)" << " ]\n";
	MPASOVisualizer::VisualizeFixedLayer(mpasoField.get(), config, img1, sycl_Q);
	VTKFileManager::SaveVTI(img1, config);
	
	std::cout << " 	[ Run remapping VisualizeFixedDepth(fixed time)" << " ]\n";
	MPASOVisualizer::VisualizeFixedDepth(mpasoField.get(), config_fixed_depth, img2, sycl_Q);
	VTKFileManager::SaveVTI(img2, config_fixed_depth);
	//MPASOVisualizer::VisualizeFixedLatitude(mpasoField.get(), config_fixed_lat, img3, sycl_Q);
  	//Debug("finished... fixed_lat_35.000000.vti");

	std::cout << " 	[ Run Trtrajector(fixed time)" << " ]\n";
	SamplingSettings* sample_conf = new SamplingSettings;
	sample_conf->sampleDepth = config_fixed_depth->FixedDepth;
	sample_conf->sampleLatitudeRange = vec2(25.0, 45.0);
	sample_conf->sampleLongitudeRange = vec2(-10.0, 10.0);
	sample_conf->sampleNumer = vec2i(31, 31);

	TrajectorySettings* traj_conf = new TrajectorySettings;
	traj_conf->deltaT = ONE_MINUTE;			
	traj_conf->simulationDuration = ONE_MINUTE * 30;	
	traj_conf->recordT = ONE_MINUTE * 6;
	traj_conf->fileName = "output_line";

	std::vector<CartesianCoord> sample_points; std::vector<int> cell_id_vec;
	MPASOVisualizer::GenerateSamplePoint(sample_points, sample_conf);
	//std::cout << "points size() " << sample_points.size() << std::endl;
	VTKFileManager::SavePointAsVTP(sample_points, "output_origin");
	
	mpasoField->calcInWhichCells(sample_points, cell_id_vec);
	//std::cout << std::fixed << std::setprecision(4) << "before trajector cell_id [ " << cell_id_vec[0] << " ]" << " " << sample_points[0].x() << " " << sample_points[0].y() << " " << sample_points[0].z() << std::endl;

	MPASOVisualizer::VisualizeTrajectory(mpasoField.get(), sample_points, traj_conf, cell_id_vec, sycl_Q);

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
	if (cmd)
    	path = cmd->input_yaml_path.c_str();

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

	std::shared_ptr<ftk::stream> stream(new ftk::stream);
	if (!data_path_prefix.empty()) stream->set_path_prefix(data_path_prefix);
	std::cout << " [ files information" << " ]\n";
	stream->parse_yaml(input_yaml_filename);
	

	TimeFixed(sycl_Q, stream.get());	

	return 0;
}
