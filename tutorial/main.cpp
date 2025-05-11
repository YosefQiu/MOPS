#include "ggl.h"
#include "api/MOPS.h"
#include "cxxopts.hpp"
#include "Utils.hpp"
#include "VTKFileManager.hpp"

int timestep;
int day_gap = 1;
double fixed_depth = 10.0;
bool isLoadFromDisk = false;
std::vector<int> time_range_vec;
enum class FileType : int {kDaily, kMonthly, kCount};
std::string input_yaml_filename;
std::string data_path_prefix;




bool parseCommandLine(int argc, char* argv[], 
	std::string& input_yaml_filename, std::string& data_path_prefix, 
	int& timestep, std::vector<int>& time_range_vec, int& day_gap, double& fixed_depth) 
{
    cxxopts::Options options(argv[0]);
	options.add_options()
		("input,i", "Input yaml file", cxxopts::value<std::string>(input_yaml_filename))
		("prefix,p", "Data path prefix", cxxopts::value<std::string>(data_path_prefix))
		("timestep,t", "single timestep", cxxopts::value<int>(timestep)->default_value("0"))  
		("range,r", "Timestep range", cxxopts::value<std::vector<int>>(time_range_vec))
		("day,g", "Day Gap", cxxopts::value<int>(day_gap)->default_value("1"))
		("depth,d", "Fixed depth", cxxopts::value<double>(fixed_depth)->default_value("10.0"))
		("help,h", "Print this information");
		

	auto results = options.parse(argc, argv);
	if (results.count("help")) 
	{
        std::cout << options.help() << std::endl;
        return false;
    }
	if (!results.count("input")) 
	{
        Debug("[ERROR]::No input file detected");
        return false;
	}
	return true;
}

int main(int argc, char* argv[])
{

	if (!parseCommandLine(argc, argv, input_yaml_filename, data_path_prefix, timestep, time_range_vec, day_gap, fixed_depth)) exit(1);
	std::cout << "== command line arguments ==" << std::endl;
	std::cout << "== input_yaml_filename: " << input_yaml_filename << std::endl;
	std::cout << "== data_path_prefix: " << data_path_prefix << std::endl;
	std::cout << "== timestep: " << timestep << std::endl;
	std::cout << "== day_gap: " << day_gap << std::endl;
	std::cout << "== time_range_vec: ";
	for (auto idx = 0; idx < time_range_vec.size(); idx++)
	{
		std::cout << time_range_vec[idx] << " ";
	}
	std::cout << std::endl;
	
	if (data_path_prefix.empty()) isLoadFromDisk = false;
	else isLoadFromDisk = true;
	
	// 1. initialize MOPS
	MOPS::MOPS_Init("gpu");
	
	// 2. initialize grid and solutions
	std::vector<int> timestep_vec; 
	if (time_range_vec.size() == 0) timestep_vec.push_back(timestep);
	else
	{
		for (auto i = 0; i < time_range_vec.size(); i++)
		{
			timestep_vec.push_back(time_range_vec[i]);
		}
	}
	
	std::shared_ptr<MOPS::MPASOGrid> mpasoGrid = std::make_shared<MOPS::MPASOGrid>();
	mpasoGrid->initGrid_DemoLoading(input_yaml_filename.c_str());

	std::vector<std::shared_ptr<MOPS::MPASOSolution>> mpasoSol_vec;
	mpasoSol_vec.resize(timestep_vec.size());
	for (auto idx = 0; idx < timestep_vec.size(); idx++)
	{
		mpasoSol_vec[idx] = std::make_shared<MOPS::MPASOSolution>();
		mpasoSol_vec[idx]->initSolution_DemoLoading(input_yaml_filename.c_str(), timestep_vec[idx]);
	}
	
	// 3. add more specific attributes on specific timestep
	for (auto idx = 0; idx < timestep_vec.size(); idx++)
	{
		mpasoSol_vec[idx]->addAttribute("temperature", MOPS::AttributeFormat::kFloat);
		mpasoSol_vec[idx]->addAttribute("salinity", MOPS::AttributeFormat::kFloat);
	}
	
	// 4. add grid and attributes
	MOPS::MOPS_Begin();
	MOPS::MOPS_AddGridMesh(mpasoGrid);
	for (auto idx = 0; idx < timestep_vec.size(); idx++)
	{
		MOPS::MOPS_AddAttribute(timestep_vec[idx], mpasoSol_vec[idx]);
	}
	MOPS::MOPS_End();
	
	// 5. activate the first attribute
	for (auto idx = 0; idx < timestep_vec.size(); idx++)
	{
		MOPS::MOPS_ActiveAttribute(timestep_vec[idx]);
		//6. run remapping
		constexpr int width = 3601; constexpr int height = 1801;
		MOPS::VisualizationSettings* config = new MOPS::VisualizationSettings();
		config->imageSize = vec2(width, height);
		config->LatRange = vec2(25.0, 45.0);
		config->LonRange = vec2(-90.0, -25.0);
		config->FixedDepth = fixed_depth;
		config->TimeStep = timestep_vec[idx];
		// config->SaveType = MOPS::SaveType::kVTI;
		// MOPS::MOPS_RunRemapping(config);

		delete config;
		config = nullptr;
	}


	// 7. generate sample points
	std::vector<CartesianCoord> sample_points; 
	if (isLoadFromDisk == false)
	{
		std::cout << "== generate sample points ==" << std::endl;
		MOPS::SamplingSettings* sampling_conf = new MOPS::SamplingSettings();
		sampling_conf->setSampleRange(vec2i(31, 31));
		sampling_conf->setGeoBox(vec2(35.0, 45.0),  vec2(-90.0, -45.0));
		sampling_conf->setDepth(fixed_depth);
		MOPS::MOPS_GenerateSamplePoints(sampling_conf, sample_points);  // x, y, z coordinates
		delete sampling_conf;
		sampling_conf = nullptr;
	}
	else
	{
		std::cout << "== load sample points from disk ==" << std::endl;
		std::string pts_name = data_path_prefix;
		std::ifstream ifs(pts_name);
		if (ifs.is_open())
		{
			std::string line;
			while (std::getline(ifs, line))
			{
				std::istringstream iss(line);
				double x, y, z;
				if (iss >> x >> y >> z)
				{
					vec3 pt = vec3{x, y, z};
					sample_points.push_back(pt);
				}
			}
			ifs.close();
		}
		else
		{
			std::cout << "[ERROR]::Failed to open file: " << pts_name << std::endl;
			return -1;
		}
	}

	// 8. generate trajectory
	MOPS::TrajectorySettings* traj_conf = new MOPS::TrajectorySettings;
	traj_conf->depth = fixed_depth;
	traj_conf->deltaT = ONE_MINUTE * 6;			
	traj_conf->simulationDuration = ONE_DAY * day_gap;
	traj_conf->recordT = ONE_MINUTE * 6;
	traj_conf->fileName = "traj_line_" + std::to_string(timestep_vec[0]);
	if (timestep_vec.size() == 1)
	{
		std::cout << "== single timestep [streamline] ==" << std::endl;
		auto lines = MOPS::MOPS_RunStreamLine(traj_conf, sample_points);
		MOPS::VTKFileManager::SaveTrajectoryLinesAsVTP(lines, traj_conf->fileName);
	}
	else
	{	
		std::cout << "== multiple timesteps [pathline] ==" << std::endl;
		auto lines = MOPS::MOPS_RunPathLine(traj_conf, sample_points, timestep_vec);
		MOPS::VTKFileManager::SaveTrajectoryLinesAsVTP(lines, traj_conf->fileName);
		// save last pts to disk
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

			if (!found) {
				// std::cerr << "[Warn]::line[" << idx << "] all points are (0,0,0), inserting dummy zero.\n";
				last_pts.push_back(pts.back());  // or CartesianCoord{0, 0, 0};
			}
		}
		std::string pts_name = "traj_pts_" + std::to_string(timestep_vec[0]) + ".txt";
		std::ofstream ofs(pts_name);
		if (ofs.is_open())
		{
			for (auto idx = 0; idx < last_pts.size(); idx++)
			{
				ofs << last_pts[idx].x() << " " << last_pts[idx].y() << " " << last_pts[idx].z() << std::endl;
			}
			ofs.close();
		}
		else
		{
			std::cout << "[ERROR]::Failed to open file: " << pts_name << std::endl;
		}
		std::cout << "== save last pts to disk: " << pts_name << std::endl;
	}
		
	
	
	
	
	delete traj_conf;
	traj_conf = nullptr;


	
	return 0;
}
