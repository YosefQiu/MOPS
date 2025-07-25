#include "Core/MOPSApp.h"



namespace MOPS
{
    void MOPSApp::init(const char* device)
    {
		if (strcmp(device, "gpu") == 0)
        	mSYCLQueue = sycl::queue(sycl::gpu_selector_v);
		else if (strcmp(device, "cpu") == 0)
			mSYCLQueue = sycl::queue(sycl::cpu_selector_v);
        std::cout << " [ system information" << " ]\n";
        std::cout << "Device selected : " << mSYCLQueue.get_device().get_info<sycl::info::device::name>() << "\n";
        std::cout << "Device vendor   : " << mSYCLQueue.get_device().get_info<sycl::info::device::vendor>() << "\n";
        std::cout << "Device version  : " << mSYCLQueue.get_device().get_info<sycl::info::device::version>() << "\n";

		

		mpasoGrid = std::make_shared<MPASOGrid>();
		
    }

    void MOPSApp::addGrid(std::shared_ptr<MPASOGrid> grid)
    {
		mpasoGrid = std::move(grid);
		mpasoGrid->createKDTree((mpasoGrid->mDataDir + "/" + "KDTree.bin").c_str(), mSYCLQueue);
		mDataDir = mpasoGrid->mDataDir;
		std::cout << " [\u2713]finished loading grid information" << std::endl;
    }

    void MOPSApp::addSol(int timestep, std::shared_ptr<MPASOSolution> sol)
    {
		std::shared_ptr<MPASOSolution> mpasoSol; 
		mpasoSol = sol;

		mpasoSol->mCellsSize = mpasoGrid->mCellsSize;
		mpasoSol->mEdgesSize = mpasoGrid->mEdgesSize;
		mpasoSol->mMaxEdgesSize = mpasoGrid->mMaxEdgesSize;
		mpasoSol->mVertexSize = mpasoGrid->mVertexSize;
		mpasoGrid->mVertLevels = mpasoSol->mVertLevels;
		mpasoGrid->mVertLevelsP1 = mpasoSol->mVertLevelsP1;

		mpasoSol->calcCellCenterZtop();
		std::cout << " [ Run calcCellVertexZtop" << " ]\n";
		mpasoSol->calcCellVertexZtop(mpasoGrid.get(), mDataDir, mSYCLQueue);
		std::cout << " [ Run calcCellCenterVelocity" << " ]\n";
		mpasoSol->calcCellCenterVelocityByZM(mpasoGrid.get(), mDataDir, mSYCLQueue);
		std::cout << " [ Run calcCellVertexVelocity" << " ]\n";
		mpasoSol->calcCellVertexVelocity(mpasoGrid.get(), mDataDir, mSYCLQueue);

		for (const auto& [name, vec] : mpasoSol->mDoubleAttributes)
		{
			std::cout << " [ Run calcCellCenterToVertex for " << name << " ]\n";
			mpasoSol->calcCellCenterToVertex(name, vec, mpasoGrid.get(), mDataDir, mSYCLQueue);
			std::cout << " [\u2713]finished loading sol information at timestep [ " << mpasoSol->mCurrentTime << " ]" << std::endl;
		}

		mpasoAttributeMap[timestep] = mpasoSol;

		std::cout << "Total number of attributes: " << mpasoAttributeMap.size() << std::endl;		
    }

    void MOPSApp::addField()
    {
        mpasoField = std::make_shared<MPASOField>();
		mpasoField->initField(mpasoGrid, mpasoAttributeMap.begin()->second);
    }

    void MOPSApp::activeAttribute(int timestep)
    {
		mpasoField.reset();
		auto iter = mpasoAttributeMap.find(timestep);
		if (iter == mpasoAttributeMap.end())
		{
			std::cout << " [activeAttribute]::Error: timestep [ " << timestep << " ] not found" << std::endl;
			return;
		}
		mpasoField = std::make_shared<MPASOField>();
        mpasoField->initField(mpasoGrid, iter->second);
    }

    std::vector<ImageBuffer<double>> MOPSApp::runRemapping(VisualizationSettings* config)
    {
		std::vector<ImageBuffer<double>> img_vec;
		// img0 : velocity 
		// img1 ++ ---> : other attributes
		auto attr_size = mpasoField->mSol_Front->mDoubleAttributes.size();
		img_vec.emplace_back(config->imageSize.x(), config->imageSize.y());  // img[0] for velocity
		if (attr_size > 0) 
		{
			int group_count = (attr_size + 2) / 3;  
			for (int i = 0; i < group_count; ++i) 
			{
				img_vec.emplace_back(config->imageSize.x(), config->imageSize.y());
			}
		}

		
		
		// remapping
		std::cout << " [ Run remapping " << " ]\n";
		MPASOVisualizer::VisualizeFixedDepth(mpasoField.get(), config, img_vec, mSYCLQueue);
		std:: cout << " ==== [\u2713] done..." << std::endl;

		return img_vec;
    }

	
	void MOPSApp::generateSamplePoints(SamplingSettings* config, std::vector<CartesianCoord>& sample_points)
	{
		MPASOVisualizer::GenerateSamplePoint(sample_points, config);
	}
	
	void MOPSApp::generateSamplePointsAtCenter(SamplingSettings* config, std::vector<CartesianCoord>& sample_points)
	{
		int number = 0;
		for (auto& pt: this->mpasoGrid->cellCoord_vec)
		{
			// if (number >= 10) break;
			sample_points.push_back(pt);
			number++;
		}

		MPASOVisualizer::GenerateSamplePointAtCenter(sample_points, config);
	}

    std::vector<TrajectoryLine> MOPSApp::runStreamLine(TrajectorySettings* config, std::vector<CartesianCoord>& sample_points)
    {
    	std::vector<int> cell_id_vec;

		mpasoField->calcInWhichCells(sample_points, cell_id_vec);
		
		auto lines = MPASOVisualizer::StreamLine(mpasoField.get(), sample_points, config, cell_id_vec, mSYCLQueue);
		std:: cout << " ==== [\u2713] done..." << std::endl;
		return lines;
	}

	std::vector<TrajectoryLine> MOPSApp::runPathLine(TrajectorySettings* config, std::vector<CartesianCoord>& sample_points, std::vector<int>& timestep_vec)
	{
		// check if the timestep_vec is valid
		for (auto timestep : timestep_vec)
		{
			if (mpasoAttributeMap.find(timestep) == mpasoAttributeMap.end())
			{
				std::cerr << "[Error]: Timestep " << timestep << " not found in attribute map." << std::endl;
				return {};
			}
		}

		std::vector<TrajectoryLine> lines;
		for (auto idx = 0; idx + 1 < timestep_vec.size(); idx++)
		{
			auto solFront = mpasoAttributeMap[timestep_vec[idx]];
			auto solBack = mpasoAttributeMap[timestep_vec[idx + 1]];
			if (solFront == nullptr || solBack == nullptr)
			{
				std::cerr << "[Error]: Solution at timestep " << timestep_vec[idx] << " is nullptr" << std::endl;
				return {};
			}
			mpasoField->initField(mpasoGrid, solFront, solBack);

			std::vector<int> cell_id_vec;
			mpasoField->calcInWhichCells(sample_points, cell_id_vec);
			auto lines_i = MPASOVisualizer::PathLine(mpasoField.get(), sample_points, config, cell_id_vec, mSYCLQueue);
			std:: cout << " ==== [\u2713] done... [ " 
			<< timestep_vec[idx] << " to " << timestep_vec[idx + 1] << " ]" << std::endl;

			// update sample_points
			for (auto sample_idx = 0; sample_idx < sample_points.size(); sample_idx++)
			{
				sample_points[sample_idx] = lines_i[sample_idx].lastPoint;
			}
			if (idx == 0)
				lines = lines_i;
			else
			{
				for (auto line_idx = 0; line_idx < lines_i.size(); line_idx++)
				{
					auto &fullLine = lines[line_idx].points;
					auto &segPts = lines_i[line_idx].points;
					if (segPts.size() > 1)
					{
						fullLine.insert(fullLine.end(), segPts.begin() + 1, segPts.end());
					}
					lines[line_idx].lastPoint = lines_i[line_idx].lastPoint;
				}
			}
		}

		return lines;
	}

	bool MOPSApp::checkAttribute() const
	{
		if (!mpasoGrid)
		{
			std::cerr << "[Error]: Grid is nullptr" << std::endl;
			return false;
		}

		bool grid_valid = mpasoGrid->checkAttribute();
		if (!grid_valid)
		{
			std::cerr << "[Error]: Grid attribute check failed" << std::endl;
			return false;
		}

		for (const auto& [timestep, sol] : mpasoAttributeMap)
		{
			if (!sol)
			{
				std::cerr << "[Error]: Solution at timestep " << timestep << " is nullptr" << std::endl;
				return false;
			}
			if (!sol->checkAttribute())
			{
				std::cerr << "[Error]: Attribute check failed at timestep " << timestep << std::endl;
				return false;
			}
		}

		return true;
	}

}