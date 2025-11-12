#include "Core/MOPSApp.h"
#include <vector>



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
		mpasoGrid->createKDTree((mpasoGrid->mCachedDataDir + "/" + "KDTree.bin").c_str(), mSYCLQueue);
		mDataDir = mpasoGrid->mCachedDataDir;
		std::cout << " [\u2713]finished loading grid information" << std::endl;
    }

    void MOPSApp::addSol(int solID, std::shared_ptr<MPASOSolution> sol)
    {
		// check if solID already exists
		auto iter = mpasoAttributeMap.find(solID);
		if (iter != mpasoAttributeMap.end())
		{
			return;
		}

		std::shared_ptr<MPASOSolution> mpasoSol; 
		mpasoSol = sol;

		mpasoSol->mCellsSize = mpasoGrid->mCellsSize;
		mpasoSol->mEdgesSize = mpasoGrid->mEdgesSize;
		mpasoSol->mMaxEdgesSize = mpasoGrid->mMaxEdgesSize;
		mpasoSol->mVertexSize = mpasoGrid->mVertexSize;
		
		mpasoGrid->mVertLevels = mpasoSol->mVertLevels;
		mpasoGrid->mVertLevelsP1 = mpasoSol->mVertLevelsP1;

		if (mpasoSol->cellVertexZTop_vec.size() == 0)
		{
			std::cout << " [ Run calcCellCenterZtop" << " ]\n";
			mpasoSol->calcCellCenterZtop();
			std::cout << " [ Run calcCellVertexZtop" << " ]\n";
			mpasoSol->calcCellVertexZtop(mpasoGrid.get(), mDataDir, mSYCLQueue);
		}
		if (mpasoSol->cellVertexZonalVelocity_vec.size() == 0 && mpasoSol->cellVertexMeridionalVelocity_vec.size() == 0
			&& mpasoSol->cellVertexVelocity_vec.size() == 0)
		{
			std::cout << " [ Run calcCellCenterVelocity" << " ]\n";
			mpasoSol->calcCellCenterVelocityByZM(mpasoGrid.get(), mDataDir, mSYCLQueue);
			std::cout << " [ Run calcCellVertexVelocity" << " ]\n";
			mpasoSol->calcCellVertexVelocity(mpasoGrid.get(), mDataDir, mSYCLQueue);
		}
		

		for (const auto& [name, vec] : mpasoSol->mDoubleAttributes)
		{
			std::cout << " [ Run calcCellCenterToVertex for " << name << " ]\n";
			mpasoSol->calcCellCenterToVertex(name, vec, mpasoGrid.get(), mDataDir, mSYCLQueue);
			std::cout << " [\u2713]finished loading sol information at timestep [ " << mpasoSol->mTimesteps << " ]" << std::endl;
		}

		mpasoAttributeMap[solID] = mpasoSol;

		std::cout << "Total number of attributes: " << mpasoAttributeMap.size() << std::endl;		
    }

    void MOPSApp::addField()
    {
        mpasoField = std::make_shared<MPASOField>();
		mpasoField->initField(mpasoGrid, mpasoAttributeMap.begin()->second);
    }

    void MOPSApp::activeAttribute(int ID1, std::optional<int> ID2)
    {
		mpasoField.reset();
		mpasoField = std::make_shared<MPASOField>();
		auto iter = mpasoAttributeMap.find(ID1);
		if (iter == mpasoAttributeMap.end())
		{
			std::cout << " [activeAttribute]::Error: solID [ " << ID1 << " ] not found" << std::endl;
			return;
		}
        if (ID2.has_value())
        {
            auto iter2 = mpasoAttributeMap.find(ID2.value());
            if (iter2 == mpasoAttributeMap.end())
            {
                std::cout << " [activeAttribute]::Error: solID [ " << ID2.value() << " ] not found" << std::endl;
                return;
            }
            mpasoField->initField(mpasoGrid, iter->second, iter2->second);
        }
        else
        {
            mpasoField->initField(mpasoGrid, iter->second);
        }
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
		
		// test
		auto test_cell_id = cell_id_vec[0];
		auto test_cell_surface_vel = mpasoField->mSol_Front->cellCenterVelocity_vec[mpasoField->mSol_Front->mVertLevels * test_cell_id + 0];
		auto vel_length = YOSEF_LENGTH(test_cell_surface_vel);
		std::cout << " test_cell_id = " << test_cell_id << " test_cell_surface_vel = " << test_cell_surface_vel.x() 
			<< " " << test_cell_surface_vel.y() << " " << test_cell_surface_vel.z() << " length = " << vel_length << std::endl;
		
		auto lines = MPASOVisualizer::StreamLine(mpasoField.get(), sample_points, config, cell_id_vec, mSYCLQueue);
		std:: cout << " ==== [\u2713] done..." << std::endl;
		return lines;
	}

	std::vector<TrajectoryLine> MOPSApp::runPathLine(TrajectorySettings* config, std::vector<CartesianCoord>& sample_points)
	{
		// check if the SolFront and SolBack are set
		if (mpasoField == nullptr)
		{
			std::cerr << "[Error]: mpasoField is nullptr, please activeAttribute first." << std::endl;
			exit(-1);
		}

		if (mpasoField->mSol_Front == nullptr || mpasoField->mSol_Back == nullptr)
		{
			std::cerr << "[Error]: Sol_Front or Sol_Back is nullptr, please activeAttribute first." << std::endl;
			exit(-1);
		}

		std::vector<TrajectoryLine> lines;


		auto solFront = mpasoField->mSol_Front;
		auto solBack = mpasoField->mSol_Back;
		
		std::vector<int> cell_id_vec;
		mpasoField->calcInWhichCells(sample_points, cell_id_vec);
		
		auto lines_i = MPASOVisualizer::PathLine(mpasoField.get(), sample_points, config, cell_id_vec, mSYCLQueue);
		std:: cout << " ==== [\u2713] done... [ " 
			<< solFront->getTimeStamp() << " to " << solBack->getTimeStamp() << " ]" << std::endl;

		// update sample_points
		for (auto sample_idx = 0; sample_idx < sample_points.size(); sample_idx++)
		{
			sample_points[sample_idx] = lines_i[sample_idx].lastPoint;
		}
		lines = lines_i;

/*
		for (auto idx = 0; idx + 1 < timestep_vec.size(); idx++)
		{
			auto solFront = mpasoField->mSol_Front;
			auto solBack = mpasoField->mSol_Back;
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
*/
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

		for (const auto& [id, sol] : mpasoAttributeMap)
		{
			if (!sol)
			{
				std::cerr << "[Error]: Solution at solID " << id << " is nullptr" << std::endl;
				return false;
			}
			if (!sol->checkAttribute())
			{
				std::cerr << "[Error]: Attribute check failed at solID " << id << std::endl;
				return false;
			}
		}

		return true;
	}

}