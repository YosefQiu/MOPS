#include "Core/MOPSApp.h"
#include "Core/CPUContext.h"
#include "Core/GPUContext.h"
#include "Core/RuntimeContext.h"
#include "Utils/Timer.hpp"
#include <vector>
#include "version.h"

namespace MOPS
{
namespace {
RuntimeContext CreateRuntimeContext(sycl::queue& q)
{
#if defined(MOPS_USE_CPU) && (MOPS_USE_CPU == 1)
    (void)q;
    CPUContext cpu_ctx;
    cpu_ctx.backend = CPUBackend::kTBB;
    return RuntimeContext::FromCPU(cpu_ctx);
#elif defined(MOPS_USE_SYCL) && (MOPS_USE_SYCL == 1)
    return RuntimeContext::FromGPU(GPUContext::FromSYCL(q));
#elif defined(MOPS_USE_HIP) && (MOPS_USE_HIP == 1)
	(void)q;
	return RuntimeContext::FromGPU(GPUContext::FromHIP(nullptr));
#elif defined(MOPS_USE_CUDA) && (MOPS_USE_CUDA == 1)
    (void)q;
    return RuntimeContext::FromGPU(GPUContext::FromCUDA(nullptr));
#else
    (void)q;
    return RuntimeContext{};
#endif
}
} // namespace

    void MOPSApp::init(const char* device)
    {
		#if defined(MOPS_USE_SYCL) && (MOPS_USE_SYCL == 1)
		if (strcmp(device, "gpu") == 0)
        	mSYCLQueue = sycl::queue(sycl::gpu_selector_v);
		else if (strcmp(device, "cpu") == 0)
			mSYCLQueue = sycl::queue(sycl::cpu_selector_v);
        std::cout << " [ system information" << " ]\n";
        std::cout << "Device selected : " << mSYCLQueue.get_device().get_info<sycl::info::device::name>() << "\n";
        std::cout << "Device vendor   : " << mSYCLQueue.get_device().get_info<sycl::info::device::vendor>() << "\n";
        std::cout << "Device version  : " << mSYCLQueue.get_device().get_info<sycl::info::device::version>() << "\n";
		#else
		(void)device;
		std::cout << " [ system information ]\n";
		#if defined(MOPS_USE_CPU) && (MOPS_USE_CPU == 1)
		std::cout << "Device selected : CPU backend (TBB)\n";
		#elif defined(MOPS_USE_HIP) && (MOPS_USE_HIP == 1)
		std::cout << "Device selected : HIP backend (compiled via CUDA/NVCC toolchain)\n";
		#elif defined(MOPS_USE_CUDA) && (MOPS_USE_CUDA == 1)
		std::cout << "Device selected : CUDA backend (runtime queue handled by CUDA implementation)\n";
		#else
		std::cout << "Device selected : Unknown backend\n";
		#endif
		#endif

		std::cout << "MOPS Version    : " << MOPS_VERSION << "\n";

		mpasoGrid = std::make_shared<MPASOGrid>();
		
    }

    void MOPSApp::addGrid(std::shared_ptr<MPASOGrid> grid)
    {
		MOPS_TIMER_START("Preprocessing::addGrid", TimerCategory::Preprocessing);
		
		mpasoGrid = std::move(grid);
		mpasoGrid->createKDTree((mpasoGrid->mCachedDataDir + "/" + "KDTree.bin").c_str(), mSYCLQueue);
		mDataDir = mpasoGrid->mCachedDataDir;
		Debug("[MOPSApp]::Finished loading grid information");
		
		MOPS_TIMER_STOP("Preprocessing::addGrid");
    }

    void MOPSApp::addSol(int solID, std::shared_ptr<MPASOSolution> sol)
    {
		MOPS_TIMER_START("Preprocessing::addSol", TimerCategory::Preprocessing);
		
		// check if solID already exists
		auto iter = mpasoAttributeMap.find(solID);
		if (iter != mpasoAttributeMap.end())
		{
			MOPS_TIMER_STOP("Preprocessing::addSol");
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
			RuntimeContext runtime_ctx = CreateRuntimeContext(mSYCLQueue);
			Debug("[MOPSApp]::Run calcCellCenterZtop");
			mpasoSol->calcCellCenterZtop();
			Debug("[MOPSApp]::Run calcCellVertexZtop");
			mpasoSol->calcCellVertexZtop(mpasoGrid.get(), mDataDir, runtime_ctx);
		}
		if (mpasoSol->cellVertexZonalVelocity_vec.size() == 0 && mpasoSol->cellVertexMeridionalVelocity_vec.size() == 0
			&& mpasoSol->cellVertexVelocity_vec.size() == 0)
		{
			RuntimeContext runtime_ctx = CreateRuntimeContext(mSYCLQueue);
			Debug("[MOPSApp]::Run calcCellCenterVelocity");
			mpasoSol->calcCellCenterVelocityByZM(mpasoGrid.get(), mDataDir, runtime_ctx);
			Debug("[MOPSApp]::Run calcCellVertexVelocity");
			mpasoSol->calcCellVertexVelocity(mpasoGrid.get(), mDataDir, runtime_ctx);
		}
		if (mpasoSol->cellVertexVertVelocity_vec.size() == 0)
		{
			RuntimeContext runtime_ctx = CreateRuntimeContext(mSYCLQueue);
			Debug("[MOPSApp]::Run calcCellVertexVertVelocity");
			mpasoSol->calcCellVertexVertVelocity(mpasoGrid.get(), mDataDir, runtime_ctx);
		}
		
		RuntimeContext runtime_ctx = CreateRuntimeContext(mSYCLQueue);
		for (const auto& [name, vec] : mpasoSol->mDoubleAttributes)
		{
			Debug("[MOPSApp]::Run calcCellCenterToVertex for %s", name.c_str());
			mpasoSol->calcCellCenterToVertex(name, vec, mpasoGrid.get(), mDataDir, runtime_ctx);
			Debug("[MOPSApp]::Finished loading sol information at timestep %d", mpasoSol->mTimesteps);
		}

		mpasoAttributeMap[solID] = mpasoSol;

		Debug("[MOPSApp]::Total number of attributes: %zu", mpasoAttributeMap.size());
		
		MOPS_TIMER_STOP("Preprocessing::addSol");
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
			Error("[MOPSApp]::activeAttribute: solID %d not found", ID1);
			return;
		}
        if (ID2.has_value())
        {
            auto iter2 = mpasoAttributeMap.find(ID2.value());
            if (iter2 == mpasoAttributeMap.end())
            {
                Error("[MOPSApp]::activeAttribute: solID %d not found", ID2.value());
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
		Debug("[MOPSApp]::Run remapping");
		RuntimeContext runtime_ctx = CreateRuntimeContext(mSYCLQueue);
		MPASOVisualizer::VisualizeFixedDepth(mpasoField.get(), config, img_vec, runtime_ctx);
		Debug("[MOPSApp]::Remapping done");

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
		MOPS_TIMER_START("GPUKernel::StreamLine", TimerCategory::GPUKernel);
		
    	std::vector<int> cell_id_vec;

		mpasoField->calcInWhichCells(sample_points, cell_id_vec);
		
		// test
		auto test_cell_id = cell_id_vec[0];
		auto test_cell_surface_vel = mpasoField->mSol_Front->cellCenterVelocity_vec[mpasoField->mSol_Front->mVertLevels * test_cell_id + 0];
		auto vel_length = MOPS_LENGTH(test_cell_surface_vel);
		Debug("[MOPSApp]::test_cell_id = %d, test_cell_surface_vel = (%.6f, %.6f, %.6f), length = %.6f", 
			test_cell_id, test_cell_surface_vel.x(), test_cell_surface_vel.y(), test_cell_surface_vel.z(), vel_length);
		
		RuntimeContext runtime_ctx = CreateRuntimeContext(mSYCLQueue);
		auto lines = MPASOVisualizer::StreamLine(mpasoField.get(), sample_points, config, cell_id_vec, runtime_ctx);
		Debug("[MOPSApp]::StreamLine done");
		
		MOPS_TIMER_STOP("GPUKernel::StreamLine");
		return lines;
	}

	std::vector<TrajectoryLine> MOPSApp::runPathLine(TrajectorySettings* config, std::vector<CartesianCoord>& sample_points)
	{
		MOPS_TIMER_START("GPUKernel::PathLine", TimerCategory::GPUKernel);
		
		// check if the SolFront and SolBack are set
		if (mpasoField == nullptr)
		{
			Error("[MOPSApp]::mpasoField is nullptr, please activeAttribute first");
			MOPS_TIMER_STOP("GPUKernel::PathLine");
			exit(-1);
		}

		if (mpasoField->mSol_Front == nullptr || mpasoField->mSol_Back == nullptr)
		{
			Error("[MOPSApp]::Sol_Front or Sol_Back is nullptr, please activeAttribute first");
			MOPS_TIMER_STOP("GPUKernel::PathLine");
			exit(-1);
		}

		std::vector<TrajectoryLine> lines;


		auto solFront = mpasoField->mSol_Front;
		auto solBack = mpasoField->mSol_Back;
		
		std::vector<int> cell_id_vec;
		mpasoField->calcInWhichCells(sample_points, cell_id_vec);
		
		RuntimeContext runtime_ctx = CreateRuntimeContext(mSYCLQueue);
		auto lines_i = MPASOVisualizer::PathLine(mpasoField.get(), sample_points, config, cell_id_vec, runtime_ctx);
		Debug("[MOPSApp]::PathLine done [%s to %s]", solFront->getTimeStamp().c_str(), solBack->getTimeStamp().c_str());

		// update sample_points
		for (auto sample_idx = 0; sample_idx < sample_points.size(); sample_idx++)
		{
			sample_points[sample_idx] = lines_i[sample_idx].lastPoint;
		}
		lines = lines_i;
		
		MOPS_TIMER_STOP("GPUKernel::PathLine");

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
			RuntimeContext runtime_ctx = CreateRuntimeContext(mSYCLQueue);
			auto lines_i = MPASOVisualizer::PathLine(mpasoField.get(), sample_points, config, cell_id_vec, runtime_ctx);
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
			Error("[MOPSApp]::Grid is nullptr");
			return false;
		}

		bool grid_valid = mpasoGrid->checkAttribute();
		if (!grid_valid)
		{
			Error("[MOPSApp]::Grid attribute check failed");
			return false;
		}

		for (const auto& [id, sol] : mpasoAttributeMap)
		{
			if (!sol)
			{
				Error("[MOPSApp]::Solution at solID %d is nullptr", id);
				return false;
			}
			if (!sol->checkAttribute())
			{
				Error("[MOPSApp]::Attribute check failed at solID %d", id);
				return false;
			}
		}

		return true;
	}

}