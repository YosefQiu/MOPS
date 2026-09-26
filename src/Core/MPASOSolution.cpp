#include "Core/MPASOSolution.h"
#include "Common/MOPSFactory.h"
#include "Common/TrajectoryCommon.h"
#include "Utils/Interpolation.hpp"
#include "Utils/GeoConverter.hpp"
#include <cmath>


using namespace MOPS;

// ============================================================================
// Vertex Data Quality Validation
// ============================================================================

namespace {

struct ValidationResult {
    int zero_count = 0;
    int nan_count = 0;
    int unrealistic_count = 0;
    double min_val = 1e30;
    double max_val = -1e30;
};

ValidationResult ValidateVertexData(
    const std::vector<double>& vertex_data,
    const std::string& data_name,
    size_t num_vertices,
    size_t num_layers,
    double valid_min = -1e10,
    double valid_max = 1e10)
{
    ValidationResult result;

    for (size_t i = 0; i < vertex_data.size(); i++) {
        double val = vertex_data[i];

        // Track min/max (excluding NaN)
        if (!std::isnan(val)) {
            if (val < result.min_val) result.min_val = val;
            if (val > result.max_val) result.max_val = val;
        }

        // Check for NaN
        if (std::isnan(val)) {
            result.nan_count++;
            if (result.nan_count <= 5) {
                size_t vid = i / num_layers;
                size_t layer = i % num_layers;
                printf("  ❌ [%s] Vertex %zu Layer %zu: NaN detected!\n",
                       data_name.c_str(), vid, layer);
            }
            continue;
        }

        // Check for suspicious zeros
        if (val == 0.0) {
            size_t layer = i % num_layers;
            bool is_ztop = (data_name.find("ZTop") != std::string::npos);
            // For zTop, 0 is only valid at surface (layer 0)
            if (!is_ztop || layer > 0) {
                result.zero_count++;
                if (result.zero_count <= 5) {
                    size_t vid = i / num_layers;
                    printf("  ⚠️  [%s] Vertex %zu Layer %zu: value=0 (suspicious)\n",
                           data_name.c_str(), vid, layer);
                }
            }
        }

        // Check for values outside valid range
        if (val < valid_min || val > valid_max) {
            result.unrealistic_count++;
            if (result.unrealistic_count <= 5) {
                size_t vid = i / num_layers;
                size_t layer = i % num_layers;
                printf("  ⚠️  [%s] Vertex %zu Layer %zu: %.2f (out of range [%.1f, %.1f])\n",
                       data_name.c_str(), vid, layer, val, valid_min, valid_max);
            }
        }
    }

    return result;
}

void PrintValidationSummary(
    const std::string& data_name,
    const ValidationResult& result,
    size_t total_size)
{
    printf("\n=== Vertex Data Validation: %s ===\n", data_name.c_str());
    printf("  Total points: %zu\n", total_size);
    printf("  Value range:  [%.6f, %.6f]\n", result.min_val, result.max_val);

    printf("  Zero values:  %d", result.zero_count);
    if (result.zero_count > 0) printf(" ⚠️");
    printf("\n");

    printf("  NaN values:   %d", result.nan_count);
    if (result.nan_count > 0) printf(" ❌");
    printf("\n");

    printf("  Out of range: %d", result.unrealistic_count);
    if (result.unrealistic_count > 0) printf(" ⚠️");
    printf("\n");

    if (result.zero_count == 0 && result.nan_count == 0 && result.unrealistic_count == 0) {
        printf("  ✅ Data quality looks OK\n");
    } else {
        printf("  ⚠️  ISSUES DETECTED!\n");
    }
    printf("=========================================\n\n");
}

// Enhanced validation: check if vertex zeros come from cell zeros (land mask)
void ValidateVertexVsCellData(
    const std::vector<double>& vertex_data,
    const std::vector<double>& cell_data,
    MPASOGrid* grid,
    const std::string& data_name,
    size_t num_layers)
{
    printf("\n=== Enhanced Validation: %s (Vertex vs Cell) ===\n", data_name.c_str());

    int legitimate_zeros = 0;  // Vertex=0 because all source cells=0 (land)
    int problematic_zeros = 0; // Vertex=0 but source cells≠0 (interpolation issue)
    int examples_shown = 0;

    for (size_t vid = 0; vid < grid->vertexCoord_vec.size() && examples_shown < 10; vid++) {
        for (size_t layer = 0; layer < num_layers; layer++) {
            size_t vertex_idx = vid * num_layers + layer;
            double vertex_val = vertex_data[vertex_idx];

            // Only check zero values
            if (vertex_val != 0.0) continue;

            // Get cells around this vertex (MPAS: 3 cells per vertex)
            // cellsOnVertex_vec is flat array: [cell0, cell1, cell2] for each vertex
            bool all_cells_zero = true;
            bool any_cell_nonzero = false;
            std::vector<int> cell_ids;
            std::vector<double> cell_values;

            for (int ci = 0; ci < 3; ci++) {
                int cell_id = grid->cellsOnVertex_vec[3 * vid + ci] - 1;  // -1: MPAS uses 1-based indexing
                if (cell_id < 0) continue;  // Invalid cell

                size_t cell_idx = cell_id * num_layers + layer;
                if (cell_idx >= cell_data.size()) continue;

                double cell_val = cell_data[cell_idx];
                cell_ids.push_back(cell_id);
                cell_values.push_back(cell_val);

                if (cell_val != 0.0) {
                    all_cells_zero = false;
                    any_cell_nonzero = true;
                }
            }

            if (all_cells_zero && cell_values.size() > 0) {
                // Legitimate: all source cells are also 0 (likely land mask)
                legitimate_zeros++;
            } else if (any_cell_nonzero) {
                // Problematic: vertex=0 but some cells are non-zero
                problematic_zeros++;
                if (examples_shown < 10) {
                    printf("  🔴 Vertex %zu Layer %zu: vertex=0 but source cells have data!\n", vid, layer);
                    printf("     Source cells: ");
                    for (size_t i = 0; i < cell_ids.size() && i < cell_values.size(); i++) {
                        printf("Cell[%d]=%.2f ", cell_ids[i], cell_values[i]);
                    }
                    printf("\n");
                    examples_shown++;
                }
            }
        }
    }

    printf("  Legitimate zeros (land mask): %d\n", legitimate_zeros);
    printf("  Problematic zeros (data loss): %d", problematic_zeros);
    if (problematic_zeros > 0) printf(" 🔴");
    printf("\n");

    if (problematic_zeros == 0) {
        printf("  ✅ All zeros are legitimate (source cells also zero)\n");
    } else {
        printf("  ⚠️  Found %d vertices with zero despite non-zero source cells!\n", problematic_zeros);
        printf("     This indicates cell→vertex interpolation issues.\n");
    }
    printf("=========================================\n\n");
}

} // anonymous namespace

template<typename T>
void writeVertexZTopToFile(const std::vector<T>& vertexZTop_vec, const std::string& filename);

template<typename T>
void readVertexZTopFromFile(std::vector<T>& vertexZTop_vec, const std::string& filename);

void MPASOSolution::calcCellVertexZtop(MPASOGrid* grid, std::string& dataDir, const RuntimeContext& ctx)
{
    if (mTotalZTopLayer == 0 || mTotalZTopLayer == -1) mTotalZTopLayer = mVertLevels;
    if (!cellVertexZTop_vec.empty()) cellVertexZTop_vec.clear();
    cellVertexZTop_vec.resize(grid->vertexCoord_vec.size() * mTotalZTopLayer);

    auto cell_vertex_ztop_path = dataDir + "/" + "cellVertexZTop_vec_"  + std::to_string(mTimesteps) + ".bin";

    if (std::filesystem::exists(cell_vertex_ztop_path)) {
        readVertexZTopFromFile<double>(cellVertexZTop_vec, cell_vertex_ztop_path);

        // // Validate cached vertex data
        // auto result = ValidateVertexData(cellVertexZTop_vec,
        //                                  "Cell Vertex ZTop (cached)",
        //                                  grid->vertexCoord_vec.size(),
        //                                  mTotalZTopLayer,
        //                                  -7000.0,  // Valid range for ocean depth
        //                                  100.0);
        // PrintValidationSummary("Cell Vertex ZTop (cached)", result, cellVertexZTop_vec.size());

        return;
    }

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(grid);
    MOPS::Factory::CalcCellVertexZtop(
        grid,
        mCellsSize,
        mTotalZTopLayer,
        cellZTop_vec,
        cellVertexZTop_vec,
        grid_info_vec,
        ctx);

    writeVertexZTopToFile<double>(cellVertexZTop_vec, cell_vertex_ztop_path);
    Debug("[MPASOSolution]::Calc Cell Vertex Z Top  = \t [ %d ] \t type = [ float64 ]",
            cellVertexZTop_vec.size());

    // // Validate newly computed vertex data
    // auto result = ValidateVertexData(cellVertexZTop_vec,
    //                                  "Cell Vertex ZTop (computed)",
    //                                  grid->vertexCoord_vec.size(),
    //                                  mTotalZTopLayer,
    //                                  -7000.0,  // Valid range for ocean depth
    //                                  100.0);
    // PrintValidationSummary("Cell Vertex ZTop (computed)", result, cellVertexZTop_vec.size());
}

void MPASOSolution::calcCellVertexZtop(MPASOGrid* grid, std::string& dataDir, const GPUContext& ctx)
{
    calcCellVertexZtop(grid, dataDir, RuntimeContext::FromGPU(ctx));
}

void MPASOSolution::calcCellCenterToVertex(const std::string& name, const std::vector<double>& vec, MPASOGrid* grid, std::string& dataDir, const RuntimeContext& ctx)
{
    if (vec.empty()) return;
    if (mTotalZTopLayer == 0 || mTotalZTopLayer == -1) mTotalZTopLayer = mVertLevels;

    // Clean invalid cell center values before C2V interpolation
    // (temperature/salinity)
    cleanInvalidCellCenterValues(name, grid);

    // Use cleaned data from mDoubleAttributes instead of the input vec parameter
    // because cleanInvalidCellCenterValues modifies mDoubleAttributes[name]
    const std::vector<double>& data_to_use = mDoubleAttributes.count(name) ? mDoubleAttributes[name] : vec;

    std::vector<double> attr_CtoV_vec;
    attr_CtoV_vec.resize(grid->vertexCoord_vec.size() * mTotalZTopLayer);

    auto cell_vertex_attribute_path = dataDir + "/" + "cellVertex" + name + "_vec_"  + std::to_string(mTimesteps) + ".bin";

    // Determine valid range based on attribute name
    double valid_min = -1e10;
    double valid_max = 1e10;
    if (name.find("temperature") != std::string::npos || name.find("Temperature") != std::string::npos) {
        valid_min = -2.5;  // Below freezing point
        valid_max = 40.0;  // Max realistic ocean temp
    } else if (name.find("salinity") != std::string::npos || name.find("Salinity") != std::string::npos) {
        valid_min = 0.0;   // Fresh water
        valid_max = 45.0;  // Hypersaline
    }

    if (std::filesystem::exists(cell_vertex_attribute_path)) {
        readVertexZTopFromFile<double>(attr_CtoV_vec, cell_vertex_attribute_path);
        mDoubleAttributes_CtoV[name] = attr_CtoV_vec;

        // // Validate cached attribute data
        // auto result = ValidateVertexData(attr_CtoV_vec,
        //                                  "Cell Vertex " + name + " (cached)",
        //                                  grid->vertexCoord_vec.size(),
        //                                  mTotalZTopLayer,
        //                                  valid_min,
        //                                  valid_max);
        // PrintValidationSummary("Cell Vertex " + name + " (cached)", result, attr_CtoV_vec.size());

        // // Enhanced validation: check if zeros are from land mask
        // if (result.zero_count > 0) {
        //     ValidateVertexVsCellData(attr_CtoV_vec, data_to_use, grid, name, mTotalZTopLayer);
        // }

        return;
    }

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(grid);
    MOPS::Factory::CalcCellCenterToVertex(
        grid,
        mCellsSize,
        mTotalZTopLayer,
        data_to_use,  // Use cleaned data
        attr_CtoV_vec,
        grid_info_vec,
        ctx);

    writeVertexZTopToFile<double>(attr_CtoV_vec, cell_vertex_attribute_path);
    Debug("[MPASOSolution]::Calc Cell Vertex Attribute  = \t [ %d ] \t type = [ float64 ]",
            attr_CtoV_vec.size());
    mDoubleAttributes_CtoV[name] = attr_CtoV_vec;

    // // Validate newly computed attribute data
    // auto result = ValidateVertexData(attr_CtoV_vec,
    //                                  "Cell Vertex " + name + " (computed)",
    //                                  grid->vertexCoord_vec.size(),
    //                                  mTotalZTopLayer,
    //                                  valid_min,
    //                                  valid_max);
    // PrintValidationSummary("Cell Vertex " + name + " (computed)", result, attr_CtoV_vec.size());

    // // Enhanced validation: check if zeros are from land mask
    // if (result.zero_count > 0) {
    //     ValidateVertexVsCellData(attr_CtoV_vec, data_to_use, grid, name, mTotalZTopLayer);
    // }
}

void MPASOSolution::calcCellCenterToVertex(const std::string& name, const std::vector<double>& vec, MPASOGrid* grid, std::string& dataDir, const GPUContext& ctx)
{
    calcCellCenterToVertex(name, vec, grid, dataDir, RuntimeContext::FromGPU(ctx));
}

void MPASOSolution::calcCellCenterVelocity(MPASOGrid* grid, std::string& dataDir, const RuntimeContext& ctx)
{
    if (!cellCenterVelocity_vec.empty()) cellCenterVelocity_vec.clear();
    cellCenterVelocity_vec.resize(grid->mCellsSize * mTotalZTopLayer);

    auto cell_center_velocity_path = dataDir + "/" + "cellCenterVelocity_vec_" + std::to_string(mTimesteps) + ".bin";

    if (std::filesystem::exists(cell_center_velocity_path)) {
        readVertexZTopFromFile<vec3>(cellCenterVelocity_vec, cell_center_velocity_path);
        return;
    }

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(grid);
    MOPS::Factory::CalcCellCenterVelocity(
        grid,
        mCellsSize,
        mTotalZTopLayer,
        cellNormalVelocity_vec,
        cellCenterVelocity_vec,
        grid_info_vec,
        ctx);

    writeVertexZTopToFile<vec3>(cellCenterVelocity_vec, cell_center_velocity_path);
    Debug("[MPASOSolution]::Calc Cell VcellCenterVelocity_vec  = \t [ %d ] \t type = [ float64 float64 float64 ]",
            cellCenterVelocity_vec.size());
}

void MPASOSolution::calcCellCenterVelocity(MPASOGrid* grid, std::string& dataDir, const GPUContext& ctx)
{
    calcCellCenterVelocity(grid, dataDir, RuntimeContext::FromGPU(ctx));
}

void MPASOSolution::calcCellCenterVelocityByZM(MPASOGrid* grid, std::string& dataDir, const RuntimeContext& ctx)
{
    if (!cellCenterVelocity_vec.empty()) cellCenterVelocity_vec.clear();
    cellCenterVelocity_vec.resize(grid->mCellsSize * mTotalZTopLayer);

    auto cell_center_velocity_path = dataDir + "/" + "cellCenterVelocity_vec_" + std::to_string(mTimesteps) + ".bin";

    if (std::filesystem::exists(cell_center_velocity_path)) {
        readVertexZTopFromFile<vec3>(cellCenterVelocity_vec, cell_center_velocity_path);
        return;
    }

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(grid);
    MOPS::Factory::CalcCellCenterVelocityByZM(
        grid,
        mCellsSize,
        mTotalZTopLayer,
        cellZonalVelocity_vec,
        cellMeridionalVelocity_vec,
        cellCenterVelocity_vec,
        grid_info_vec,
        ctx);

    writeVertexZTopToFile<vec3>(cellCenterVelocity_vec, cell_center_velocity_path);
    Debug("[MPASOSolution]::Calc Cell VcellCenterVelocity_vec  = \t [ %d ] \t type = [ float64 float64 float64 ]",
            cellCenterVelocity_vec.size());
}

void MPASOSolution::calcCellCenterVelocityByZM(MPASOGrid* grid, std::string& dataDir, const GPUContext& ctx)
{
    calcCellCenterVelocityByZM(grid, dataDir, RuntimeContext::FromGPU(ctx));
}

void MPASOSolution::calcCellVertexVelocity(MPASOGrid* grid, std::string& dataDir, const RuntimeContext& ctx)
{
    if(!cellVertexVelocity_vec.empty()) cellVertexVelocity_vec.clear();
    cellVertexVelocity_vec.resize(mVertexSize * mTotalZTopLayer);

    auto cell_vertex_velocity_path = dataDir + "/" + "cellVertexVelocity_vec_" + std::to_string(mTimesteps) + ".bin";

    if (std::filesystem::exists(cell_vertex_velocity_path)) {
        readVertexZTopFromFile<vec3>(cellVertexVelocity_vec, cell_vertex_velocity_path);
        return;
    }

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(grid);
    MOPS::Factory::CalcCellVertexVelocity(
        grid,
        mCellsSize,
        mTotalZTopLayer,
        cellCenterVelocity_vec,
        cellVertexVelocity_vec,
        grid_info_vec,
        ctx);

    writeVertexZTopToFile<vec3>(cellVertexVelocity_vec, cell_vertex_velocity_path);
    Debug("[MPASOSolution]::Calc Cell cellVertexVelocity_vec  = \t [ %d ] \t type = [ float64 float64 float64 ]",
            cellVertexVelocity_vec.size());
}

void MPASOSolution::calcCellVertexVelocity(MPASOGrid* grid, std::string& dataDir, const GPUContext& ctx)
{
    calcCellVertexVelocity(grid, dataDir, RuntimeContext::FromGPU(ctx));
}

void MPASOSolution::calcCellVertexVelocityByZM(MPASOGrid* grid, std::string& dataDir, const RuntimeContext& ctx)
{
    if (!cellVertexVelocity_vec.empty()) cellVertexVelocity_vec.clear();
    cellVertexVelocity_vec.resize(grid->mVertexSize * mTotalZTopLayer);

    auto cell_vertex_velocity_path = dataDir + "/" + "cellVertexVelocity_vec_" + std::to_string(mTimesteps) + ".bin";

    if (std::filesystem::exists(cell_vertex_velocity_path)) {
        readVertexZTopFromFile<vec3>(cellVertexVelocity_vec, cell_vertex_velocity_path);
        return;
    }

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(grid);
    MOPS::Factory::CalcCellVertexVelocityByZM(
        grid,
        mVertexSize,
        mTotalZTopLayer,
        cellVertexZonalVelocity_vec,
        cellVertexMeridionalVelocity_vec,
        cellVertexVelocity_vec,
        grid_info_vec,
        ctx);

    writeVertexZTopToFile<vec3>(cellVertexVelocity_vec, cell_vertex_velocity_path);
    Debug("[MPASOSolution]::Calc Cell VertexVelocity_vec  = \t [ %d ] \t type = [ float64 float64 float64 ]",
            cellVertexVelocity_vec.size());
}

void MPASOSolution::calcCellVertexVelocityByZM(MPASOGrid* grid, std::string& dataDir, const GPUContext& ctx)
{
    calcCellVertexVelocityByZM(grid, dataDir, RuntimeContext::FromGPU(ctx));
}

void MPASOSolution::calcCellVertexVertVelocity(MPASOGrid* grid, std::string& dataDir, const RuntimeContext& ctx)
{
    if (mTotalZTopLayer != 0)
        mTotalZTopLayerP1 = mTotalZTopLayer + 1;

    if(!cellVertexVertVelocity_vec.empty()) cellVertexVertVelocity_vec.clear();
    cellVertexVertVelocity_vec.resize(mVertexSize * (mTotalZTopLayerP1));

    auto cell_vertex_vert_velocity_path = dataDir + "/" + "cellVertexVertVelocity_vec_" + std::to_string(mTimesteps) + ".bin";

    if (std::filesystem::exists(cell_vertex_vert_velocity_path)) {
        readVertexZTopFromFile<double>(cellVertexVertVelocity_vec, cell_vertex_vert_velocity_path);
        return;
    }

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(grid);
    MOPS::Factory::CalcCellVertexVertVelocity(
        grid,
        mCellsSize,
        mTotalZTopLayerP1,
        cellVertVelocity_vec,
        cellVertexVertVelocity_vec,
        grid_info_vec,
        ctx);

    writeVertexZTopToFile<double>(cellVertexVertVelocity_vec, cell_vertex_vert_velocity_path);
    Debug("[MPASOSolution]::Calc Cell cellVertexVertVelocity_vec  = \t [ %d ] \t type = [ float64 ]",
            cellVertexVertVelocity_vec.size());
}

void MPASOSolution::calcCellVertexVertVelocity(MPASOGrid* grid, std::string& dataDir, const GPUContext& ctx)
{
    calcCellVertexVertVelocity(grid, dataDir, RuntimeContext::FromGPU(ctx));
}

void MPASOSolution::initSolution_DemoLoading(const char* yaml_path, int timestep)
{
    std::shared_ptr<ftk::stream> stream(new ftk::stream);
	stream->parse_yaml(yaml_path);
	auto vel_path = stream->substreams[1]->filenames[0];
	this->gt = stream->read(timestep);
    this->initSolution(this->gt.get(), MOPS::MPASOReader::readSolInfo(vel_path, timestep).get());

}

void MPASOSolution::initSolution_FromBin(const char* prefix)
{
  



    readFromBlock_Vec3(std::string(prefix) + "Horizontal_vector.bin", cellVertexVelocity_vec);
    readFromBlock_DoubleBasedK(std::string(prefix) + "ZTop.bin", cellVertexZTop_vec);
    readFromBlock_DoubleBasedK(std::string(prefix) + "LayerThickness.bin", cellLayerThickness_vec);
    

    Debug("[MPASOSolution]::cellVertexZTop_vec size = %zu", cellVertexZTop_vec.size());
    Debug("[MPASOSolution]::cellVertexVelocity_vec size = %zu", cellVertexVelocity_vec.size());
    Debug("[MPASOSolution]::cellVertexMeridionalVelocity_vec size = %zu", cellVertexMeridionalVelocity_vec.size());

}


void MPASOSolution::initSolution(MPASOReader* reader)
{
    
    this->mTimeStamp = std::move(reader->mTimeStamp);
    this->mVertLevels = std::move(reader->mVertLevels);
    this->mVertLevelsP1 = std::move(reader->mVertLevelsP1);
    this->mTimesteps = std::move(reader->mTimesteps);
    this->mDataName = std::move(reader->mDataName);

    this->gt = std::move(reader->mGroupT);
    
    this->cellBottomDepth_vec = std::move(reader->cellBottomDepth_vec);
    this->cellSurfaceHeight_vec = std::move(reader->cellSurfaceHeight_vec);
    this->cellZonalVelocity_vec = std::move(reader->cellZonalVelocity_vec);
    this->cellMeridionalVelocity_vec = std::move(reader->cellMeridionalVelocity_vec);
    this->cellLayerThickness_vec = std::move(reader->cellLayerThickness_vec);
    this->cellZTop_vec = std::move(reader->cellZTop_vec);
    this->cellNormalVelocity_vec = std::move(reader->cellNormalVelocity_vec);
    this->cellVertVelocity_vec = std::move(reader->cellVertVelocity_vec);

    mID.timeStamp = this->mTimeStamp;
    mID.timestep = this->mTimesteps;

    cellVertexZTop_vec.resize(0);
    cellVertexMeridionalVelocity_vec.resize(0);
    cellVertexZonalVelocity_vec.resize(0);
    cellVertexVertVelocity_vec.resize(0);
    cellCenterVelocity_vec.resize(0);
    cellVertexVelocity_vec.resize(0);


    // std::cout << "mTimeStamp = " << this->mTimeStamp << std::endl;
    // std::cout << "mVertLevels = " << this->mVertLevels << std::endl;
    // std::cout << "mVertLevelsP1 = " << this->mVertLevelsP1 << std::endl;
    // std::cout << "cellBottomDepth_vec size = " << this->cellBottomDepth_vec.size() << std::endl;
    // std::cout << "cellSurfaceHeight_vec size = " << this->cellSurfaceHeight_vec.size() << std::endl;    
    // std::cout << "cellZonalVelocity_vec size = " << this->cellZonalVelocity_vec.size() << std::endl;
    // std::cout << "cellMeridionalVelocity_vec size = " << this->cellMeridionalVelocity_vec.size() << std::endl;
    // std::cout << "cellLayerThickness_vec size = " << this->cellLayerThickness_vec.size() << std::endl;
    // std::cout << "cellZTop_vec size = " << this->cellZTop_vec.size() << std::endl;
    // std::cout << "cellNormalVelocity_vec size = " << this->cellNormalVelocity_vec.size() << std::endl;
    // std::cout << "cellVertVelocity_vec size = " << this->cellVertVelocity_vec.size() << std::endl;
}

void MPASOSolution::initSolution(ftk::ndarray_group* g, MPASOReader* reader)
{
    Debug("[MPASOSolution]::===========================================");
    
    // this->mTimeStamp = std::move(reader->currentTimestep);
    this->mCellsSize = reader->mCellsSize;
    this->mEdgesSize = reader->mEdgesSize;
    this->mMaxEdgesSize = reader->mMaxEdgesSize;
    this->mVertexSize = reader->mVertexSize;
    this->mTimesteps = reader->mTimesteps;
    this->mVertLevels = reader->mVertLevels;
    this->mVertLevelsP1 = reader->mVertLevelsP1;

    this->gt = std::move(reader->mGroupT);

    std::vector<char> time_vec_s;
    std::vector<char> time_vec_e;
   
    Debug("[MPASOSolution]::initSolution::timestep = %d", this->mTimesteps);
    copyFromNdarray_Double(g, "bottomDepth", this->cellBottomDepth_vec);
    copyFromNdarray_Double(g, "velocityZonal", this->cellZonalVelocity_vec);
    copyFromNdarray_Double(g, "velocityMeridional", this->cellMeridionalVelocity_vec);
    copyFromNdarray_Double(g, "layerThickness", this->cellLayerThickness_vec);
    copyFromNdarray_Double(g, "timeMonthly_avg_zTop", this->cellZTop_vec);
    copyFromNdarray_Double(g, "normalVelocity", this->cellNormalVelocity_vec);
    copyFromNdarray_Double(g, "vertVelocityTop", this->cellVertVelocity_vec);
    mID.timeStamp = this->mTimeStamp;
    mID.timestep = this->mTimesteps;
   
    
    // copyFromNdarray_Char(g, "xtime_startMonthly", time_vec_s);
    // copyFromNdarray_Char(g, "xtime_endMonthly", time_vec_e);


    // std::cout << "time_vec_s.size() = " << time_vec_s.size() << std::endl;
    // std::cout << "time_vec_e.size() = " << time_vec_e.size() << std::endl;
    
    
    
    // std::cout << " bottomDepth size = " << this->cellBottomDepth_vec.size() << std::endl;
    // std::cout << " surfaceVelocityMeridional size = " << this->cellMeridionalVelocity_vec.size() << std::endl;
    // std::cout << " surfaceVelocityZonal size = " << this->cellZonalVelocity_vec.size() << std::endl;
    // std::cout << " layerThickness size = " << this->cellLayerThickness_vec.size() << std::endl;
    // std::cout << " zTop size = " << this->cellZTop_vec.size() << std::endl;
    // std::cout << " normalVelocity size = " << this->cellNormalVelocity_vec.size() << std::endl;

    // std::cout << "==========================================\n";
    // std::cout << "zonal velocity size = " << this->cellZonalVelocity_vec.size() << std::endl;
    // std::cout << "meridional velocity size = " << this->cellMeridionalVelocity_vec.size() << std::endl;
    
    // //ReadZonalVelocity(timestep, cellZonalVelocity_vec);



    // //ReadVelocity(timestep, cellVelocity_vec);
    // //ReadVertVelocityTop(timestep, cellVertVelocity_vec);
}


void MPASOSolution::addAttribute(std::string name, AttributeFormat type)
{
    if (this->gt == nullptr)
    {
        Error("[MPASOSolution]::gt is not initialized");
        return;
    }

    if (type == AttributeFormat::kDouble)
    {
        std::vector<double> vec;
        copyFromNdarray_Double(this->gt.get(), name, vec);
        this->mDoubleAttributes[name] = vec;
    }
    else if (type == AttributeFormat::kFloat)
    {
        std::vector<double> vec;
        copyFromNdarray_Double(this->gt.get(), name, vec);
        this->mDoubleAttributes[name] = vec;
    }
    else if (type == AttributeFormat::kChar)
    {
        std::vector<char> vec;
        copyFromNdarray_Char(this->gt.get(), name, vec);
        this->mCharAttributes[name] = vec;
    }
    else if (type == AttributeFormat::kVec3)
    {
        Debug("[MPASOSolution]::kVec3 is not supported temporarily");
    }

    // Standardize and clean invalid values for temperature and salinity
    // Different datasets may use different invalid markers (e.g., -1e30, -9999, NaN)
    // We standardize all invalid values to a consistent internal marker (INVALID_VALUE)
    // Then replace invalid cell values with nearest valid neighbor values
    if (name == "temperature" || name == "salinity")
    {
        // Step 1: Standardize invalid markers
        if (mDoubleAttributes.count(name))
        {
            auto& data = mDoubleAttributes[name];
            for (auto& val : data)
            {
                // If value is below threshold or NaN, mark as invalid
                if (val <= INVALID_VALUE_THRESHOLD || std::isnan(val))
                {
                    val = INVALID_VALUE;  // Standardized invalid marker
                }
            }
        }

        if (mFloatAttributes.count(name))
        {
            auto& data = mFloatAttributes[name];
            for (auto& val : data)
            {
                if (val <= static_cast<float>(INVALID_VALUE_THRESHOLD) || std::isnan(val))
                {
                    val = static_cast<float>(INVALID_VALUE);
                }
            }
        }

        // Step 2: Replace invalid cell center values with nearest valid neighbors
        // NOTE: This will be done in calcCellCenterToVertex, where grid is available
        // (Removed cleanInvalidCellCenterValues call from here)
    }
}

void MPASOSolution::cleanInvalidCellCenterValues(const std::string& name, MPASOGrid* grid)
{
    // This function replaces invalid cell center values with nearest valid neighbor values
    // Must be called with valid grid (requires cellsOnCell connectivity)

    if (!mDoubleAttributes.count(name)) {
        return;  // No data to clean
    }

    if (grid == nullptr || mCellsSize == 0 || mTotalZTopLayer == 0) {
        Debug("[MPASOSolution]::cleanInvalidCellCenterValues - grid not initialized, skipping");
        return;
    }

    auto& data = mDoubleAttributes[name];
    const size_t nCells = static_cast<size_t>(mCellsSize);
    const size_t nLayers = static_cast<size_t>(mTotalZTopLayer);

    // Get cellsOnCell connectivity from grid
    // Format: (nCells, maxEdges), 1-based indexing, 0 means no neighbor
    const std::vector<size_t>& cells_on_cell = grid->cellsOnCell_vec;
    const std::vector<vec3>& cell_coord = grid->cellCoord_vec;

    if (cells_on_cell.empty() || cell_coord.empty()) {
        Debug("[MPASOSolution]::cleanInvalidCellCenterValues - grid data not available");
        return;
    }

    const size_t max_edges = cells_on_cell.size() / nCells;

    int replaced_count = 0;

    // Process each cell and layer
    for (size_t cell_id = 0; cell_id < nCells; ++cell_id) {
        for (size_t layer = 0; layer < nLayers; ++layer) {
            const size_t idx = cell_id * nLayers + layer;
            double& value = data[idx];

            // Check if value is invalid
            if (value <= INVALID_VALUE + 1.0) {
                // Find nearest valid neighbor
                double nearest_value = INVALID_VALUE;
                double min_dist = 1e30;

                const vec3 cell_pos = cell_coord[cell_id];

                // Search through neighbors
                for (size_t i = 0; i < max_edges; ++i) {
                    const size_t raw_neighbor_id = cells_on_cell[cell_id * max_edges + i];
                    if (raw_neighbor_id == 0) {
                        continue;  // No neighbor
                    }

                    const size_t neighbor_id = raw_neighbor_id - 1;  // Convert to 0-based
                    if (neighbor_id >= nCells) {
                        continue;  // Invalid
                    }

                    // Get neighbor's value at same layer
                    const double neighbor_value = data[neighbor_id * nLayers + layer];

                    // Check if neighbor is valid
                    if (neighbor_value > INVALID_VALUE + 1.0) {
                        // Calculate distance
                        const vec3 neighbor_pos = cell_coord[neighbor_id];
                        const double dx = cell_pos.x() - neighbor_pos.x();
                        const double dy = cell_pos.y() - neighbor_pos.y();
                        const double dz = cell_pos.z() - neighbor_pos.z();
                        const double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

                        // Update if closer
                        if (dist < min_dist) {
                            min_dist = dist;
                            nearest_value = neighbor_value;
                        }
                    }
                }

                // Replace if found valid neighbor
                if (nearest_value > INVALID_VALUE + 1.0) {
                    value = nearest_value;
                    replaced_count++;
                } else {
                    // No valid neighbors found, set to 0
                    value = 0.0;
                }
            }
        }
    }

    Debug("[MPASOSolution]::cleanInvalidCellCenterValues - %s: replaced %d invalid values",
          name.c_str(), replaced_count);
}

void MPASOSolution::getCellVelocity(const size_t cell_id, const size_t level, std::vector<vec3>& cell_on_velocity, vec3& vel)
{
    auto VertLevels = mVertLevels;
    if (VertLevels == -1 || VertLevels == 0)
    {
        Debug("ERROR, VertLevels is not defined");
    }
    auto idx = VertLevels * cell_id + level;
    vel = cell_on_velocity[idx];
}

void MPASOSolution::getCellVertVelocity(const size_t cell_id,
    const size_t level,
    std::vector<double>& cell_vert_velocity,
    double& vel)
{
    auto VertLevelsP1 = mVertLevelsP1;
    if (VertLevelsP1 == -1 || VertLevelsP1 == 0)
    {
        Debug("ERROR, VertLevelsP1 is not defined");
    }
    auto idx = VertLevelsP1 * cell_id + level;
    vel = cell_vert_velocity[idx];
}

void MPASOSolution::getCellZTop(const size_t cell_id, const size_t level, std::vector<double>& cell_ztop, double& ztop)
{
    // Cell 0, the top Z coordinate of layer 0
    // Cell 0, the top Z coordinate of layer 1
    // Cell 0, the top Z coordinate of layer 2
    // Cell 1, the top Z coordinate of layer 0
    // Cell 1, the top Z coordinate of layer 1
    // Cell 1, the top Z coordinate of layer 2
    auto VertLevels = mVertLevels;
    if (VertLevels == -1 || VertLevels == 0)
    {
        Debug("ERROR, VertLevels is not defined");
    }
    auto idx = VertLevels * cell_id + level;
    ztop = cell_ztop[idx];
}

void MPASOSolution::getEdgeNormalVelocity(const size_t edge_id, const size_t level, std::vector<double>& edge_normal_velocity, double& vel)
{
    auto VertLevels = mVertLevels;
    if (VertLevels == -1 || VertLevels == 0)
    {
        Debug("ERROR, VertLevels is not defined");
    }
    auto idx = VertLevels * edge_id + level;
    vel = edge_normal_velocity[idx];
}

void MPASOSolution::getCellLayerThickness(const size_t cell_id, const size_t level, std::vector<double>& cell_thickness, double& thinckness)
{
    // Cell 0, the thickness of layer 0
    // Cell 0, the thickness of layer 1
    // Cell 0, the thickness of layer 2
    // Cell 1, the thickness of layer 0
    // Cell 1, the thickness of layer 1
    // Cell 1, the thickness of layer 2

    auto VertLevels = mVertLevels;
    if (VertLevels == -1 || VertLevels == 0)
    {
        Debug("ERROR, VertLevels is not defined");
    }
    auto idx = VertLevels * cell_id + level;
    thinckness = cell_thickness[idx];
}

void MPASOSolution::getCellSurfaceMeridionalVelocity(const size_t cell_id, std::vector<double>& cell_meridional_velocity, double& vel)
{
    vel = cell_meridional_velocity[cell_id];
}

void MPASOSolution::getCellSurfaceZonalVelocity(const size_t cell_id, std::vector<double>& cell_zonal_velocity, double& vel)
{
    vel = cell_zonal_velocity[cell_id];
}

void MPASOSolution::getCellCenterZTop(const size_t cell_id, const size_t level, std::vector<double>& cell_ztop, double& ztop)
{
    // Cell 0, the top Z coordinate of layer 0
    // Cell 0, the top Z coordinate of layer 1
    // Cell 0, the top Z coordinate of layer 2
    // Cell 1, the top Z coordinate of layer 0
    // Cell 1, the top Z coordinate of layer 1
    // Cell 1, the top Z coordinate of layer 2

    auto VertLevels = mVertLevels;
    if (VertLevels == -1 || VertLevels == 0)
    {
        Debug("ERROR, VertLevels is not defined");
    }
    auto idx = VertLevels * cell_id + level;
    ztop = cell_ztop[idx];
}

void MPASOSolution::getCellVertexZTop(const size_t vertex_id, const size_t level, std::vector<double>& cell_vertex_ztop, double& ztop)
{
    ztop = cell_vertex_ztop[vertex_id * mVertLevels + level];
}

void MPASOSolution::getCellCenterVelocity(const size_t cell_id, const size_t level, std::vector<vec3>& cell_on_velocity, vec3& vel)
{
    auto VertLevels = mVertLevels;
    if (VertLevels == -1 || VertLevels == 0)
    {
        Debug("ERROR, VertLevels is not defined");
    }
    auto idx = VertLevels * cell_id + level;
    vel = cell_on_velocity[idx];
}

void MPASOSolution::getCellVertexVelocity(const size_t vertex_id, const size_t level, std::vector<vec3>& cell_vertex_velocity, vec3& vel)
{
    vel = cell_vertex_velocity[vertex_id * mVertLevels + level];
}

void MPASOSolution::calcCellCenterZtop()
{
    Debug(("[MPASOSolution]::Calc Cell Center Z Top at t = " + std::to_string(mTimesteps)).c_str());

    // 1. Check if layerThickness exists
    if (cellLayerThickness_vec.empty())
    {
        Debug("ERROR, cellLayerThickness is not defined");
        exit(0);
    }

    // 2. Calculate ZTop
    auto nCellsSize = mCellsSize;
    auto nVertLevelsP1      =  mVertLevelsP1;
    auto nVertLevels = mVertLevels;
    auto nTimesteps = mTimesteps;

    // std::cout << "=======\n";
    // std::cout << "nCellsSize = " << nCellsSize << std::endl;
    // std::cout << "nVertLevelsP1 = " << nVertLevelsP1 << std::endl;
    // std::cout << "nVertLevels = " << nVertLevels << std::endl;
    // std::cout << "nTimesteps = " << nTimesteps << std::endl;

    bool hasBottomDepth = !cellBottomDepth_vec.empty();
    bool hasSurfaceHeight = !cellSurfaceHeight_vec.empty();

    if (hasBottomDepth && hasSurfaceHeight)
        hasSurfaceHeight = false;                   // use bottom depth only    

    if (!cellZTop_vec.empty()) cellZTop_vec.clear();
    cellZTop_vec.resize(nCellsSize * nVertLevels);

    for (size_t i = 0; i < nCellsSize; ++i)
    {
        if (hasBottomDepth)
        {
            // Calculate from bottom to top
            double z = -cellBottomDepth_vec[i];
            for (int k = nVertLevels - 1; k >= 0; --k) {
                double layerThickness;
                getCellLayerThickness(i, k, cellLayerThickness_vec, layerThickness);
                z += layerThickness;
                cellZTop_vec[i * nVertLevels + k] = z;
            }
        }
        else if (hasSurfaceHeight)
        {
            // Calculate from top to bottom
            double z = cellSurfaceHeight_vec[i];
            cellZTop_vec[i * nVertLevels] = z; 
            for (size_t j = 1; j < nVertLevels; ++j)
            {
                double layerThickness;
                getCellLayerThickness(i, j - 1, cellLayerThickness_vec, layerThickness);
                z -= layerThickness;
                cellZTop_vec[i * nVertLevels + j] = z;
            }

            // DEBUG: Print cell 26664 depth and layer thickness
            // if (i == 26664) {
            //     printf("\n[CELL DEPTH DEBUG] Cell %zu:\n", i);
            //     printf("  Total layers: %zu\n", nVertLevels);
            //     printf("  Layer structure (from surface to bottom):\n");
            //     printf("  %-6s %-15s %-15s %-15s\n", "Layer", "ZTop (m)", "Thickness (m)", "Status");
            //     printf("  %s\n", std::string(55, '-').c_str());

            //     for (size_t layer = 0; layer < nVertLevels; ++layer) {
            //         double ztop = cellZTop_vec[i * nVertLevels + layer];
            //         double thickness = 0.0;
            //         if (layer < nVertLevels - 1) {
            //             double ztop_next = cellZTop_vec[i * nVertLevels + layer + 1];
            //             thickness = ztop - ztop_next;  // positive value
            //         } else {
            //             // Last layer - get thickness from raw data
            //             getCellLayerThickness(i, layer, cellLayerThickness_vec, thickness);
            //         }

            //         const char* status = "";
            //         if (thickness < 1e-6) {
            //             status = " (MASKED/ZERO)";
            //         }

            //         printf("  %-6zu %-15.2f %-15.2f%s\n",
            //                layer, ztop, thickness, status);
            //     }

            //     double bottom_depth = -cellZTop_vec[i * nVertLevels + nVertLevels - 1];
            //     printf("\n  Bottom depth: %.2f m\n\n", bottom_depth);
            // }

        }
        else
        {
            // Assume surface height is 0
            cellZTop_vec[i * nVertLevels] = 0.0;
            for (size_t j = 1; j < nVertLevels; ++j)
            {
                double layerThickness;
                getCellLayerThickness(i, j - 1, cellLayerThickness_vec, layerThickness);
                cellZTop_vec[i * nVertLevels + j] = cellZTop_vec[i * nVertLevels + j - 1] - layerThickness;
            }
        }
        
    }

    for (auto i = 0; i < cellZTop_vec.size(); ++i)
    {
        cellZTop_vec[i] *= 1.0;
    }

    mTotalZTopLayer = nVertLevels;

    // std::cout << "cell ZTOP_vec " << cellZTop_vec[0] << std::endl;
    // exit(-1);
    // std::cout << "cellZTop_vec.size() = " << cellZTop_vec.size() << std::endl;
    // std::cout << "nCellSize x nVertLevels = " << nCellsSize * nVertLevels << std::endl;
}


template<typename T>
void writeVertexZTopToFile(const std::vector<T>& vertexZTop_vec, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (outFile) {
        size_t size = vertexZTop_vec.size();
        outFile.write(reinterpret_cast<const char*>(&size), sizeof(size)); // Write the size of the vector
        outFile.write(reinterpret_cast<const char*>(vertexZTop_vec.data()), size * sizeof(T)); // Write the data
        Debug("[MPASOSolution]::Wrote %zu elements to %s", size, filename.c_str());
        outFile.close();
    }
    else {
        Error("[MPASOSolution]::Unable to open file for writing: %s", filename.c_str());
    }
}

template<typename T>
void readVertexZTopFromFile(std::vector<T>& vertexZTop_vec, const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (inFile) {
        size_t size;
        inFile.read(reinterpret_cast<char*>(&size), sizeof(size)); // Read the size of the vector
        vertexZTop_vec.resize(size);
        inFile.read(reinterpret_cast<char*>(vertexZTop_vec.data()), size * sizeof(T)); // Read the data
        inFile.close();
        Debug("[MPASOSolution]::Read %zu elements from %s", size, filename.c_str());
    }
    else {
        Error("[MPASOSolution]::Unable to open file for reading: %s", filename.c_str());
    }
}


void saveDataToTextFile2(const std::vector<vec3>& data, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        Error("[MPASOSolution]::Failed to open file: %s", filename.c_str());
        return;
    }

    for (const auto& value : data) {
        outfile << value.x() << " " << value.y() << " " << value.x() << std::endl;
    }

    outfile.close();
    Debug("[MPASOSolution]::Data saved to %s", filename.c_str());
}

//TODO
void saveDataToTextFile(const std::vector<double>& data, const std::string& filename) {
    int MAX_VERTEX_NUM = 7;
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        Error("[MPASOSolution]::Failed to open file: %s", filename.c_str());
        return;
    }

    int count = 0;
    for (const auto& value : data) {
        outfile << value << " ";
        count++;
        if (count % MAX_VERTEX_NUM == 0) {
            outfile << std::endl;
        }
    }

    if (count % MAX_VERTEX_NUM != 0) {  // If the total number of data is not a multiple of 7, add a newline at the end of the file
        outfile << std::endl;
    }

    outfile.close();
    Debug("[MPASOSolution]::Data saved to %s", filename.c_str());
}


void saveDataToTextFile3(const std::vector<vec3>& data, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        Error("[MPASOSolution]::Failed to open file: %s", filename.c_str());
        return;
    }
    int MAX_VERTEX_NUM = 7;
    int count = 0;
    for (const auto& value : data) {
        outfile << value.x() << " ";
        count++;
        if (count % MAX_VERTEX_NUM == 0) {
            outfile << std::endl;
        }
    }

    if (count % MAX_VERTEX_NUM != 0) {  // If the total number of data is not a multiple of 7, add a newline at the end of the file
        outfile << std::endl;
    }

    outfile.close();
    Debug("[MPASOSolution]::Data saved to %s", filename.c_str());
}



void MPASOSolution::calcCellVertexZtop(MPASOGrid* grid, std::string& dataDir, sycl::queue& q)
{
    if (mTotalZTopLayer == 0 || mTotalZTopLayer == -1) mTotalZTopLayer = mVertLevels;
    if (!cellVertexZTop_vec.empty()) cellVertexZTop_vec.clear();
    cellVertexZTop_vec.resize(grid->vertexCoord_vec.size() * mTotalZTopLayer);

    auto cell_vertex_ztop_path = dataDir + "/" + "cellVertexZTop_vec_"  + std::to_string(mTimesteps) + ".bin";

    if (std::filesystem::exists(cell_vertex_ztop_path)) {
        readVertexZTopFromFile<double>(cellVertexZTop_vec, cell_vertex_ztop_path);
        return;
    }

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(grid);
    
#if USE_SYCL
    MOPS::GPUContext ctx = MOPS::GPUContext::FromSYCL(q);
    MOPS::Factory::CalcCellVertexZtop(
        grid,
        mCellsSize,
        mTotalZTopLayer,
        cellZTop_vec,
        cellVertexZTop_vec,
        grid_info_vec,
        RuntimeContext::FromGPU(ctx));

    writeVertexZTopToFile<double>(cellVertexZTop_vec, cell_vertex_ztop_path);
    Debug("[MPASOSolution]::Calc Cell Vertex Z Top  = \t [ %d ] \t type = [ float64 ]", 
            cellVertexZTop_vec.size());
#endif
  
}


void MPASOSolution::calcCellCenterToVertex(const std::string& name, const std::vector<double>& vec, MPASOGrid* grid, std::string& dataDir, sycl::queue& q)
{
    if (vec.empty()) return;
    if (mTotalZTopLayer == 0 || mTotalZTopLayer == -1) mTotalZTopLayer = mVertLevels;
    std::vector<double> attr_CtoV_vec;
    attr_CtoV_vec.resize(grid->vertexCoord_vec.size() * mTotalZTopLayer);

    auto cell_vertex_attribute_path = dataDir + "/" + "cellVertex" + name + "_vec_"  + std::to_string(mTimesteps) + ".bin";

    if (std::filesystem::exists(cell_vertex_attribute_path)) {
        readVertexZTopFromFile<double>(attr_CtoV_vec, cell_vertex_attribute_path);
        mDoubleAttributes_CtoV[name] = attr_CtoV_vec;
        return;
    }

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(grid);
    
#if USE_SYCL
    MOPS::GPUContext ctx = MOPS::GPUContext::FromSYCL(q);
    MOPS::Factory::CalcCellCenterToVertex(
        grid,
        mCellsSize,
        mTotalZTopLayer,
        vec,
        attr_CtoV_vec,
        grid_info_vec,
        RuntimeContext::FromGPU(ctx));

    writeVertexZTopToFile<double>(attr_CtoV_vec, cell_vertex_attribute_path);
    Debug("[MPASOSolution]::Calc Cell Vertex Attribute  = \t [ %d ] \t type = [ float64 ]", 
            attr_CtoV_vec.size());
#endif
    mDoubleAttributes_CtoV[name] = attr_CtoV_vec;
}




void MPASOSolution::calcCellCenterVelocity(MPASOGrid* grid, std::string& dataDir, sycl::queue& q)
{

    if (!cellCenterVelocity_vec.empty()) cellCenterVelocity_vec.clear();
    cellCenterVelocity_vec.resize(grid->mCellsSize * mTotalZTopLayer);

    auto cell_center_velocity_path = dataDir + "/" + "cellCenterVelocity_vec_" + std::to_string(mTimesteps) + ".bin";

    if (std::filesystem::exists(cell_center_velocity_path)) {
        readVertexZTopFromFile<vec3>(cellCenterVelocity_vec, cell_center_velocity_path);
        return;
    }

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(grid);

#if USE_SYCL
    MOPS::GPUContext ctx = MOPS::GPUContext::FromSYCL(q);
    MOPS::Factory::CalcCellCenterVelocity(
        grid,
        mCellsSize,
        mTotalZTopLayer,
        cellNormalVelocity_vec,
        cellCenterVelocity_vec,
        grid_info_vec,
        RuntimeContext::FromGPU(ctx));

    writeVertexZTopToFile<vec3>(cellCenterVelocity_vec, cell_center_velocity_path);
    Debug("[MPASOSolution]::Calc Cell VcellCenterVelocity_vec  = \t [ %d ] \t type = [ float64 float64 float64]", 
            cellCenterVelocity_vec.size());
#endif
}

void MPASOSolution::calcCellCenterVelocityByZM(MPASOGrid *grid, std::string& dataDir, sycl::queue &q)
{
    if (!cellCenterVelocity_vec.empty()) cellCenterVelocity_vec.clear();
    cellCenterVelocity_vec.resize(grid->mCellsSize * mTotalZTopLayer);

    auto cell_center_velocity_path = dataDir + "/" + "cellCenterVelocity_vec_" + std::to_string(mTimesteps) + ".bin";

    if (std::filesystem::exists(cell_center_velocity_path)) {
        readVertexZTopFromFile<vec3>(cellCenterVelocity_vec, cell_center_velocity_path);
        return;
    }

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(grid);

#if USE_SYCL
    MOPS::GPUContext ctx = MOPS::GPUContext::FromSYCL(q);
    MOPS::Factory::CalcCellCenterVelocityByZM(
        grid,
        mCellsSize,
        mTotalZTopLayer,
        cellZonalVelocity_vec,
        cellMeridionalVelocity_vec,
        cellCenterVelocity_vec,
        grid_info_vec,
        RuntimeContext::FromGPU(ctx));

    writeVertexZTopToFile<vec3>(cellCenterVelocity_vec, cell_center_velocity_path);
    Debug("[MPASOSolution]::Calc Cell VcellCenterVelocity_vec  = \t [ %d ] \t type = [ float64 float64 float64]", 
            cellCenterVelocity_vec.size());
#endif
}

void MPASOSolution::calcCellVertexVelocityByZM(MPASOGrid *grid, std::string& dataDir, sycl::queue &q)
{
    if (!cellVertexVelocity_vec.empty()) cellVertexVelocity_vec.clear();
    cellVertexVelocity_vec.resize(grid->mVertexSize * mTotalZTopLayer);

    auto cell_vertex_velocity_path = dataDir + "/" + "cellVertexVelocity_vec_" + std::to_string(mTimesteps) + ".bin";

    if (std::filesystem::exists(cell_vertex_velocity_path)) {
        readVertexZTopFromFile<vec3>(cellVertexVelocity_vec, cell_vertex_velocity_path);
        return;
    }

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(grid);

#if USE_SYCL
    MOPS::GPUContext ctx = MOPS::GPUContext::FromSYCL(q);
    MOPS::Factory::CalcCellVertexVelocityByZM(
        grid,
        mVertexSize,
        mTotalZTopLayer,
        cellVertexZonalVelocity_vec,
        cellVertexMeridionalVelocity_vec,
        cellVertexVelocity_vec,
        grid_info_vec,
        RuntimeContext::FromGPU(ctx));

    writeVertexZTopToFile<vec3>(cellVertexVelocity_vec, cell_vertex_velocity_path);
    Debug("[MPASOSolution]::Calc Cell VertexVelocity_vec  = \t [ %d ] \t type = [ float64 float64 float64]",
            cellVertexVelocity_vec.size());
#endif
}

void MPASOSolution::calcCellVertexVelocity(MPASOGrid* grid, std::string& dataDir, sycl::queue& q)
{
    if(!cellVertexVelocity_vec.empty()) cellVertexVelocity_vec.clear();
    cellVertexVelocity_vec.resize(mVertexSize * mTotalZTopLayer);

    auto cell_vertex_velocity_path = dataDir + "/" + "cellVertexVelocity_vec_" + std::to_string(mTimesteps) + ".bin";

    if (std::filesystem::exists(cell_vertex_velocity_path)) {
        readVertexZTopFromFile<vec3>(cellVertexVelocity_vec, cell_vertex_velocity_path);
        return;
    }

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(grid);

#if USE_SYCL
    MOPS::GPUContext ctx = MOPS::GPUContext::FromSYCL(q);
    MOPS::Factory::CalcCellVertexVelocity(
        grid,
        mCellsSize,
        mTotalZTopLayer,
        cellCenterVelocity_vec,
        cellVertexVelocity_vec,
        grid_info_vec,
        RuntimeContext::FromGPU(ctx));

    writeVertexZTopToFile<vec3>(cellVertexVelocity_vec, cell_vertex_velocity_path);
    Debug("[MPASOSolution]::Calc Cell cellVertexVelocity_vec  = \t [ %d ] \t type = [ float64 float64 float64]", 
            cellVertexVelocity_vec.size());
#endif

}

void MPASOSolution::calcCellVertexVertVelocity(MPASOGrid* grid, std::string& dataDir, sycl::queue& q)
{
    if (mTotalZTopLayer != 0)
        mTotalZTopLayerP1 = mTotalZTopLayer + 1;

    if(!cellVertexVertVelocity_vec.empty()) cellVertexVertVelocity_vec.clear();
    cellVertexVertVelocity_vec.resize(mVertexSize * (mTotalZTopLayerP1));

    auto cell_vertex_vert_velocity_path = dataDir + "/" + "cellVertexVertVelocity_vec_" + std::to_string(mTimesteps) + ".bin";

    if (std::filesystem::exists(cell_vertex_vert_velocity_path)) {
        readVertexZTopFromFile<double>(cellVertexVertVelocity_vec, cell_vertex_vert_velocity_path);
        return;
    }

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(grid);


#if USE_SYCL
    MOPS::GPUContext ctx = MOPS::GPUContext::FromSYCL(q);
    MOPS::Factory::CalcCellVertexVertVelocity(
        grid,
        mCellsSize,
        mTotalZTopLayerP1,
        cellVertVelocity_vec,
        cellVertexVertVelocity_vec,
        grid_info_vec,
        RuntimeContext::FromGPU(ctx));

    writeVertexZTopToFile<double>(cellVertexVertVelocity_vec, cell_vertex_vert_velocity_path);
    Debug("[MPASOSolution]::Calc Cell cellVertexVertVelocity_vec  = \t [ %d ] \t type = [ float64 ]", 
            cellVertexVertVelocity_vec.size());
#endif
}

void MPASOSolution::readFromBlock_Vec3(const std::string& filename, std::vector<vec3>& vec)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "[MPASOSolution]::Error: Unable to open file " << filename << std::endl;
        return;
    }

    vec.clear();
    int nNodeNum;
    file.read(reinterpret_cast<char*>(&nNodeNum), sizeof(int));
    vec.resize(nNodeNum);
    // VECTOR3 = 24 bytes (double[3])
    std::vector<double> vectorData(nNodeNum * 3);
    file.read(reinterpret_cast<char*>(vectorData.data()), nNodeNum * 3 * sizeof(double));
    for (int i = 0; i < nNodeNum; i++) 
    {
        double x = vectorData[i * 3 + 0];
        double y = vectorData[i * 3 + 1]; 
        double z = vectorData[i * 3 + 2];
        vec[i] = vec3{x, y, z};
    }
    file.close();
}

void MPASOSolution::readFromBlock_Double(const std::string& filename, std::vector<double>& vec)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "[MPASOSolution]::Error: Unable to open file " << filename << std::endl;
        return;
    }

    vec.clear();
    int dataSize;
    file.read(reinterpret_cast<char*>(&dataSize), sizeof(int));
    vec.resize(dataSize);
    for (std::size_t i = 0; i < dataSize; ++i)
    {
        double value;
        file.read(reinterpret_cast<char*>(&value), sizeof(double));
        vec[i] = value;
    }
    file.close();
    std::cout << "[MPASOSolution]::Info: Loaded " << filename << " with " << vec.size() << " entries." << std::endl;
}

void MPASOSolution::readFromBlock_DoubleBasedK(const std::string& filename, std::vector<double>& vec, int K)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "[MPASOSolution]::Error: Unable to open file " << filename << std::endl;
        return;
    }

    vec.clear();
    int dataSize;
    int k;
    file.read(reinterpret_cast<char*>(&dataSize), sizeof(int));
    if (K == -1)
        file.read(reinterpret_cast<char*>(&k), sizeof(int));
    else
        k = K;
    vec.resize(dataSize * k);
    for (int i = 0; i < dataSize * k; i++) 
    {
        double value;
        file.read(reinterpret_cast<char*>(&value), sizeof(double));
        vec[i] = value;
    }
    file.close();
    K = k;
    std::cout << "[MPASOSolution]::Info: Loaded " << filename << " with " << dataSize << " entries and " << k << " components each." << std::endl;
}

void MPASOSolution::copyFromNdarray_Double(ftk::ndarray_group* g, std::string value, std::vector<double>& vec)
{
    if (g->has(value))
    {
        Debug("[Ndarray]::%s found [✓]", value.c_str());
        auto tmp_get = g->get(value);
        // std::cout << "tmp_get = " << tmp_get.get() << std::endl;
        //  std::cout << "Actual type of tmp_get: " << typeid(*tmp_get).name() << std::endl;
        auto tmp_ptr_float = std::dynamic_pointer_cast<ftk::ndarray<float>>(g->get(value));
        if (tmp_ptr_float) 
        {

            // std::cout << "tmp_ptr = " << tmp_ptr.get() << std::endl;
            auto tmp_vec = tmp_ptr_float->std_vector();
            // std::cout << "tmp_vec.size() = " << tmp_vec.size() << std::endl;
            vec.resize(tmp_vec.size());
            for (auto i = 0; i < tmp_vec.size(); i++)
                vec[i] = static_cast<double>(tmp_vec[i]);
            Debug("[Ndarray]::loading %-10s_vec = \t [ %-8d ] \t type = [ %-10s ]",
                value.c_str(),
                vec.size(), 
                ftk::ndarray_base::dtype2str(tmp_get->type()).c_str());
            return;
        }
        auto tmp_ptr_double = std::dynamic_pointer_cast<ftk::ndarray<double>>(tmp_get);
        if (tmp_ptr_double)
        {
            auto tmp_vec = tmp_ptr_double->std_vector();
            // std::cout << "tmp_vec.size() = " << tmp_vec.size() << std::endl;
            vec.resize(tmp_vec.size());
            for (auto i = 0; i < tmp_vec.size(); i++)
                vec[i] = tmp_vec[i];
            Debug("[Ndarray]::loading %-10s_vec = \t [ %-8d ] \t type = [ %-10s ]",
                value.c_str(),
                vec.size(), 
                ftk::ndarray_base::dtype2str(tmp_get->type()).c_str());
            return;
        }
        std::cerr << "[Error]: The value found is not of type ftk::ndarray<double>" << std::endl;
    }
    else
    {
        Debug("[Ndarray]::%s not found [\u2717]", value.c_str());
    }

}

void MPASOSolution::copyFromNdarray_Char(ftk::ndarray_group* g, std::string value, std::vector<char>& vec)
{
    if (g->has(value))
    {
        Debug("[Ndarray]::%s found [✓]", value.c_str());
        auto tmp_get = g->get(value);
        auto tmp_ptr = std::dynamic_pointer_cast<ftk::ndarray<char>>(g->get(value));
        if (tmp_ptr) 
        {
            auto tmp_vec = tmp_ptr->std_vector();
            vec.resize(tmp_vec.size());
            for (auto i = 0; i < tmp_vec.size(); i++)
                vec[i] = static_cast<double>(tmp_vec[i]);
            Debug("[Ndarray]::loading _vec %-30s= \t [ %-8d ] \t type = [ %-10s ]",
                value.c_str(),
                vec.size(), 
                ftk::ndarray_base::dtype2str(tmp_get->type()).c_str());
        }
        else
        {
            Error("[Ndarray]::The value found is not of type ftk::ndarray<char>");
        }
    }
    else
    {
        Debug("[Ndarray]::%s not found [\u2717]", value.c_str());
    }

}

void MPASOSolution::copyFromNdarray_Float(ftk::ndarray_group* g, std::string value, std::vector<float>& vec)
{
    if (g->has(value))
    {
        Debug("[Ndarray]::%s found [✓]", value.c_str());
        auto tmp_get = g->get(value);
       
        auto tmp_ptr = std::dynamic_pointer_cast<ftk::ndarray<float>>(g->get(value));
        if (tmp_ptr) 
        {

            // std::cout << "tmp_ptr = " << tmp_ptr.get() << std::endl;
            auto tmp_vec = tmp_ptr->std_vector();
            // std::cout << "tmp_vec.size() = " << tmp_vec.size() << std::endl;
            vec.resize(tmp_vec.size());
            for (auto i = 0; i < tmp_vec.size(); i++)
                vec[i] = static_cast<float>(tmp_vec[i]);
            Debug("[Ndarray]::loading %-30s_vec = \t [ %-8d ] \t type = [ %-10s ]",
                value.c_str(),
                vec.size(), 
                ftk::ndarray_base::dtype2str(tmp_get->type()).c_str());
        }
        else
        {
            std::cerr << "[Error]: The value found is not of type ftk::ndarray<double>" << std::endl;
        }
    }
    else
    {
        Debug("[Ndarray]::%s not found [\u2717]", value.c_str());
    }

}

void MPASOSolution::setAttribute(GridAttributeType type, int val)
{
    switch (type)
    {
        case GridAttributeType::kCellSize:
            mCellsSize = val;
            break;
        case GridAttributeType::kEdgeSize:
            mEdgesSize = val;
            break;
        case GridAttributeType::kVertexSize:
            mVertexSize = val;
            break;
        case GridAttributeType::kMaxEdgesSize:
            mMaxEdgesSize = val;
            break;
        case GridAttributeType::kVertLevels:
            mVertLevels = val;
            break;
        case GridAttributeType::kVertLevelsP1:
            mVertLevelsP1 = val;
            break;
        default:
            std::cerr << "[Error]: Invalid GridAttributeType" << std::endl;
            break;      
    }
}
void MPASOSolution::setAttributesVec3(AttributeType type, const std::vector<vec3>& vec)
{
    switch (type)
    {
        case AttributeType::kVelocity:
            cellCenterVelocity_vec = vec;
            break;
        default:
            std::cerr << "[Error]: Invalid AttributeType" << std::endl;
            break;
    }
}
void MPASOSolution::setAttributesDouble(AttributeType type, const std::vector<double>& vec)
{
    switch (type)
    {
        case AttributeType::kZTop:
            cellZTop_vec = vec;
            break;
        case AttributeType::kLayerThickness:
            cellLayerThickness_vec = vec;
            break;
        case AttributeType::kBottomDepth:
            cellBottomDepth_vec = vec;
            break;
        case AttributeType::kZonalVelocity:
            cellZonalVelocity_vec = vec;
            break;
        case AttributeType::kMeridionalVelocity:
            cellMeridionalVelocity_vec = vec;
            break;
        case AttributeType::kNormalVelocity:
            cellNormalVelocity_vec = vec;
            break;
        default:
            std::cerr << "[Error]: Invalid AttributeType" << std::endl;
            break;
    }
}

bool MPASOSolution::checkAttribute()
{
    // 1. check velocity
    // must have cellCenterVelocity_vec or (cellMeridionalVelocity_vec, cellZonalVelocity_vec), or (cellNormalVelocity_vec)
    // if (cellCenterVelocity_vec.empty() &&
    //     !( !cellMeridionalVelocity_vec.empty() && !cellZonalVelocity_vec.empty() ) &&

    //     cellNormalVelocity_vec.empty())
    // {
    //     std::cerr << "[MPASOSolution]::Error: Invalid Velocity Attribute" << std::endl;
    //     return false;
    // }
    // 2. check ztop
    // must have cellZTop_vec or (kLayerThickness, kBottomDepth), or kLayerThickness
    if (cellZTop_vec.empty() && cellLayerThickness_vec.empty())
    {
        std::cerr << "[MPASOSolution]::Error: Invalid ZTop Attribute" << std::endl;
        return false;
    }
    
    return true;
}