#include "Core/MPASOSolution.h"
#include "Utils/Interpolation.hpp"
#include "Utils/GeoConverter.hpp"
#include "SYCL/SYCLKernel.h"


using namespace MOPS;

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
    

    std::cout << "cellVertexZTop_vec size " << cellVertexZTop_vec.size() << std::endl;
    std::cout << "cellVertexVelocity_vec size " << cellVertexVelocity_vec.size() << std::endl;
    std::cout << "cellVertexMeridionalVelocity_vec size " << cellVertexMeridionalVelocity_vec.size() << std::endl;

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

    mID.timeStamp = this->mTimeStamp;
    mID.timestep = this->mTimesteps;


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

}

void MPASOSolution::initSolution(ftk::ndarray_group* g, MPASOReader* reader)
{
    std::cout << "==========================================\n";
    
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
   
    std::cout << "          [ MPASOSolution::initSolution::timestep = " << this->mTimesteps << " ]\n";
    copyFromNdarray_Double(g, "bottomDepth", this->cellBottomDepth_vec);
    copyFromNdarray_Double(g, "velocityZonal", this->cellZonalVelocity_vec);
    copyFromNdarray_Double(g, "velocityMeridional", this->cellMeridionalVelocity_vec);
    copyFromNdarray_Double(g, "layerThickness", this->cellLayerThickness_vec);
    copyFromNdarray_Double(g, "timeMonthly_avg_zTop", this->cellZTop_vec);
    copyFromNdarray_Double(g, "normalVelocity", this->cellNormalVelocity_vec);

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
        std::cerr << "[MPASOSolution]]::Error: gt is not initialized" << std::endl;
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
        std::cout << "kVec3 is not supported temporarily" << std::endl;
    }

   
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
        std::cout << "Wrote " << size << " elements to " << filename << std::endl;
        outFile.close();
    }
    else {
        std::cerr << "Unable to open file for writing: " << filename << std::endl;
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
        std::cout << "Read " << size << " elements from " << filename << std::endl;
    }
    else {
        std::cerr << "Unable to open file for reading: " << filename << std::endl;
    }
}


void saveDataToTextFile2(const std::vector<vec3>& data, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    for (const auto& value : data) {
        outfile << value.x() << " " << value.y() << " " << value.x() << std::endl;
    }

    outfile.close();
    std::cout << "Data saved to " << filename << std::endl;
}

//TODO
void saveDataToTextFile(const std::vector<double>& data, const std::string& filename) {
    int MAX_VERTEX_NUM = 7;
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
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
    std::cout << "Data saved to " << filename << std::endl;
}


void saveDataToTextFile3(const std::vector<vec3>& data, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
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
    std::cout << "Data saved to " << filename << std::endl;
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

    std::vector<size_t> grid_info_vec;
    // tbl
    // 0 : mCellsSize
    // 1 : mEdgesSize
    // 2 : mMaxEdgesSize
    // 3 : mVertexSize
    // 4 : mVertLevels
    // 5 : mVertLevelsP1
    grid_info_vec.push_back(grid->mCellsSize);
    grid_info_vec.push_back(grid->mEdgesSize);
    grid_info_vec.push_back(grid->mMaxEdgesSize);
    grid_info_vec.push_back(grid->mVertexSize);
    grid_info_vec.push_back(grid->mVertLevels);
    grid_info_vec.push_back(grid->mVertLevelsP1);
    
#if USE_SYCL
    sycl::buffer<vec3, 1> vertexCoord_buf(grid->vertexCoord_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size())); // CELL vertex coordinate
    sycl::buffer<vec3, 1> cellCoord_buf(grid->cellCoord_vec.data(), sycl::range<1>(grid->cellCoord_vec.size()));       // CELL center coordinate

    sycl::buffer<size_t, 1> numberVertexOnCell_buf(grid->numberVertexOnCell_vec.data(), sycl::range<1>(grid->numberVertexOnCell_vec.size())); // Number of vertices per CELL
    sycl::buffer<size_t, 1> verticesOnCell_buf(grid->verticesOnCell_vec.data(), sycl::range<1>(grid->verticesOnCell_vec.size()));             // 
    sycl::buffer<size_t, 1> cellsOnVertex_buf(grid->cellsOnVertex_vec.data(), sycl::range<1>(grid->cellsOnVertex_vec.size()));

    sycl::buffer<double, 1> cellCenterZTop_buf(cellZTop_vec.data(), sycl::range<1>(cellZTop_vec.size()));                           //CELL center ZTOP
    sycl::buffer<double, 1> cellVertexZTop_buf(cellVertexZTop_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size() * mTotalZTopLayer));        //CELL vertex ZTOP (required)
  
    sycl::buffer<size_t, 1> grid_info_buf(grid_info_vec.data(), sycl::range<1>(grid_info_vec.size())); 

    q.submit([&](sycl::handler& cgh) {
        
        auto acc_vertexCoord_buf        = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf          = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnCell_buf     = verticesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnVertex_buf      = cellsOnVertex_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCenterZTop_buf     = cellCenterZTop_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexZTop_buf     = cellVertexZTop_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto acc_grid_info_buf          = grid_info_buf.get_access<sycl::access::mode::read>(cgh);
        
        
        cgh.parallel_for(sycl::range<2>(mCellsSize, mTotalZTopLayer), [=](sycl::id<2> idx) {
            size_t j = idx[0];
            size_t i = idx[1]; 

            auto cell_id = j;
            auto current_layer = i;

            const int CELL_SIZE = acc_grid_info_buf[0]; 
            const int max_edge = acc_grid_info_buf[2];
            const int MAX_VERTEX_NUM = 20;
            const int NEIGHBOR_NUM = 3;
            const int TOTAY_ZTOP_LAYER = acc_grid_info_buf[4];
            const int VERTLEVELS = acc_grid_info_buf[4];
            // 1. Find all vertices of this cell, set non-existent ones to nan
            
            // 1.1 Calculate how many vertices this cell has
            auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
            auto nan = std::numeric_limits<size_t>::max();
            // 1.2 Find all candidate vertices
            size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
            // for (size_t k = 0; k < MAX_VERTEX_NUM; ++k)
            // {
            //     current_cell_vertices_idx[k] = acc_verticesOnCell_buf[cell_id * MAX_VERTEX_NUM + k] - 1; // Assuming 7 is the max number of vertices per cell
            // }
            // // 1.3 Set non-existent vertices to nan
            // for (size_t k = current_cell_vertices_number; k < MAX_VERTEX_NUM; ++k)
            // {
            //     current_cell_vertices_idx[k] = nan;
            // }
            SYCLKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx, MAX_VERTEX_NUM, max_edge, acc_verticesOnCell_buf);
            // =============================== Find max_edges vertices.
            double current_cell_vertices_value[MAX_VERTEX_NUM];
            bool bBoundary = false;
            for (auto k = 0; k < MAX_VERTEX_NUM; ++k)
            {
                auto vertex_idx = current_cell_vertices_idx[k];
                // 2.1 If it is nan, skip
                if (vertex_idx == nan) { current_cell_vertices_value[k] = std::numeric_limits<double>::quiet_NaN(); continue; }
                auto current_vertex = acc_vertexCoord_buf[vertex_idx];
                // 2.2 If it is not nan, find the 3 cell ids containing this vertex (candidates) **Boundary cases do not have 3**
                size_t tmp_cell_id[3];
                tmp_cell_id[0] = acc_cellsOnVertex_buf[3 * vertex_idx + 0] - 1;
                tmp_cell_id[1] = acc_cellsOnVertex_buf[3 * vertex_idx + 1] - 1;
                tmp_cell_id[2] = acc_cellsOnVertex_buf[3 * vertex_idx + 2] - 1;
                // 2.3 Find the center ZTOP of these 3 cells
                double tmp_cell_center_ztop[3];
                int valid_count = 0;
                for (auto tmp_cell = 0; tmp_cell < NEIGHBOR_NUM; tmp_cell++)
                {
                    double value;
                    if (tmp_cell_id[tmp_cell] > CELL_SIZE + 1)
                    {
                        value = 0.0;
                        tmp_cell_center_ztop[tmp_cell] = value;
                        bBoundary = true;
                    }
                    else
                    {
                        double ztop;
                        auto ztop_idx = VERTLEVELS * tmp_cell_id[tmp_cell] + current_layer;
                        ztop = acc_cellCenterZTop_buf[ztop_idx];
                        tmp_cell_center_ztop[tmp_cell] = ztop;
                        valid_count++;
                    }
                }
                // 2.4 If it is a boundary point
                if (bBoundary)
                {
                    current_cell_vertices_value[k] = 0.0 * tmp_cell_center_ztop[0] + 0.0 * tmp_cell_center_ztop[1] + 0.0 * tmp_cell_center_ztop[2];
                }
                else
                {
                    double u, v, w;
                    vec3 p1 = acc_cellCoord_buf[tmp_cell_id[0]];
                    vec3 p2 = acc_cellCoord_buf[tmp_cell_id[1]];
                    vec3 p3 = acc_cellCoord_buf[tmp_cell_id[2]];
                    Interpolator::TRIANGLE tri(p1, p2, p3);
                    Interpolator::calcTriangleBarycentric(current_vertex, &tri, u, v, w);
                    current_cell_vertices_value[k] = u * tmp_cell_center_ztop[0] + v * tmp_cell_center_ztop[1] + w * tmp_cell_center_ztop[2];
                }

                acc_cellVertexZTop_buf[vertex_idx * TOTAY_ZTOP_LAYER + current_layer] = current_cell_vertices_value[k];
            }
        });
    });
    q.wait_and_throw();
    // Debug("finished the sycl part");
    // auto host_accessor = cellVertexZTop_buf.get_access<sycl::access::mode::read>();
    auto host_accessor = cellVertexZTop_buf.get_host_access(sycl::read_only);
    auto range = host_accessor.get_range();
    size_t acc_length = range.size(); // Get the total size of the buffer

    // std::cout << "acc_cellVertexZTop_buf.size() = " << cellVertexZTop_vec.size() << " " << acc_length << std::endl;
    // std::cout << "mVerticesSize x nVertLevels = " << grid->mVertexSize * mVertLevels << std::endl;
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

    std::vector<size_t> grid_info_vec;
    // tbl
    // 0 : mCellsSize
    // 1 : mEdgesSize
    // 2 : mMaxEdgesSize
    // 3 : mVertexSize
    // 4 : mVertLevels
    // 5 : mVertLevelsP1
    grid_info_vec.push_back(grid->mCellsSize);
    grid_info_vec.push_back(grid->mEdgesSize);
    grid_info_vec.push_back(grid->mMaxEdgesSize);
    grid_info_vec.push_back(grid->mVertexSize);
    grid_info_vec.push_back(grid->mVertLevels);
    grid_info_vec.push_back(grid->mVertLevelsP1);
    
#if USE_SYCL
    sycl::buffer<vec3, 1> vertexCoord_buf(grid->vertexCoord_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size())); // CELL vertex coordinates
    sycl::buffer<vec3, 1> cellCoord_buf(grid->cellCoord_vec.data(), sycl::range<1>(grid->cellCoord_vec.size()));       // CELL center coordinates

    sycl::buffer<size_t, 1> numberVertexOnCell_buf(grid->numberVertexOnCell_vec.data(), sycl::range<1>(grid->numberVertexOnCell_vec.size())); // Number of vertices per CELL
    sycl::buffer<size_t, 1> verticesOnCell_buf(grid->verticesOnCell_vec.data(), sycl::range<1>(grid->verticesOnCell_vec.size()));             // 
    sycl::buffer<size_t, 1> cellsOnVertex_buf(grid->cellsOnVertex_vec.data(), sycl::range<1>(grid->cellsOnVertex_vec.size()));

    sycl::buffer<double, 1> cellCenterAttr_buf(vec.data(), sycl::range<1>(vec.size()));                           //CELL center
    sycl::buffer<double, 1> cellVertexAttr_buf(attr_CtoV_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size() * mTotalZTopLayer));        //CELL vertex ZTOP (required)
  
    sycl::buffer<size_t, 1> grid_info_buf(grid_info_vec.data(), sycl::range<1>(grid_info_vec.size())); 

    q.submit([&](sycl::handler& cgh) {
        
        auto acc_vertexCoord_buf        = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf          = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnCell_buf     = verticesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnVertex_buf      = cellsOnVertex_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCenterAttr_buf     = cellCenterAttr_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexAttr_buf     = cellVertexAttr_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto acc_grid_info_buf          = grid_info_buf.get_access<sycl::access::mode::read>(cgh);
        
        
        cgh.parallel_for(sycl::range<2>(mCellsSize, mTotalZTopLayer), [=](sycl::id<2> idx) {
            size_t j = idx[0];
            size_t i = idx[1]; 

            auto cell_id = j;
            auto current_layer = i;

            const int CELL_SIZE = acc_grid_info_buf[0]; 
            const int max_edge = acc_grid_info_buf[2];
            const int MAX_VERTEX_NUM = 20;
            const int NEIGHBOR_NUM = 3;
            const int TOTAY_ZTOP_LAYER = acc_grid_info_buf[4];
            const int VERTLEVELS = acc_grid_info_buf[4];
            // 1. Find all vertices of this cell, set non-existent ones to nan
            
            // 1.1 Calculate how many vertices this CELL has
            auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
            auto nan = std::numeric_limits<size_t>::max();
            // 1.2 Find all candidate vertices
            size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
            SYCLKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx, MAX_VERTEX_NUM, max_edge, acc_verticesOnCell_buf);
            // =============================== Find max_edges vertices
            double current_cell_vertices_value[MAX_VERTEX_NUM];
            bool bBoundary = false;
            for (auto k = 0; k < MAX_VERTEX_NUM; ++k)
            {
                auto vertex_idx = current_cell_vertices_idx[k];
                // 2.1 If it is nan, skip
                if (vertex_idx == nan) { current_cell_vertices_value[k] = std::numeric_limits<double>::quiet_NaN(); continue; }
                auto current_vertex = acc_vertexCoord_buf[vertex_idx];
                // 2.2 If it is not nan, find the 3 cell ids containing this vertex (candidates) **Boundary cases may not have 3**
                size_t tmp_cell_id[3];
                tmp_cell_id[0] = acc_cellsOnVertex_buf[3 * vertex_idx + 0] - 1;
                tmp_cell_id[1] = acc_cellsOnVertex_buf[3 * vertex_idx + 1] - 1;
                tmp_cell_id[2] = acc_cellsOnVertex_buf[3 * vertex_idx + 2] - 1;
                // 2.3 Find the center attribute value of these 3 CELLS
                double tmp_cell_center_attr[3];
                for (auto tmp_cell = 0; tmp_cell < NEIGHBOR_NUM; tmp_cell++)
                {
                    double value;
                    if (tmp_cell_id[tmp_cell] > CELL_SIZE + 1)
                    {
                        value = 0.0;
                        tmp_cell_center_attr[tmp_cell] = value;
                        bBoundary = true;
                    }
                    else
                    {
                        double attr;
                        auto attr_idx = VERTLEVELS * tmp_cell_id[tmp_cell] + current_layer;
                        attr = acc_cellCenterAttr_buf[attr_idx];
                        tmp_cell_center_attr[tmp_cell] = attr;
                    }
                }
                // 2.4 If it is a boundary point
                if (bBoundary)
                {
                    current_cell_vertices_value[k] = 0.0 * tmp_cell_center_attr[0] + 0.0 * tmp_cell_center_attr[1] + 0.0 * tmp_cell_center_attr[2];
                }
                else
                {
                    double u, v, w;
                    vec3 p1 = acc_cellCoord_buf[tmp_cell_id[0]];
                    vec3 p2 = acc_cellCoord_buf[tmp_cell_id[1]];
                    vec3 p3 = acc_cellCoord_buf[tmp_cell_id[2]];
                    Interpolator::TRIANGLE tri(p1, p2, p3);
                    Interpolator::calcTriangleBarycentric(current_vertex, &tri, u, v, w);
                    current_cell_vertices_value[k] = u * tmp_cell_center_attr[0] + v * tmp_cell_center_attr[1] + w * tmp_cell_center_attr[2];
                    if (current_cell_vertices_value[k] < 0) current_cell_vertices_value[k] = 0.0;
                }

                acc_cellVertexAttr_buf[vertex_idx * TOTAY_ZTOP_LAYER + current_layer] = current_cell_vertices_value[k];
            }
        });
    });
    q.wait_and_throw();
    // Debug("finished the sycl part");
    // auto host_accessor = cellVertexZTop_buf.get_access<sycl::access::mode::read>();
    auto host_accessor = cellVertexAttr_buf.get_host_access(sycl::read_only);
    auto range = host_accessor.get_range();
    size_t acc_length = range.size(); // Get the total size of the buffer

    // std::cout << "acc_cellVertexZTop_buf.size() = " << cellVertexZTop_vec.size() << " " << acc_length << std::endl;
    // std::cout << "mVerticesSize x nVertLevels = " << grid->mVertexSize * mVertLevels << std::endl;
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

    std::vector<size_t> grid_info_vec;
    // tbl
    // 0 : mCellsSize
    // 1 : mEdgesSize
    // 2 : mMaxEdgesSize
    // 3 : mVertexSize
    // 4 : mVertLevels
    // 5 : mVertLevelsP1
    grid_info_vec.push_back(grid->mCellsSize);
    grid_info_vec.push_back(grid->mEdgesSize);
    grid_info_vec.push_back(grid->mMaxEdgesSize);
    grid_info_vec.push_back(grid->mVertexSize);
    grid_info_vec.push_back(grid->mVertLevels);
    grid_info_vec.push_back(grid->mVertLevelsP1);

#if USE_SYCL
    sycl::buffer<vec3, 1> vertexCoord_buf(grid->vertexCoord_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size())); // CELL Vertex coordinates
    sycl::buffer<vec3, 1> cellCoord_buf(grid->cellCoord_vec.data(), sycl::range<1>(grid->cellCoord_vec.size()));       // CELL center coordinates
    sycl::buffer<vec3, 1> edgeCoord_buf(grid->edgeCoord_vec.data(), sycl::range<1>(grid->edgeCoord_vec.size()));       // EDGE center coordinates

    sycl::buffer<size_t, 1> numberVertexOnCell_buf(grid->numberVertexOnCell_vec.data(), sycl::range<1>(grid->numberVertexOnCell_vec.size()));   // Number of vertices on CELL
    sycl::buffer<size_t, 1> edgesOnCell_buf(grid->edgesOnCell_vec.data(), sycl::range<1>(grid->edgesOnCell_vec.size()));                        // CELL edge IDs
    sycl::buffer<size_t, 1> cellsOnEdge_buf(grid->cellsOnEdge_vec.data(), sycl::range<1>(grid->cellsOnEdge_vec.size()));                        // EDGE cell IDs
    sycl::buffer<size_t, 1> verticesOnEdge_buf(grid->verticesOnEdge_vec.data(), sycl::range<1>(grid->verticesOnEdge_vec.size()));               // EDGE vertex IDs

    sycl::buffer<double, 1> cellNormalVelocity_buf(cellNormalVelocity_vec.data(), sycl::range<1>(cellNormalVelocity_vec.size()));               // EDGE Normal Velocity
    sycl::buffer<vec3, 1> cellCenterVelocity_buf(cellCenterVelocity_vec.data(), sycl::range<1>(cellCenterVelocity_vec.size()));                 // CELL Center Velocity

    sycl::buffer<size_t, 1> grid_info_buf(grid_info_vec.data(), sycl::range<1>(grid_info_vec.size())); 

    q.submit([&](sycl::handler& cgh) {
        auto acc_vertexCoord_buf = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_edgeCoord_buf = edgeCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_edgesOnCell_buf = edgesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnEdge_buf = cellsOnEdge_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnEdge_buf = verticesOnEdge_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellNormalVelocity_buf = cellNormalVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCenterVelocity_buf = cellCenterVelocity_buf.get_access<sycl::access::mode::write>(cgh);
        auto acc_grid_info_buf          = grid_info_buf.get_access<sycl::access::mode::read>(cgh);
        sycl::stream out(8192, 256, cgh);
        cgh.parallel_for(sycl::range<2>(mCellsSize, mTotalZTopLayer), [=](sycl::id<2> idx) {
            
            size_t j = idx[0];
            size_t i = idx[1];

            auto cell_id = j;
            auto current_layer = i;

            const int CELL_SIZE = acc_grid_info_buf[0];
            const int max_edge = acc_grid_info_buf[2];
            const int MAX_VERTEX_NUM = 7;
            const int NEIGHBOR_NUM = 3;
            const int TOTAY_ZTOP_LAYER = acc_grid_info_buf[4];
            const int VERTLEVELS = acc_grid_info_buf[4];
            // 1. Find the cell position based on cell_id
            vec3 cell_center_velocity = { 0.0, 0.0, 0.0 };
            vec3 cell_position = acc_cellCoord_buf[cell_id];
            double total_length = 0.0;
            double edge_length = 0.0;
            // 2. Find all edge_ids based on cell_id 
            size_t current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
            size_t current_cell_edges_id[MAX_VERTEX_NUM];
            for (auto k = 0; k < current_cell_vertices_number; ++k)
            {
                current_cell_edges_id[k] = acc_edgesOnCell_buf[cell_id * MAX_VERTEX_NUM + k] -1;
            }
            // 2.1 Set non-existent edge_id to nan
            auto nan = std::numeric_limits<size_t>::max();
            for (auto k = current_cell_vertices_number; k < MAX_VERTEX_NUM; ++k)
            {
                current_cell_edges_id[k] = nan;
            }
            

            // Calculate planeBasisVector
            vec3 eZonal; vec3 eMeridional;

            vec3 up = cell_position / YOSEF_LENGTH(cell_position); // Unit up vector

            // Select the global z-axis as a reference
            vec3 k = vec3(0.0, 0.0, 1.0);
            vec3 east = YOSEF_CROSS(k, up);
            if (YOSEF_LENGTH(east) < 1e-6) {
                // If cell_center is near the pole, use a different reference vector, e.g., (0,1,0)
                east = YOSEF_CROSS(vec3(0.0, 1.0, 0.0), up);
            }
            east = east / YOSEF_LENGTH(east); // Unit east vector

            vec3 north = YOSEF_CROSS(up, east); // normalized (if up and east are normalized)

            //GeoConverter::convertXYZPositionToENUUnitVectory(cell_position, eZonal, eMeridional);
            double planeBasisVector[2][3] = { {east.x(), east.y(), east.z()}, {north.x(), north.y(), north.z()} };
            // Calculate alpha
            double cellCenter[3] = { cell_position.x(), cell_position.y(), cell_position.z() };
            int pointCount = MAX_VERTEX_NUM; 
            double edge_center[MAX_VERTEX_NUM][3];
            double unit_vector[MAX_VERTEX_NUM][3];
            double noraml_vel[MAX_VERTEX_NUM][1];
            double coeffs[MAX_VERTEX_NUM][3] = {{0.0}};
            // 3. Iterate - Calculate the vector velocity for each edge
            for (auto k = 0; k < MAX_VERTEX_NUM; ++k)
            {
                auto edge_id = current_cell_edges_id[k];
                if (edge_id == nan) { continue; } 
                // 3.1 Find the center position of the edge
                vec3 edge_position = acc_edgeCoord_buf[edge_id];
                edge_center[k][0] = edge_position.x(); edge_center[k][1] = edge_position.y(); edge_center[k][2] = edge_position.z();
                // 3.2 Find the normal vector of the edge
                size_t tmp_cell_id[2];
                tmp_cell_id[0] = acc_cellsOnEdge_buf[edge_id * 2.0f + 0] - 1;
                tmp_cell_id[1] = acc_cellsOnEdge_buf[edge_id * 2.0f + 1] - 1;
                auto min_cell_id = tmp_cell_id[0] < tmp_cell_id[1] ? tmp_cell_id[0] : tmp_cell_id[1];
                auto max_cell_id = tmp_cell_id[0] > tmp_cell_id[1] ? tmp_cell_id[0] : tmp_cell_id[1];
                vec3 normal_vector;
                double length;
                if (max_cell_id > CELL_SIZE)
                {
                    //vec3 edge_position_min = acc_edgeCoord_buf[min_cell_id];
                    vec3 min_cell_position = acc_cellCoord_buf[min_cell_id];
                    normal_vector = edge_position - min_cell_position; 
                    length = YOSEF_LENGTH(normal_vector);
                    if (length == 0.0) { continue; }
                    normal_vector /= length;
                }
                else
                {
                    vec3 min_cell_position = acc_cellCoord_buf[min_cell_id];
                    vec3 max_cell_position = acc_cellCoord_buf[max_cell_id];
                    // 3.2 Method 1: Difference between the centers of the two cells
                    normal_vector = max_cell_position - min_cell_position;
                    length = YOSEF_LENGTH(normal_vector);
                    if (length == 0.0) { continue; }
                    normal_vector /= length;
                }
                // 3.3 Find the normal velocity of the edge
                auto normal_velocity = acc_cellNormalVelocity_buf[edge_id * TOTAY_ZTOP_LAYER + current_layer];
                noraml_vel[k][0] = normal_velocity;
                // 3.4 Find the normal of the edge
                unit_vector[k][0] = normal_vector.x() ; 
                unit_vector[k][1] = normal_vector.y() ; 
                unit_vector[k][2] = normal_vector.z() ;
                
               
                // // Calculate the weights of the edges 
                // size_t vertices_idx_on_edge[2];
                // vertices_idx_on_edge[0] = acc_verticesOnEdge_buf[edge_id * 2.0f + 0] - 1;
                // vertices_idx_on_edge[1] = acc_verticesOnEdge_buf[edge_id * 2.0f + 1] - 1;
                // size_t VID1 = vertices_idx_on_edge[0]; 
                // size_t VID2 = vertices_idx_on_edge[1];
                // vec3 vertex1 = acc_vertexCoord_buf[VID1];
                // vec3 vertex2 = acc_vertexCoord_buf[VID2];
                // vec3 edge_vector = vertex2 - vertex1;
                // edge_length  = YOSEF_LENGTH(edge_vector);
                // total_length += edge_length;
   

                // vec3 current_edge_velocity;
                // current_edge_velocity.x() = normal_velocity * normal_vector.x() * edge_length;
                // current_edge_velocity.y() = normal_velocity * normal_vector.y() * edge_length;
                // current_edge_velocity.z() = normal_velocity * normal_vector.z() * edge_length;
            }

            // Calculate alpha
            double alpha = Interpolator::compute_alpha(edge_center, pointCount, cellCenter);
            alpha = 1.0;
            // out << "alpha = " << alpha << sycl::endl;
            Interpolator::mpas_rbf_interp_func_3D_plane_vec_const_dir_comp_coeffs(pointCount, edge_center, unit_vector, cellCenter, alpha, planeBasisVector, coeffs);
            double xVel = 0.0;
            double yVel = 0.0;
            double zVel = 0.0;

            for (auto k = 0; k < MAX_VERTEX_NUM; ++k)
            {

                xVel += coeffs[k][0] * noraml_vel[k][0];
                yVel += coeffs[k][1] * noraml_vel[k][0];
                zVel += coeffs[k][2] * noraml_vel[k][0];
                out << "coeffs[" << k << "][0] = " << coeffs[k][0] << " coeffs[" << k << "][1] = " << coeffs[k][1] << " coeffs[" << k << "][2] = " << coeffs[k][2] << sycl::endl;

            }
            
            // out << "xVel = " << xVel << " yVel = " << yVel << " zVel = " << zVel << sycl::endl;
            // 4. average
            // cell_center_velocity /= static_cast<double>(current_cell_vertices_number);
            // cell_center_velocity *= 2.0;
            // cell_center_velocity /= total_length;

            cell_center_velocity = vec3(xVel, yVel, zVel);
            acc_cellCenterVelocity_buf[cell_id * TOTAY_ZTOP_LAYER + current_layer] = cell_center_velocity;

        });
    });
    q.wait_and_throw();

    // auto host_accessor = cellCenterVelocity_buf.get_access<sycl::access::mode::read>();
    auto host_accessor = cellCenterVelocity_buf.get_host_access(sycl::read_only);
    auto range = host_accessor.get_range();
    size_t acc_length = range.size(); // Get the total size of the buffer

    // std::cout << "cellCenterVelocity_buf.size() = " << cellCenterVelocity_vec.size() << " " << acc_length << std::endl;
    // std::cout << "mCellSize x nVertLevels = " << grid->mCellsSize * mVertLevels << std::endl;
    //saveDataToTextFile2(cellCenterVelocity_vec, "OUTPUT1_ztop.txt");
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

    std::vector<size_t> grid_info_vec;
    // tbl
    // 0 : mCellsSize
    // 1 : mEdgesSize
    // 2 : mMaxEdgesSize
    // 3 : mVertexSize
    // 4 : mVertLevels
    // 5 : mVertLevelsP1
    grid_info_vec.push_back(grid->mCellsSize);
    grid_info_vec.push_back(grid->mEdgesSize);
    grid_info_vec.push_back(grid->mMaxEdgesSize);
    grid_info_vec.push_back(grid->mVertexSize);
    grid_info_vec.push_back(grid->mVertLevels);
    grid_info_vec.push_back(grid->mVertLevelsP1);

#if USE_SYCL
    sycl::buffer<vec3, 1> vertexCoord_buf(grid->vertexCoord_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size())); // CELL vertex coordinate
    sycl::buffer<vec3, 1> cellCoord_buf(grid->cellCoord_vec.data(), sycl::range<1>(grid->cellCoord_vec.size()));       // CELL center coordinate
    sycl::buffer<vec3, 1> edgeCoord_buf(grid->edgeCoord_vec.data(), sycl::range<1>(grid->edgeCoord_vec.size()));       // EDGE center coordinate

    sycl::buffer<size_t, 1> numberVertexOnCell_buf(grid->numberVertexOnCell_vec.data(), sycl::range<1>(grid->numberVertexOnCell_vec.size())); // Number of vertices on CELL
    sycl::buffer<size_t, 1> edgesOnCell_buf(grid->edgesOnCell_vec.data(), sycl::range<1>(grid->edgesOnCell_vec.size()));                       // CELL edge IDs
    sycl::buffer<size_t, 1> cellsOnEdge_buf(grid->cellsOnEdge_vec.data(), sycl::range<1>(grid->cellsOnEdge_vec.size()));                       // EDGE cell IDs
    sycl::buffer<size_t, 1> verticesOnEdge_buf(grid->verticesOnEdge_vec.data(), sycl::range<1>(grid->verticesOnEdge_vec.size()));               // EDGE vertex IDs

    sycl::buffer<double, 1> cellZonalVelocity_buf(cellZonalVelocity_vec.data(), sycl::range<1>(cellZonalVelocity_vec.size()));            
    sycl::buffer<double, 1> cellMeridionalVelocity_buf(cellMeridionalVelocity_vec.data(), sycl::range<1>(cellMeridionalVelocity_vec.size()));
    sycl::buffer<vec3, 1> cellCenterVelocity_buf(cellCenterVelocity_vec.data(), sycl::range<1>(cellCenterVelocity_vec.size()));        // CELL center velocity
    sycl::buffer<size_t, 1> grid_info_buf(grid_info_vec.data(), sycl::range<1>(grid_info_vec.size())); 

    q.submit([&](sycl::handler& cgh) {
        auto acc_vertexCoord_buf = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_edgeCoord_buf = edgeCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_edgesOnCell_buf = edgesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnEdge_buf = cellsOnEdge_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnEdge_buf = verticesOnEdge_buf.get_access<sycl::access::mode::read>(cgh);
        
        auto acc_cellZonalVelocity_buf = cellZonalVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellMeridionalVelocity_buf = cellMeridionalVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCenterVelocity_buf = cellCenterVelocity_buf.get_access<sycl::access::mode::write>(cgh);
        auto acc_grid_info_buf          = grid_info_buf.get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for(sycl::range<2>(mCellsSize, mTotalZTopLayer), [=](sycl::id<2> idx) {
            
            size_t j = idx[0];
            size_t i = idx[1];

            auto cell_id = j;
            auto current_layer = i;

            const int CELL_SIZE = acc_grid_info_buf[0];
            const int max_edge = acc_grid_info_buf[2];
            const int TOTAY_ZTOP_LAYER = acc_grid_info_buf[4];
            const int VERTLEVELS = acc_grid_info_buf[4];
            // 1. Find cell position based on cell_id
            vec3 cell_center_velocity = { 0.0, 0.0, 0.0 };
            vec3 cell_position = acc_cellCoord_buf[cell_id];
            // 2. Based on (cell_id , current_layer) -> zonal velocity and meridional velocity
            double tmp_zonal = acc_cellZonalVelocity_buf[cell_id * TOTAY_ZTOP_LAYER + current_layer];
            double tmp_mer = acc_cellMeridionalVelocity_buf[cell_id * TOTAY_ZTOP_LAYER + current_layer];   
            GeoConverter::convertENUVelocityToXYZ(cell_position, tmp_zonal, tmp_mer, 0.0, cell_center_velocity);
            acc_cellCenterVelocity_buf[cell_id * TOTAY_ZTOP_LAYER + current_layer] = cell_center_velocity;
        });
    });
    q.wait_and_throw();

    // auto host_accessor = cellCenterVelocity_buf.get_access<sycl::access::mode::read>();
    auto host_accessor = cellCenterVelocity_buf.get_host_access(sycl::read_only);
    auto range = host_accessor.get_range();
    size_t acc_length = range.size(); 

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

    std::vector<size_t> grid_info_vec;
    // tbl
    // 0 : mCellsSize
    // 1 : mEdgesSize
    // 2 : mMaxEdgesSize
    // 3 : mVertexSize
    // 4 : mVertLevels
    // 5 : mVertLevelsP1
    grid_info_vec.push_back(grid->mCellsSize);
    grid_info_vec.push_back(grid->mEdgesSize);
    grid_info_vec.push_back(grid->mMaxEdgesSize);
    grid_info_vec.push_back(grid->mVertexSize);
    grid_info_vec.push_back(grid->mVertLevels);
    grid_info_vec.push_back(grid->mVertLevelsP1);

#if USE_SYCL
    sycl::buffer<vec3, 1> vertexCoord_buf(grid->vertexCoord_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size())); // CELL vertex coordinate
    sycl::buffer<vec3, 1> cellCoord_buf(grid->cellCoord_vec.data(), sycl::range<1>(grid->cellCoord_vec.size()));       // CELL center coordinate

    sycl::buffer<size_t, 1> numberVertexOnCell_buf(grid->numberVertexOnCell_vec.data(), sycl::range<1>(grid->numberVertexOnCell_vec.size())); // Number of vertices on CELL

    sycl::buffer<double, 1> cellZonalVelocity_buf(cellVertexZonalVelocity_vec.data(), sycl::range<1>(cellVertexZonalVelocity_vec.size()));            
    sycl::buffer<double, 1> cellMeridionalVelocity_buf(cellVertexMeridionalVelocity_vec.data(), sycl::range<1>(cellVertexMeridionalVelocity_vec.size()));
    sycl::buffer<vec3, 1> cellVertexVelocity_buf(cellVertexVelocity_vec.data(), sycl::range<1>(cellVertexVelocity_vec.size()));        // CELL vertex velocity

    sycl::buffer<size_t, 1> grid_info_buf(grid_info_vec.data(), sycl::range<1>(grid_info_vec.size())); 

    q.submit([&](sycl::handler& cgh) {
        auto acc_vertexCoord_buf = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellZonalVelocity_buf = cellZonalVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellMeridionalVelocity_buf = cellMeridionalVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexVelocity_buf = cellVertexVelocity_buf.get_access<sycl::access::mode::write>(cgh);
        auto acc_grid_info_buf          = grid_info_buf.get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for(sycl::range<2>(mVertexSize, mTotalZTopLayer), [=](sycl::id<2> idx) {
            
            size_t j = idx[0];
            size_t i = idx[1];

            auto vertex_id = j;
            auto current_layer = i;

            const int VERTEX_SIZE = acc_grid_info_buf[3];
            const int max_edge = acc_grid_info_buf[2];
            const int TOTAY_ZTOP_LAYER = acc_grid_info_buf[4];
            const int VERTLEVELS = acc_grid_info_buf[4];
            // 1. Base on vertex_id to find vertex position
            vec3 vertex_center_velocity = { 0.0, 0.0, 0.0 };
            vec3 vertex_position = acc_vertexCoord_buf[vertex_id];
            // 2. Based on (vertex_id , current_layer) -> zonal velocity and meridional velocity
            double tmp_zonal = acc_cellZonalVelocity_buf[vertex_id * TOTAY_ZTOP_LAYER + current_layer];
            double tmp_mer = acc_cellMeridionalVelocity_buf[vertex_id * TOTAY_ZTOP_LAYER + current_layer];
            GeoConverter::convertENUVelocityToXYZ(vertex_position, tmp_zonal, tmp_mer, 0.0, vertex_center_velocity);
            acc_cellVertexVelocity_buf[vertex_id * TOTAY_ZTOP_LAYER + current_layer] = vertex_center_velocity;
        });
    });
    q.wait_and_throw();

    // auto host_accessor = cellCenterVelocity_buf.get_access<sycl::access::mode::read>();
    auto host_accessor = cellVertexVelocity_buf.get_host_access(sycl::read_only);
    auto range = host_accessor.get_range();
    size_t acc_length = range.size(); 

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

    std::vector<size_t> grid_info_vec;
    // tbl
    // 0 : mCellsSize
    // 1 : mEdgesSize
    // 2 : mMaxEdgesSize
    // 3 : mVertexSize
    // 4 : mVertLevels
    // 5 : mVertLevelsP1
    grid_info_vec.push_back(grid->mCellsSize);
    grid_info_vec.push_back(grid->mEdgesSize);
    grid_info_vec.push_back(grid->mMaxEdgesSize);
    grid_info_vec.push_back(grid->mVertexSize);
    grid_info_vec.push_back(grid->mVertLevels);
    grid_info_vec.push_back(grid->mVertLevelsP1);

#if USE_SYCL
    sycl::buffer<vec3, 1> vertexCoord_buf(grid->vertexCoord_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size())); // CELL vertex coordinate
    sycl::buffer<vec3, 1> cellCoord_buf(grid->cellCoord_vec.data(), sycl::range<1>(grid->cellCoord_vec.size()));       // CELL center coordinate

    sycl::buffer<size_t, 1> numberVertexOnCell_buf(grid->numberVertexOnCell_vec.data(), sycl::range<1>(grid->numberVertexOnCell_vec.size())); // Number of vertices on CELL
    sycl::buffer<size_t, 1> verticesOnCell_buf(grid->verticesOnCell_vec.data(), sycl::range<1>(grid->verticesOnCell_vec.size()));             // 
    sycl::buffer<size_t, 1> cellsOnVertex_buf(grid->cellsOnVertex_vec.data(), sycl::range<1>(grid->cellsOnVertex_vec.size()));

    sycl::buffer<vec3, 1> cellCenterVelocity_buf(cellCenterVelocity_vec.data(), sycl::range<1>(cellCenterVelocity_vec.size()));                           //CELL center ZTOP
    sycl::buffer<vec3, 1> cellVertexVelocity_buf(cellVertexVelocity_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size() * mTotalZTopLayer));        //CELL vertex ZTOP (wanan to calculate)

    sycl::buffer<size_t, 1> grid_info_buf(grid_info_vec.data(), sycl::range<1>(grid_info_vec.size())); 

    q.submit([&](sycl::handler& cgh) {

        auto acc_vertexCoord_buf = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnCell_buf = verticesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnVertex_buf = cellsOnVertex_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCenterVelocity_buf = cellCenterVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexVelocity_buf = cellVertexVelocity_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto acc_grid_info_buf          = grid_info_buf.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::range<2>(mCellsSize, mTotalZTopLayer), [=](sycl::id<2> idx) {
            size_t j = idx[0];
            size_t i = idx[1];

            auto cell_id = j;
            auto current_layer = i;

            const int CELL_SIZE = acc_grid_info_buf[0];
            const int max_edge = acc_grid_info_buf[2];
            const int MAX_VERTEX_NUM = 20;
            const int NEIGHBOR_NUM = 3;
            const int TOTAY_ZTOP_LAYER = acc_grid_info_buf[4];
            const int VERTLEVELS = acc_grid_info_buf[4];
            // 1. Find all vertices of this cell, set to nan if not exist

            // 1.1 Calculate how many vertices this CELL has
            auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
            auto nan = std::numeric_limits<size_t>::max();
            // 1.2 Find all candidate vertices
            size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
            SYCLKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx, MAX_VERTEX_NUM, max_edge, acc_verticesOnCell_buf);
            // =============================== Find max_edge vertices ===============================
            vec3 current_cell_vertices_value[MAX_VERTEX_NUM];
            bool bBoundary = false;
            for (auto k = 0; k < MAX_VERTEX_NUM; ++k)
            {
                auto vertex_idx = current_cell_vertices_idx[k];
                // 2.1 If it is nan, skip
                if (vertex_idx == nan)
                {
                    auto double_nan = std::numeric_limits<double>::quiet_NaN();
                    current_cell_vertices_value[k] = { double_nan , double_nan , double_nan };
                    continue; 
                }
                auto current_vertex = acc_vertexCoord_buf[vertex_idx];
                // 2.2 If it is not nan, find the 3 cell ids (candidates) containing this vertex; boundary cases may have fewer than 3
                size_t tmp_cell_id[3];
                tmp_cell_id[0] = acc_cellsOnVertex_buf[3 * vertex_idx + 0] - 1;
                tmp_cell_id[1] = acc_cellsOnVertex_buf[3 * vertex_idx + 1] - 1;
                tmp_cell_id[2] = acc_cellsOnVertex_buf[3 * vertex_idx + 2] - 1;
                // 2.3 Find the center velocities of these 3 cells
                vec3 tmp_cell_center_vels[3];
                for (auto tmp_cell = 0; tmp_cell < NEIGHBOR_NUM; tmp_cell++)
                {
                    vec3 value;
                    if (tmp_cell_id[tmp_cell] > CELL_SIZE + 1)
                    {
                        value = { 0.0, 0.0, 0.0 };
                        tmp_cell_center_vels[tmp_cell] = value;
                        bBoundary = true;
                    }
                    else
                    {
                        vec3 vel;
                        auto vel_idx = VERTLEVELS * tmp_cell_id[tmp_cell] + current_layer;
                        vel = acc_cellCenterVelocity_buf[vel_idx];
                        tmp_cell_center_vels[tmp_cell] = vel;
                    }
                }
               
                // 2.4 If it's a boundary point
                if (bBoundary)
                {
                    current_cell_vertices_value[k] = 0.0 * tmp_cell_center_vels[0] + 0.0 * tmp_cell_center_vels[1] + 0.0 * tmp_cell_center_vels[2];
                }
                else
                {
                    double u, v, w;
                    vec3 p1 = acc_cellCoord_buf[tmp_cell_id[0]];
                    vec3 p2 = acc_cellCoord_buf[tmp_cell_id[1]];
                    vec3 p3 = acc_cellCoord_buf[tmp_cell_id[2]];
                    Interpolator::TRIANGLE tri(p1, p2, p3);
                    Interpolator::calcTriangleBarycentric(current_vertex, &tri, u, v, w);
                    current_cell_vertices_value[k] = u * tmp_cell_center_vels[0] + v * tmp_cell_center_vels[1] + w * tmp_cell_center_vels[2];
                }

                acc_cellVertexVelocity_buf[vertex_idx * TOTAY_ZTOP_LAYER + current_layer] = current_cell_vertices_value[k];
            }
            });
        });
    q.wait_and_throw();
    auto host_accessor = cellVertexVelocity_buf.get_host_access(sycl::read_only);
    auto range = host_accessor.get_range();
    size_t acc_length = range.size(); 
    writeVertexZTopToFile<vec3>(cellVertexVelocity_vec, cell_vertex_velocity_path);
    Debug("[MPASOSolution]::Calc Cell cellVertexVelocity_vec  = \t [ %d ] \t type = [ float64 float64 float64]", 
            cellVertexVelocity_vec.size());
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
        std::cout << "====== " << value << " found [\u2713]" << std::endl;
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
        std::cout << "====== " << value << " found [\u2713]" << std::endl;
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
            std::cerr << "[Error]: The value found is not of type ftk::ndarray<char>" << std::endl;
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
        std::cout << "====== " << value << " found [\u2713]" << std::endl;
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