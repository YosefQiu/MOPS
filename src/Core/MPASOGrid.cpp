#include "Core/MPASOGrid.h"
#include "Utils/Utils.hpp"

using namespace MOPS;

MPASOGrid::MPASOGrid()
{
    mKDTree = nullptr;
}

void MPASOGrid::initGrid_DemoLoading(const char* yaml_path)
{
    std::shared_ptr<ftk::stream> stream(new ftk::stream);
	stream->parse_yaml(yaml_path);
	auto grid_path = stream->substreams[0]->filenames[0];
	auto gs = stream->read_static();
    std::string fileName = removeFileExtension(stream->substreams[0]->filenames[0]);
	std::string dataDir = createDataPath(".data", fileName);
	this->initGrid(gs.get(), MOPS::MPASOReader::readGridInfo(grid_path).get());
    this->mDataDir = dataDir;
    this->mMeshName = fileName;

}

void MPASOGrid::setGridAttribute(GridAttributeType type, int val)
{
    // kCellSize, kEdgeSize, kVertexSize, kMaxEdgesSize, kVertLevels, kVertLevelsP1,
    switch (type)
    {
        case GridAttributeType::kCellSize:
            this->mCellsSize = val;
            break;
        case GridAttributeType::kEdgeSize:
            this->mEdgesSize = val;
            break;
        case GridAttributeType::kVertexSize:
            this->mVertexSize = val;
            break;
        case GridAttributeType::kMaxEdgesSize:
            this->mMaxEdgesSize = val;
            break;
        case GridAttributeType::kVertLevels:
            this->mVertLevels = val;
            break;
        case GridAttributeType::kVertLevelsP1:
            this->mVertLevelsP1 = val;
            break;
        default:
            std::cout << "Error: Invalid GridAttributeType" << std::endl;
            break;
    }
}

void MPASOGrid::setGridAttributesVec3(GridAttributeType type, const std::vector<vec3>& vec)
{
   
    // kVertexCoord, kCellCoord, kEdgeCoord,
    switch (type)
    {
        case GridAttributeType::kVertexCoord:
            this->vertexCoord_vec = vec;
            break;
        case GridAttributeType::kCellCoord:
            this->cellCoord_vec = vec;
            break;
        case GridAttributeType::kEdgeCoord:
            this->edgeCoord_vec = vec;
            break;
        default:
            std::cout << "Error: Invalid GridAttributeType" << std::endl;
    }
}

void MPASOGrid::setGridAttributesVec2(GridAttributeType type, const std::vector<vec2>& vec)
{
    // kVertexLatLon
    switch (type)
    {
        case GridAttributeType::kVertexLatLon:
            this->vertexLatLon_vec = vec;
            break;
        default:
            std::cout << "Error: Invalid GridAttributeType" << std::endl;
    }
}

void MPASOGrid::setGridAttributesInt(GridAttributeType type, const std::vector<size_t>& vec)
{
    // kVerticesOnCell, kVerticesOnEdge, kCellsOnVertex, kCellsOnCell, kNumberVertexOnCell, kCellsOnEdge, kEdgesOnCell
    switch (type)
    {
        case GridAttributeType::kVerticesOnCell:
            this->verticesOnCell_vec = vec;
            break;
        case GridAttributeType::kVerticesOnEdge:
            this->verticesOnEdge_vec = vec;
            break;
        case GridAttributeType::kCellsOnVertex:
            this->cellsOnVertex_vec = vec;
            break;
        case GridAttributeType::kCellsOnCell:
            this->cellsOnCell_vec = vec;
            break;
        case GridAttributeType::kNumberVertexOnCell:
            this->numberVertexOnCell_vec = vec;
            break;
        case GridAttributeType::kCellsOnEdge:
            this->cellsOnEdge_vec = vec;
            break;
        case GridAttributeType::kEdgesOnCell:
            this->edgesOnCell_vec = vec;
            break;
        default:
            std::cout << "Error: Invalid GridAttributeType" << std::endl;
    }
}

void MPASOGrid::setGridAttributesFloat(GridAttributeType type, const std::vector<float>& vec)
{
    // kCellWeight
    switch (type)
    {
        case GridAttributeType::kCellWeight:
            this->cellWeight_vec = vec;
            break;
        default:
            std::cout << "Error: Invalid GridAttributeType" << std::endl;
    }
}


[[deprecated]]
void MPASOGrid::initGrid(MPASOReader* reader)
{
    this->mCellsSize    = reader->mCellsSize;
    this->mEdgesSize    = reader->mEdgesSize;
    this->mMaxEdgesSize = reader->mMaxEdgesSize;
    this->mVertexSize   = reader->mVertexSize;

    this->mTimesteps    = reader->mTimesteps;
    this->mVertLevels   = reader->mVertLevels;
    this->mVertLevelsP1 = reader->mVertLevelsP1;

    this->vertexCoord_vec           = std::move(reader->vertexCoord_vec);
    this->cellCoord_vec             = std::move(reader->cellCoord_vec);
    this->edgeCoord_vec             = std::move(reader->edgeCoord_vec);
    this->vertexLatLon_vec          = std::move(reader->vertexLatLon_vec);
    this->verticesOnCell_vec        = std::move(reader->verticesOnCell_vec);
    this->cellsOnVertex_vec         = std::move(reader->cellsOnVertex_vec);
    this->cellsOnCell_vec           = std::move(reader->cellsOnCell_vec);
    this->numberVertexOnCell_vec    = std::move(reader->numberVertexOnCell_vec);
    this->cellsOnEdge_vec           = std::move(reader->cellsOnEdge_vec);
    this->edgesOnCell_vec           = std::move(reader->edgesOnCell_vec);

 
}

void MPASOGrid::initGrid(ftk::ndarray_group* g, MPASOReader* reader)
{
    this->mCellsSize    = reader->mCellsSize;
    this->mEdgesSize    = reader->mEdgesSize;
    this->mMaxEdgesSize = reader->mMaxEdgesSize;
    this->mVertexSize   = reader->mVertexSize;
    this->mTimesteps    = reader->mTimesteps;
    this->mVertLevels   = reader->mVertLevels;
    this->mVertLevelsP1 = reader->mVertLevelsP1;

    
    copyFromNdarray_Vec3(g, "xCell", "yCell", "zCell", this->cellCoord_vec, "cellCoord_vec");
    copyFromNdarray_Vec3(g, "xVertex", "yVertex", "zVertex", this->vertexCoord_vec, "vertexCoord_vec");
    copyFromNdarray_Vec3(g, "xEdge", "yEdge", "zEdge", this->edgeCoord_vec, "edgeCoord_vec");
    copyFromNdarray_Vec2(g, "latVertex", "lonVertex", this->vertexLatLon_vec, "vertexLatLon_vec");
    
    copyFromNdarray_Int(g, "verticesOnCell", this->verticesOnCell_vec);
    copyFromNdarray_Int(g, "verticesOnEdge", this->verticesOnEdge_vec);
    copyFromNdarray_Int(g, "cellsOnVertex", this->cellsOnVertex_vec);
    copyFromNdarray_Int(g, "cellsOnCell", this->cellsOnCell_vec);
    copyFromNdarray_Int(g, "nEdgesOnCell", this->numberVertexOnCell_vec);
    copyFromNdarray_Int(g, "cellsOnEdge", this->cellsOnEdge_vec);
    copyFromNdarray_Int(g, "edgesOnCell", this->edgesOnCell_vec);

   


}

void MPASOGrid::createKDTree(const char* kdTree_path, sycl::queue& SYCL_Q)
{
#if _WIN32 || __linux__
    std::ifstream f_in(kdTree_path, std::ifstream::binary);
    if (!f_in)
    {
        // File does not exist or cannot be read, create a new KD Tree
        if (mKDTree) throw std::runtime_error("KDTree already exists!");
        mKDTree = std::make_unique<KDTree_t>(3, cellCoord_vec, 10, 5, true);
        std::ofstream f_out(kdTree_path, std::ofstream::binary);
        if (!f_out) throw std::runtime_error("Error writing index file!");
        mKDTree->index->saveIndex(f_out);
        Debug("[MPASOGrid]::Create KD Tree...");
        Debug("[MPASOGrid]::Saved KD Tree in [ %s ]", kdTree_path);
    }
    else
    {
        // File exists and can be read, load the KD Tree
        mKDTree = std::make_unique<KDTree_t>(3, cellCoord_vec, 10, 5, false);
        mKDTree->index->loadIndex(f_in);
        Debug("[MPASOGrid]::Loading KD Tree in [ %s ]", kdTree_path);
        
    }
#elif __APPLE__
    int n = cellCoord_vec.size();
    int neighbor_num = 1;
    int query_num = 1;
    int dim = 3;
    mKDTree = std::make_unique<kdtreegpu>(n, neighbor_num, query_num, dim, SYCL_Q, cellCoord_vec);
    mKDTree->build();
    Debug("[MPASOGrid]::Create KD Tree...");
#endif

}

void MPASOGrid::searchKDT(const CartesianCoord& point, int& cell_id)
{
#if _WIN32 || __linux__
    const int dim = 3;

    // Query point:
    std::vector<double> query_pt(dim);
    for (size_t d = 0; d < dim; d++)
        query_pt[d] = point[d];

    // do a knn search
    const size_t        num_results = 1;
    std::vector<size_t> ret_indexes(num_results);
    std::vector<double> out_dists_sqr(num_results);

    nanoflann::KNNResultSet<double> resultSet(num_results);

    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
    mKDTree->index->findNeighbors(resultSet, &query_pt[0]);

    cell_id = ret_indexes[0];
#elif __APPLE__
    float query_points[3] = { point.x(), point.y(), point.z() };
    cell_id = mKDTree->query_gpu(query_points);
#endif
}

void MPASOGrid::getNeighborCells(const size_t cell_id, std::vector<size_t>& cell_on_cell, std::vector<size_t>& neighbor_id)
{

    std::vector<size_t>::const_iterator first = cell_on_cell.begin() + (mMaxEdgesSize * cell_id + 0);
    std::vector<size_t>::const_iterator last = cell_on_cell.begin() + (mMaxEdgesSize * cell_id + mMaxEdgesSize);
    if (!neighbor_id.empty()) neighbor_id.clear();
    neighbor_id = std::vector<size_t>(first, last);
    for (auto& val : neighbor_id) val -= 1;
}

void MPASOGrid::getVerticesOnCell(const size_t cell_id, std::vector<size_t>& vertex_on_cell, std::vector<size_t>& vertex_id)
{
    std::vector<size_t>::const_iterator first = vertex_on_cell.begin() + (mMaxEdgesSize * cell_id + 0);
    std::vector<size_t>::const_iterator last = vertex_on_cell.begin() + (mMaxEdgesSize * cell_id + mMaxEdgesSize);
    if (!vertex_id.empty()) vertex_id.clear();
    vertex_id = std::vector<size_t>(first, last);
    for (auto& val : vertex_id) val -= 1;
}

void MPASOGrid::getCellsOnVertex(const size_t vertex_id, std::vector<size_t>& cell_on_vertex, std::vector<size_t>& cell_id)
{
    std::vector<size_t>::const_iterator first = cell_on_vertex.begin() + (3 * vertex_id + 0);
    std::vector<size_t>::const_iterator last = cell_on_vertex.begin() + (3 * vertex_id + 3);
    if (!cell_id.empty()) cell_id.clear();
    cell_id = std::vector<size_t>(first, last);
    for (auto& val : cell_id) val -= 1;
}

void MPASOGrid::getCellsOnEdge(const size_t edge_id, std::vector<size_t>& cell_on_edge, std::vector<size_t>& cell_id)
{
    // 输入一个边的ID，返回这个边上的两个cell的ID
    std::vector<size_t>::const_iterator first = cell_on_edge.begin() + (2 * edge_id + 0);
    std::vector<size_t>::const_iterator last = cell_on_edge.begin() + (2 * edge_id + 2);
    if (!cell_id.empty()) cell_id.clear();
    cell_id = std::vector<size_t>(first, last);
    for (auto& val : cell_id) val -= 1;
}

void MPASOGrid::getEdgesOnCell(const size_t cell_id, std::vector<size_t>& edge_on_cell, std::vector<size_t>& edge_id)
{
    // 输入一个cell的ID，返回这个cell上的所有边的ID
    std::vector<size_t>::const_iterator first = edge_on_cell.begin() + (mMaxEdgesSize * cell_id + 0);
    std::vector<size_t>::const_iterator last = edge_on_cell.begin() + (mMaxEdgesSize * cell_id + mMaxEdgesSize);
    if (!edge_id.empty()) edge_id.clear();
    edge_id = std::vector<size_t>(first, last);
    for (auto& val : edge_id) val -= 1;
} 

void MPASOGrid::copyFromNdarray_Int(ftk::ndarray_group* g, std::string value, std::vector<size_t>& vec)
{
    if (g->has(value))
    {
        auto tmp_ptr = std::dynamic_pointer_cast<ftk::ndarray<int32_t>>(g->get(value));
        auto tmp_vec = tmp_ptr->std_vector();
        vec.resize(tmp_vec.size());
        for (auto i = 0; i < tmp_vec.size(); i++)
            vec[i] = static_cast<size_t>(tmp_vec[i]);
        Debug("[Ndarray]::loading %-30s = \t [ %-8d ] \t type = [ %-10s ]", 
            value.c_str(),
            vec.size(), 
            ftk::ndarray_base::dtype2str(g->get(value).get()->type()).c_str());
    }
}

void MPASOGrid::copyFromNdarray_Vec2(ftk::ndarray_group* g, std::string xValue, std::string yValue, std::vector<vec2>& vec, std::string name)
{
    if (g->has(xValue) && g->has(yValue))
    {
        auto Lat_ptr = std::dynamic_pointer_cast<ftk::ndarray<double>>(g->get(xValue));
        auto Lon_ptr = std::dynamic_pointer_cast<ftk::ndarray<double>>(g->get(yValue));
        auto Lat_vec = Lat_ptr->std_vector();
        auto Lon_vec = Lon_ptr->std_vector();
        vec.resize(Lat_vec.size());
        for (auto i = 0; i < Lat_vec.size(); i++)
        {
            vec[i] = vec2(Lat_vec[i], Lon_vec[i]);
        }
        Debug("[Ndarray]::loading %-30s = \t [ %-8d ] \t type = [ %-10s %-10s ]", 
            name.c_str(),
            vec.size(), 
            ftk::ndarray_base::dtype2str(g->get(xValue).get()->type()).c_str(),
            ftk::ndarray_base::dtype2str(g->get(yValue).get()->type()).c_str());
    }
}

void MPASOGrid::copyFromNdarray_Vec3(ftk::ndarray_group* g, std::string xValue, std::string yValue, std::string zValue, std::vector<vec3>& vec, std::string name)
{
    if(g->has(xValue) && g->has(yValue) && g->has(zValue))
    {
        auto xEdge_ptr = std::dynamic_pointer_cast<ftk::ndarray<double>>(g->get(xValue));
        auto yEdge_ptr = std::dynamic_pointer_cast<ftk::ndarray<double>>(g->get(yValue));
        auto zEdge_ptr = std::dynamic_pointer_cast<ftk::ndarray<double>>(g->get(zValue));
        auto xEdge_vec = xEdge_ptr->std_vector();
        auto yEdge_vec = yEdge_ptr->std_vector();
        auto zEdge_vec = zEdge_ptr->std_vector();
        vec.resize(xEdge_vec.size());
        for (auto i = 0; i < xEdge_vec.size(); i++)
        {
           vec[i] = vec3(xEdge_vec[i], yEdge_vec[i], zEdge_vec[i]);
        }
        // std::cout << "Inside function, cellCoord_vec size: " << vec.size() << " at address: " << &vec << std::endl;
        // Debug("[Ndarray]::loading %s = \t [ %d ] \t type = [ %s %s %s]", 
        //     name.c_str(),
        //     vec.size(), 
        //     ftk::ndarray_base::dtype2str(g->get(xValue).get()->type()).c_str(),
        //     ftk::ndarray_base::dtype2str(g->get(yValue).get()->type()).c_str(),
        //     ftk::ndarray_base::dtype2str(g->get(zValue).get()->type()).c_str());

        Debug("[Ndarray]::loading %-30s = \t [ %-8d ] \t type = [ %-10s %-10s %-10s ]", 
            name.c_str(),
            vec.size(), 
            ftk::ndarray_base::dtype2str(g->get(xValue).get()->type()).c_str(),
            ftk::ndarray_base::dtype2str(g->get(yValue).get()->type()).c_str(),
            ftk::ndarray_base::dtype2str(g->get(zValue).get()->type()).c_str());

    }
    else
    {
        std::cout << "Error: Missing data in ndarray_group for " << name << std::endl;
    }
}

bool MPASOGrid::checkAttribute()
{
    if (this->mCellsSize == 0)
    {
        std::cout << "Error: mCellsSize is not set" << std::endl;
        return false;
    }
    if (this->mEdgesSize == 0)
    {
        std::cout << "Error: mEdgesSize is not set" << std::endl;
        return false;
    }
    if (this->mVertexSize == 0)
    {
        std::cout << "Error: mVertexSize is not set" << std::endl;
        return false;
    }
    if (this->mMaxEdgesSize == 0)
    {
        std::cout << "Error: mMaxEdgesSize is not set" << std::endl;
        return false;
    }
    if (this->mVertLevels == 0)
    {
        std::cout << "Error: mVertLevels is not set" << std::endl;
        return false;
    }
    if (this->mVertLevelsP1 == 0)
    {
        std::cout << "Error: mVertLevelsP1 is not set" << std::endl;
        return false;
    }
    if (this->cellCoord_vec.size() < 1)
    {
        std::cout << "Error: cellCoord_vec is not set" << std::endl;
        return false;
    }
    if (this->vertexCoord_vec.size() < 1)
    {
        std::cout << "Error: vertexCoord_vec is not set" << std::endl;
        return false;
    }
    if (this->edgeCoord_vec.size() < 1)
    {
        std::cout << "Error: edgeCoord_vec is not set" << std::endl;
        return false;
    }
    if (this->verticesOnCell_vec.size() < 1)
    {
        std::cout << "Error: verticesOnCell_vec is not set" << std::endl;
        return false;
    }
    if (this->verticesOnEdge_vec.size() < 1)
    {
        std::cout << "Error: verticesOnEdge_vec is not set" << std::endl;
        return false;
    }
    if (this->cellsOnVertex_vec.size() < 1)
    {
        std::cout << "Error: cellsOnVertex_vec is not set" << std::endl;
        return false;
    }
    if (this->cellsOnCell_vec.size() < 1)
    {
        std::cout << "Error: cellsOnCell_vec is not set" << std::endl;
        return false;
    }
    if (this->numberVertexOnCell_vec.size() < 1)
    {
        std::cout << "Error: numberVertexOnCell_vec is not set" << std::endl;
        return false;
    }
    if (this->cellsOnEdge_vec.size() < 1)
    {
        std::cout << "Error: cellsOnEdge_vec is not set" << std::endl;
        return false;
    }
    if (this->edgesOnCell_vec.size() < 1)
    {
        std::cout << "Error: edgesOnCell_vec is not set" << std::endl;
        return false;
    }
    return true;
}