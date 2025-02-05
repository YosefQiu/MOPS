#pragma once
#include "MPASOReader.h"
#include "KDTree.h"
#include "Interpolation.hpp"
#include "ndarray/ndarray_group_stream.hh"

//TODO ? why in here
enum class CalcType : int{ kVelocity, kZTOP, kCount };

typedef struct VisualizationConfig
{
    CalcType mType;
    size_t mLevel;
    VisualizationConfig(CalcType config, size_t level = 0) : mType(config), mLevel(level) {}
}VConfig;

class MPASOGrid
{
public:
    MPASOGrid();
public:
    int mCellsSize;
    int mEdgesSize;
    int mMaxEdgesSize;
    int mVertexSize;

    int mTimesteps;
    int mVertLevels;
    int mVertLevelsP1;

public:
    std::vector<vec3>		vertexCoord_vec;
    std::vector<vec3>		cellCoord_vec;
    std::vector<vec3>       edgeCoord_vec;

    std::vector<vec2>		vertexLatLon_vec;
    std::vector<size_t>		verticesOnCell_vec;
    std::vector<size_t>		verticesOnEdge_vec;
    std::vector<size_t>		cellsOnVertex_vec;
    std::vector<size_t>		cellsOnCell_vec;
    std::vector<size_t>		numberVertexOnCell_vec;
    std::vector<size_t>     cellsOnEdge_vec;
	std::vector<size_t>     edgesOnCell_vec;
    std::vector<float>      cellWeight_vec;

public:
    [[deprecated]] void initGrid(MPASOReader* reader);
    void initGrid(ftk::ndarray_group* g, MPASOReader* reader = nullptr);
	void createKDTree(const char* kdTree_path, sycl::queue& SYCL_Q);
    void searchKDT(const CartesianCoord& point, int& cell_id);

    void getNeighborCells(const size_t cell_id, std::vector<size_t>& cell_on_cell, std::vector<size_t>& neighbor_id);
    void getVerticesOnCell(const size_t cell_id, std::vector<size_t>& vertex_on_cell, std::vector<size_t>& vertex_id);
    void getCellsOnVertex(const size_t vertex_id, std::vector<size_t>& cell_on_vertex, std::vector<size_t>& cell_id);
    void getCellsOnEdge(const size_t edge_id, std::vector<size_t>& cell_on_edge, std::vector<size_t>& cell_id);
    void getEdgesOnCell(const size_t cell_id, std::vector<size_t>& edge_on_cell, std::vector<size_t>& edge_id);
    
private:
    void copyFromNdarray_Int(ftk::ndarray_group* g, std::string value, std::vector<size_t>& vec);
    void copyFromNdarray_Vec2(ftk::ndarray_group* g, std::string xValue, std::string yValue, std::vector<vec2>& vec, std::string name = nullptr);
    void copyFromNdarray_Vec3(ftk::ndarray_group* g, std::string xValue, std::string yValue, std::string zValue, std::vector<vec3>& vec, std::string name = nullptr);
public:
#if _WIN32 || __linux__
    std::unique_ptr<KDTree_t> mKDTree;
#elif __APPLE__
    std::unique_ptr<kdtreegpu> mKDTree;
#endif
};


