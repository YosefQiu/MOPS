#pragma once
#include "IO/MPASOReader.h"
#include "Utils/KDTree.h"
#include "Utils/Interpolation.hpp"
#include "ndarray/ndarray_group_stream.hh"
#include <vector>

namespace MOPS
{
    //TODO ? why in here
    enum class CalcType : int{ kVelocity, kZTOP, kCount };
    enum class GridAttributeType : int{ kCellSize, kEdgeSize, kVertexSize, kMaxEdgesSize, kVertLevels, kVertLevelsP1, 
            kVertexCoord, kCellCoord, kEdgeCoord, kVertexLatLon, kVerticesOnCell, kVerticesOnEdge, kCellsOnVertex, kCellsOnCell, 
            kNumberVertexOnCell, kCellsOnEdge, kEdgesOnCell, kCellWeight, krefBottomDepth, kCount };
    inline std::string GridAttributeTypeToString(GridAttributeType type)
    {
        switch (type)
        {
            case GridAttributeType::kCellSize: return "kCellSize";
            case GridAttributeType::kEdgeSize: return "kEdgeSize";
            case GridAttributeType::kVertexSize: return "kVertexSize";
            case GridAttributeType::kMaxEdgesSize: return "kMaxEdgesSize";
            case GridAttributeType::kVertLevels: return "kVertLevels";
            case GridAttributeType::kVertLevelsP1: return "kVertLevelsP1";
            case GridAttributeType::kVertexCoord: return "kVertexCoord";
            case GridAttributeType::kCellCoord: return "kCellCoord";
            case GridAttributeType::kEdgeCoord: return "kEdgeCoord";
            case GridAttributeType::kVertexLatLon: return "kVertexLatLon";
            case GridAttributeType::kVerticesOnCell: return "kVerticesOnCell";
            case GridAttributeType::kVerticesOnEdge: return "kVerticesOnEdge";
            case GridAttributeType::kCellsOnVertex: return "kCellsOnVertex";
            case GridAttributeType::kCellsOnCell: return "kCellsOnCell";
            case GridAttributeType::kNumberVertexOnCell: return "kNumberVertexOnCell";
            case GridAttributeType::kCellsOnEdge: return "kCellsOnEdge";
            case GridAttributeType::kEdgesOnCell: return "kEdgesOnCell";
            case GridAttributeType::kCellWeight: return "kCellWeight";
            case GridAttributeType::krefBottomDepth: return "krefBottomDepth";
            default: return "kCount";
        }
    }
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
        int mCellsSize = 0;
        int mEdgesSize = 0;
        int mMaxEdgesSize = 0;
        int mVertexSize = 0;
        int mTimesteps = 0;
        int mVertLevels = 0;
        int mVertLevelsP1 = 0;
    public:
        std::string mMeshName;
        std::string mCachedDataDir;
        std::string mFolderPath;
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

        std::vector<double>     cellRefBottomDepth_vec;

    public:
        void initGrid(MPASOReader* reader);
        void initGrid_DemoLoading(const char* yaml_path);
        void initGrid_FromBin(const char* prefix);
        void initGrid(ftk::ndarray_group* g, MPASOReader* reader = nullptr);
        void createKDTree(const char* kdTree_path, sycl::queue& SYCL_Q);
        void searchKDT(const CartesianCoord& point, int& cell_id);

        void setGridAttribute(GridAttributeType type, int val);
        void setGridAttributesVec3(GridAttributeType type, const std::vector<vec3>& vec);
        void setGridAttributesVec2(GridAttributeType type, const std::vector<vec2>& vec);
        void setGridAttributesInt(GridAttributeType type, const std::vector<size_t>& vec);
        void setGridAttributesFloat(GridAttributeType type, const std::vector<float>& vec);

        void getNeighborCells(const size_t cell_id, std::vector<size_t>& cell_on_cell, std::vector<size_t>& neighbor_id);
        void getVerticesOnCell(const size_t cell_id, std::vector<size_t>& vertex_on_cell, std::vector<size_t>& vertex_id);
        void getCellsOnVertex(const size_t vertex_id, std::vector<size_t>& cell_on_vertex, std::vector<size_t>& cell_id);
        void getCellsOnEdge(const size_t edge_id, std::vector<size_t>& cell_on_edge, std::vector<size_t>& cell_id);
        void getEdgesOnCell(const size_t cell_id, std::vector<size_t>& edge_on_cell, std::vector<size_t>& edge_id);
        std::string getFolderPath() const { return mFolderPath; }   
        bool checkAttribute();
    private:
        void readFromBlock_Vec3(const std::string& filename, std::vector<vec3>& vec);    
        void readFromBlock_Int(const std::string& filename, std::vector<size_t>& vec);
        void readFromBlock_IntBasedK(const std::string& filename, std::vector<size_t>& vec, int K = -1);

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



}
