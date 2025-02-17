#pragma once
#include "MPASOReader.h"
#include "MPASOGrid.h"
#include "ndarray/ndarray_group_stream.hh"

class MPASOSolution
{
public:
    MPASOSolution() = default;
public:
    [[deprecated]] void initSolution(MPASOReader* reader);
    void initSolution(ftk::ndarray_group* g, MPASOReader* reader = nullptr);
public:
    int mCurrentTime;

    int mCellsSize;
    int mEdgesSize;
    int mMaxEdgesSize;
    int mVertexSize;
    int mTimesteps;
    int mVertLevels;
    int mVertLevelsP1;

    int mTotalZTopLayer;

    // related velocity --> need to split a another class.
    std::vector<vec3>		cellVelocity_vec;
    std::vector<double>	    cellLayerThickness_vec;
    std::vector<double>     cellZTop_vec;
    std::vector<double>     cellVertVelocity_vec;
    std::vector<double>     cellNormalVelocity_vec;
    std::vector<double>     cellMeridionalVelocity_vec;
    std::vector<double>	 	cellZonalVelocity_vec;
    std::vector<double>     cellBottomDepth_vec;

    std::vector<double>     cellVertexZTop_vec;
    std::vector<vec3>      cellCenterVelocity_vec;
    std::vector<vec3>      cellVertexVelocity_vec;

public:
    // ** Deprecated **
    // All get functions are the default data on the CPU side. 
    // When the data is on the GPU side, 
    // it should NOT be used.
    [[deprecated]] void getCellVertVelocity(const size_t cell_id, const size_t level, std::vector<double>& cell_vert_velocity, double& vel);
    [[deprecated]] void getCellVelocity(const size_t cell_id, const size_t level, std::vector<vec3>& cell_on_velocity, vec3& vel);
    [[deprecated]] void getCellZTop(const size_t cell_id, const size_t level, std::vector<double>& cell_ztop, double& ztop);
    [[deprecated]] void getEdgeNormalVelocity(const size_t edge_id, const size_t level, std::vector<double>& edge_normal_velocity, double& vel);
    void getCellLayerThickness(const size_t cell_id, const size_t level, std::vector<double>& cell_thickness, double& thinckness);
    [[deprecated]] void getCellSurfaceMeridionalVelocity(const size_t cell_id, std::vector<double>& cell_meridional_velocity, double& vel);
    [[deprecated]] void getCellSurfaceZonalVelocity(const size_t cell_id, std::vector<double>& cell_zonal_velocity, double& vel);
    [[deprecated]] void getCellCenterZTop(const size_t cell_id, const size_t level, std::vector<double>& cell_ztop, double& ztop);
    [[deprecated]] void getCellVertexZTop(const size_t vertex_id, const size_t level, std::vector<double>& cell_vertex_ztop, double& ztop);
    [[deprecated]] void getCellCenterVelocity(const size_t cell_id, const size_t level, std::vector<vec3>& cell_on_velocity, vec3& vel);
    [[deprecated]] void getCellVertexVelocity(const size_t vertex_id, const size_t level, std::vector<vec3>& cell_vertex_velocity, vec3& vel);
public:
    void calcCellCenterZtop();
    void calcCellVertexZtop(MPASOGrid* grid, std::string& dataDir, sycl::queue& q);
    void calcCellCenterVelocity(MPASOGrid* grid, std::string& dataDir, sycl::queue& q);
    void calcCellCenterVelocityByZM(MPASOGrid* grid, std::string& dataDir, sycl::queue& q);
    void calcCellVertexVelocity(MPASOGrid* grid, std::string& dataDir, sycl::queue& q);
private:
    void copyFromNdarray_Double(ftk::ndarray_group* g, std::string value, std::vector<double>& vec);
    void copyFromNdarray_Char(ftk::ndarray_group* g, std::string value, std::vector<char>& vec);
};

