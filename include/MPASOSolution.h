#pragma once
#include "MPASOReader.h"
#include "MPASOGrid.h"
#include "ndarray/ndarray_group_stream.hh"

class MPASOSolution
{
public:
    MPASOSolution() = default;
public:
    void initSolution(MPASOReader* reader);[[deprecated]]
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

    void getCellVelocity(const size_t cell_id, const size_t level, std::vector<vec3>& cell_on_velocity, vec3& vel);
    void getCellVertVelocity(const size_t cell_id, const size_t level, std::vector<double>& cell_vert_velocity, double& vel);
    void getCellZTop(const size_t cell_id, const size_t level, std::vector<double>& cell_ztop, double& ztop);
    void getEdgeNormalVelocity(const size_t edge_id, const size_t level, std::vector<double>& edge_normal_velocity, double& vel);
    void getCellLayerThickness(const size_t cell_id, const size_t level, std::vector<double>& cell_thickness, double& thinckness);
    void getCellSurfaceMeridionalVelocity(const size_t cell_id, std::vector<double>& cell_meridional_velocity, double& vel);
    void getCellSurfaceZonalVelocity(const size_t cell_id, std::vector<double>& cell_zonal_velocity, double& vel);

    void calcCellCenterZtop();
    void calcCellVertexZtop(MPASOGrid* grid, sycl::queue& q);
    void calcCellCenterVelocity(MPASOGrid* grid, sycl::queue& q);
    void calcCellCenterVelocityByZM(MPASOGrid* grid, sycl::queue& q);
    void calcCellVertexVelocity(MPASOGrid* grid, sycl::queue& q);

    void getCellCenterZTop(const size_t cell_id, const size_t level, std::vector<double>& cell_ztop, double& ztop);
    void getCellVertexZTop(const size_t vertex_id, const size_t level, std::vector<double>& cell_vertex_ztop, double& ztop);
    void getCellCenterVelocity(const size_t cell_id, const size_t level, std::vector<vec3>& cell_on_velocity, vec3& vel);
    void getCellVertexVelocity(const size_t vertex_id, const size_t level, std::vector<vec3>& cell_vertex_velocity, vec3& vel);
private:
    void copyFromNdarray_Double(ftk::ndarray_group* g, std::string value, std::vector<double>& vec);
    
    //void copyFromNdarray_Double
};

