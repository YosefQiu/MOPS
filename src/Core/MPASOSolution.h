#pragma once
#include "IO/MPASOReader.h"
#include "Core/MPASOGrid.h"
#include "ndarray/ndarray_group_stream.hh"
namespace MOPS
{
    enum class AttributeFormat : int { kDouble, kFloat, kChar, kVec3, kCount };
    enum class AttributeType : int { kZonalVelocity, kMeridionalVelocity, kVelocity, kNormalVelocity, kZTop, kLayerThickness, kBottomDepth, kCount };
    class MPASOSolution
    {
    public:
        MPASOSolution() = default;
    public:
        void initSolution(MPASOReader* reader);
        void initSolution(ftk::ndarray_group* g, MPASOReader* reader = nullptr);
        void initSolution_DemoLoading(const char* yaml_path, int timestep);
        void initSolution_FromBin(const char* prefix);
    public:
        std::string mCurrentTime;
        std::string mDataName;

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
        std::vector<double>     cellSurfaceHeight_vec;

        std::vector<double>     cellVertexZTop_vec;
        std::vector<double>     cellVertexMeridionalVelocity_vec;
        std::vector<double>	 	cellVertexZonalVelocity_vec;
        std::vector<vec3>       cellCenterVelocity_vec;
        std::vector<vec3>       cellVertexVelocity_vec;

        // cell center attributes
        std::map<std::string, std::vector<char>>    mCharAttributes;
        std::map<std::string, std::vector<float>>   mFloatAttributes;
        std::map<std::string, std::vector<double>>  mDoubleAttributes;
        std::map<std::string, std::vector<vec3>>    mVec3Attributes;

        // cell vertex attributes
        std::map<std::string, std::vector<double>>  mDoubleAttributes_CtoV; // center to vertex
        std::map<std::string, std::vector<float>>   mFloatAttributes_CtoV;


    public:
        std::string getCurrentTime() const { return mCurrentTime; }
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
        void setAttribute(GridAttributeType type, int val);
        void setAttributesVec3(AttributeType type, const std::vector<vec3>& vec);
        void setAttributesDouble(AttributeType type, const std::vector<double>& vec);
        void setTimestep(int timestep) { mTimesteps = timestep; }
        bool checkAttribute();

        void addAttribute(std::string name, AttributeFormat type);
        void calcCellCenterZtop();
        void calcCellVertexZtop(MPASOGrid* grid, std::string& dataDir, sycl::queue& q);
        void calcCellCenterToVertex(const std::string& name, const std::vector<double>& vec, MPASOGrid* grid, std::string& dataDir, sycl::queue& q);
        void calcCellCenterVelocity(MPASOGrid* grid, std::string& dataDir, sycl::queue& q);
        void calcCellCenterVelocityByZM(MPASOGrid* grid, std::string& dataDir, sycl::queue& q);
        void calcCellVertexVelocity(MPASOGrid* grid, std::string& dataDir, sycl::queue& q);
        void calcCellVertexVelocityByZM(MPASOGrid* grid, std::string& dataDir, sycl::queue& q);
    private:
        void readFromBlock_Vec3(const std::string& filename, std::vector<vec3>& vec);    
        void readFromBlock_Double(const std::string& filename, std::vector<double>& vec);
        void readFromBlock_DoubleBasedK(const std::string& filename, std::vector<double>& vec, int K = -1);

        std::shared_ptr<ftk::ndarray_group> gt;
        void copyFromNdarray_Double(ftk::ndarray_group* g, std::string value, std::vector<double>& vec);
        void copyFromNdarray_Char(ftk::ndarray_group* g, std::string value, std::vector<char>& vec);
        void copyFromNdarray_Float(ftk::ndarray_group* g, std::string value, std::vector<float>& vec);
        // void copyFromNdarray_Vec3(ftk::ndarray_group* g, std::string value, std::vector<vec3>& vec);
        // void copyFromNdarray_Int(ftk::ndarray_group* g, std::string value, std::vector<int>& vec);
    };
}  

