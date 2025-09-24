#pragma once
#include "ggl.h"
#include "Utils/Log.hpp"
#include "ndarray/ndarray_group_stream.hh"

namespace MOPS
{
	class MPASOReader
	{
	public:
		using Ptr = std::shared_ptr<MPASOReader>;
		MPASOReader() = default;
		MPASOReader(const std::string& path);
		~MPASOReader();
	public:
		int mCellsSize = 0;
        int mEdgesSize = 0;
        int mMaxEdgesSize = 0;
        int mVertexSize = 0;
        int mTimesteps = 0;
		int mTimestepLength = 0;
        int mVertLevels = 0;
        int mVertLevelsP1 = 0;

		std::string mCurrentTime;
	public:
		std::string mMeshName;
		std::string mDataName;
		std::string mFolderName;
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
		

		std::vector<vec3>		cellVelocity_vec;
        std::vector<double>     cellSurfaceHeight_vec;
        std::vector<double>	    cellLayerThickness_vec;
        std::vector<double>     cellZTop_vec;
        std::vector<double>     cellVertVelocity_vec;
        std::vector<double>     cellNormalVelocity_vec;
        std::vector<double>     cellMeridionalVelocity_vec;
        std::vector<double>	 	cellZonalVelocity_vec;
        std::vector<double>     cellBottomDepth_vec;

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

        std::shared_ptr<ftk::ndarray_group> mGroupT = nullptr;

	public:
		static Ptr readGridInfo(const std::string& path)
		{
			std::shared_ptr<MPASOReader> reader(new MPASOReader(path));
			reader->readData(path.c_str()); 
			return reader;
		}
		static Ptr readSolInfo(const std::string& path, int timestep)
		{
			std::shared_ptr<MPASOReader> reader(new MPASOReader(path));
			reader->readSol(path.c_str());
			return reader;
		}
        static Ptr readMPASO(const std::string& yaml_path, int timestep);
		static Ptr readGridData(const std::string& yaml_path);
		static Ptr readSolData(const std::string& yaml_path, const std::string& data_name = "", const int& timestep = 0);
        void appendSolData(MPASOReader* reader, const std::string& yaml_path, const int& timestep);
		void readData(const char* nc_filename); 
		void readSol(const char* nc_filename);
		
	private:
        std::shared_ptr<ftk::stream> mStream = nullptr;
	private:
        static void readFromBlock_Vec3(const std::string& filename, std::vector<vec3>& vec);    
        static void readFromBlock_Int(const std::string& filename, std::vector<size_t>& vec);
        static void readFromBlock_IntBasedK(const std::string& filename, std::vector<size_t>& vec, int K = -1);
		

		static void copyFromNdarray_Double(ftk::ndarray_group* g, std::string value, std::vector<double>& vec);
		static void copyFromNdarray_Char(ftk::ndarray_group* g, std::string value, std::vector<char>& vec);
        static void copyFromNdarray_Float(ftk::ndarray_group* g, std::string value, std::vector<float>& vec);
        static void copyFromNdarray_Int(ftk::ndarray_group* g, std::string value, std::vector<size_t>& vec);
        static void copyFromNdarray_Vec2(ftk::ndarray_group* g, std::string xValue, std::string yValue, std::vector<vec2>& vec, std::string name = "");
        static void copyFromNdarray_Vec3(ftk::ndarray_group* g, std::string xValue, std::string yValue, std::string zValue, std::vector<vec3>& vec, std::string name = "");
	};


}
