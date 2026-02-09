#include "IO/MPASOReader.h"
#include "MPASOReader.h"
#include "Utils/Utils.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <unistd.h>
#include "ndarray/ndarray_group_stream.hh"
#include "netcdf.h"

using namespace MOPS;
#define NC_CHECK(err) \
    if ((err) != NC_NOERR) { \
        std::cerr << "NetCDF error: " << nc_strerror(err) << std::endl; \
        std::exit(1); \
    }

MPASOReader::MPASOReader(const std::string& path) 
{
    mCellsSize = -1;
    mMaxEdgesSize = -1;
    mVertexSize = -1;
    mTimesteps = -1;
    mVertLevels = -1;
    mVertLevelsP1 = -1;
 
}

MPASOReader::~MPASOReader()
{
    ftk::ndarray_finalize();
}


void MPASOReader::readData(const char* nc_filename)
{

    Debug("[MPASOReader]::readData: %s", nc_filename);
    int ncid, err, dimid;
    size_t tmp;

    err = nc_open(nc_filename, NC_NOWRITE, &ncid);
    NC_CHECK(err);

    err = nc_inq_dimid(ncid, "nCells", &dimid);
    NC_CHECK(err);
    err = nc_inq_dimlen(ncid, dimid, &tmp);
    NC_CHECK(err);
    mCellsSize = static_cast<int>(tmp);

    err = nc_inq_dimid(ncid, "nEdges", &dimid);
    NC_CHECK(err);
    err = nc_inq_dimlen(ncid, dimid, &tmp);
    NC_CHECK(err);
    mEdgesSize = static_cast<int>(tmp);

    err = nc_inq_dimid(ncid, "maxEdges", &dimid);
    NC_CHECK(err);
    err = nc_inq_dimlen(ncid, dimid, &tmp);
    NC_CHECK(err);
    mMaxEdgesSize = static_cast<int>(tmp);

    err = nc_inq_dimid(ncid, "nVertices", &dimid);
    NC_CHECK(err);
    err = nc_inq_dimlen(ncid, dimid, &tmp);
    NC_CHECK(err);
    mVertexSize = static_cast<int>(tmp);

    err = nc_inq_dimid(ncid, "nVertLevels", &dimid);
    NC_CHECK(err);
    err = nc_inq_dimlen(ncid, dimid, &tmp);
    NC_CHECK(err);
    mVertLevels = static_cast<int>(tmp);
    
    err = nc_inq_dimid(ncid, "nVertLevelsP1", &dimid);
    NC_CHECK(err);
    err = nc_inq_dimlen(ncid, dimid, &tmp);
    NC_CHECK(err);
    mVertLevelsP1 = static_cast<int>(tmp);

    err = nc_close(ncid);
    NC_CHECK(err);

    Debug("%-40s = \t [ %d ]", "[MPASOReader]::loading mCellsSize",     mCellsSize);
    Debug("%-40s = \t [ %d ]", "[MPASOReader]::loading mEdgesSize",     mEdgesSize);
    Debug("%-40s = \t [ %d ]", "[MPASOReader]::loading mMaxEdgesSize",  mMaxEdgesSize);
    Debug("%-40s = \t [ %d ]", "[MPASOReader]::loading mVertexSize",    mVertexSize);
    Debug("%-40s = \t [ %d ]", "[MPASOReader]::loading mVertLevels",    mVertLevels);
    Debug("%-40s = \t [ %d ]", "[MPASOReader]::loading mVertLevelsP1",  mVertLevelsP1);
}

void MPASOReader::readSol(const char* nc_filename)
{

    Debug("[MPASOReader]::readSol: %s", nc_filename);
    int ncid, err, dimid;
    size_t tmp;

    err = nc_open(nc_filename, NC_NOWRITE, &ncid);
    NC_CHECK(err);

    err = nc_inq_dimid(ncid, "Time", &dimid);
    NC_CHECK(err);
    err = nc_inq_dimlen(ncid, dimid, &tmp);
    NC_CHECK(err);
    mTimestepLength = static_cast<int>(tmp);

    err = nc_inq_dimid(ncid, "nVertLevels", &dimid);
    NC_CHECK(err);
    err = nc_inq_dimlen(ncid, dimid, &tmp);
    NC_CHECK(err);
    mVertLevels = static_cast<int>(tmp);

    Debug("%-40s = \t [ %d ]", "[MPASOReader]::loading mTimestepLength", mTimestepLength);
    Debug("%-40s = \t [ %d ]", "[MPASOReader]::loading mVertLevels", mVertLevels);
}

MPASOReader::Ptr MPASOReader::readMPASO(const std::string& yaml_path, int timestep)
{
    auto reader = readGridData(yaml_path);
    reader->appendSolData(reader.get(), yaml_path, timestep);
    return reader;
}

MPASOReader::Ptr MPASOReader::readGridData(const std::string& yaml_path)
{
    std::shared_ptr<MPASOReader> reader(new MPASOReader(yaml_path));
    reader->mStream = std::make_shared<ftk::stream>();
	reader->mStream->parse_yaml(yaml_path);
	auto gs = reader->mStream->read_static();
    auto gridName = removeFileExtension(reader->mStream->substreams[0]->filenames[0]);
    reader->mMeshName = gridName;
    reader->mFolderName = reader->mStream->path_prefix;

    copyFromNdarray_Vec3(gs.get(), "xCell", "yCell", "zCell", reader->cellCoord_vec, "cellCoord_vec");
    copyFromNdarray_Vec3(gs.get(), "xVertex", "yVertex", "zVertex", reader->vertexCoord_vec, "vertexCoord_vec");
    copyFromNdarray_Vec3(gs.get(), "xEdge", "yEdge", "zEdge", reader->edgeCoord_vec, "edgeCoord_vec");
    
    copyFromNdarray_Vec2(gs.get(), "latVertex", "lonVertex", reader->vertexLatLon_vec, "vertexLatLon_vec");
    
    copyFromNdarray_Int(gs.get(), "verticesOnCell", reader->verticesOnCell_vec);
    copyFromNdarray_Int(gs.get(), "verticesOnEdge", reader->verticesOnEdge_vec);
    copyFromNdarray_Int(gs.get(), "cellsOnVertex", reader->cellsOnVertex_vec);
    copyFromNdarray_Int(gs.get(), "cellsOnCell", reader->cellsOnCell_vec);
    copyFromNdarray_Int(gs.get(), "nEdgesOnCell", reader->numberVertexOnCell_vec);
    copyFromNdarray_Int(gs.get(), "cellsOnEdge", reader->cellsOnEdge_vec);
    copyFromNdarray_Int(gs.get(), "edgesOnCell", reader->edgesOnCell_vec);
    copyFromNdarray_Double(gs.get(), "refBottomDepth", reader->cellRefBottomDepth_vec);

    reader->mCellsSize = reader->cellCoord_vec.size();
    reader->mEdgesSize = reader->edgeCoord_vec.size();
    reader->mVertexSize = reader->vertexCoord_vec.size();
    reader->mMaxEdgesSize = reader->edgesOnCell_vec.size() / reader->mCellsSize;

    Debug("%-40s = \t [ %d ]", "[MPASOReader]:: mCellsSize",     reader->mCellsSize);
    Debug("%-40s = \t [ %d ]", "[MPASOReader]:: mEdgesSize",     reader->mEdgesSize);
    Debug("%-40s = \t [ %d ]", "[MPASOReader]:: mMaxEdgesSize",  reader->mMaxEdgesSize);
    Debug("%-40s = \t [ %d ]", "[MPASOReader]:: mVertexSize",    reader->mVertexSize);


    
    return reader;
}

MPASOReader::Ptr MPASOReader::readSolData(const std::string& yaml_path, const std::string& data_name, const int& timestep)
{
    std::shared_ptr<MPASOReader> reader(new MPASOReader(yaml_path));
    reader->mStream = std::make_shared<ftk::stream>();
    reader->mStream->parse_yaml(yaml_path); 
    
    // data
    int fi, index;
    auto sub = reader->mStream->substreams[1];
    auto it = std::find_if(sub->filenames.begin(), sub->filenames.end(),
        [&data_name](const std::string& filename) {
            return filename.find(data_name) != std::string::npos;
    });
    
    if (it == sub->filenames.end()) {
        std::cerr << "[MPASOReader]::Error: Data file with name containing '" << data_name << "' not found in YAML." << std::endl;
        exit(-1);
    }
    else
    {
        fi = static_cast<int>(std::distance(sub->filenames.begin(), it));
        index = sub->first_timestep_per_file[fi] + timestep;
    }
    
    auto gs = reader->mStream->read(index);
    reader->mGroupT = gs;

    reader->mTimesteps = timestep;
    if (reader->mTimesteps < 0)
    {
        std::cerr << "[MPASOReader]::Error: Invalid timestep index " << reader->mTimesteps << std::endl;
        exit(-1);
    }
    auto dataName = removeFileExtension(*it);
    reader->mDataName = dataName;
    reader->mFolderName = reader->mStream->path_prefix;

    std::vector<char> time_vec_s;
    std::vector<char> time_vec_e;
   
    Debug("[MPASOReader]::readSolData::timestep = %d", timestep);
    copyFromNdarray_Double(gs.get(), "bottomDepth", reader->cellBottomDepth_vec);
    copyFromNdarray_Double(gs.get(), "seaSurfaceHeight", reader->cellSurfaceHeight_vec);
    copyFromNdarray_Double(gs.get(), "velocityZonal", reader->cellZonalVelocity_vec);
    copyFromNdarray_Double(gs.get(), "velocityMeridional", reader->cellMeridionalVelocity_vec);
    copyFromNdarray_Double(gs.get(), "layerThickness", reader->cellLayerThickness_vec);
    copyFromNdarray_Double(gs.get(), "zTop", reader->cellZTop_vec);
    copyFromNdarray_Double(gs.get(), "normalVelocity", reader->cellNormalVelocity_vec);
    copyFromNdarray_Char(gs.get(), "xtime", time_vec_s);
    reader->mTimeStamp = std::string(time_vec_s.begin(), time_vec_s.end());
    
    if (reader->cellSurfaceHeight_vec.size() != 0)
    {
        reader->mVertLevels = reader->cellLayerThickness_vec.size() / reader->cellSurfaceHeight_vec.size();
        reader->mVertLevelsP1 = reader->mVertLevels + 1;
    }
    else if (reader->cellBottomDepth_vec.size() != 0 && reader->cellSurfaceHeight_vec.size() == 0)
    {
        reader->mVertLevels = reader->cellLayerThickness_vec.size() / reader->cellBottomDepth_vec.size();
        reader->mVertLevelsP1 = reader->mVertLevels + 1;
    }
        
    
    Debug("%-40s = \t [ %s ]", "[MPASOReader]:: mTimeStamp", reader->mTimeStamp.c_str());
    Debug("%-40s = \t [ %d ]", "[MPASOReader]:: mVertLevels", reader->mVertLevels);
    Debug("%-40s = \t [ %d ]", "[MPASOReader]:: mVertLevelsP1", reader->mVertLevelsP1);



    return reader;
}

void MPASOReader::appendSolData(MPASOReader* reader, const std::string& yaml_path, const int& timestep)
{
    
    // reader->mStream->parse_yaml(yaml_path); 

    // record reading time
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    auto gs = reader->mStream->read(timestep);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    Debug("[MPASOReader]::Reading time: %.6f seconds", time_span.count());

    reader->mTimesteps = timestep;
    if (reader->mTimesteps < 0)
    {
        std::cerr << "[MPASOReader]::Error: Invalid timestep index " << reader->mTimesteps << std::endl;
        exit(-1);
    }
    auto dataName = removeFileExtension(reader->mStream->substreams[1]->filenames[0]);
    reader->mDataName = dataName;
    reader->mFolderName = reader->mStream->path_prefix;

    // reader->readData((stream->substreams[0]->filenames[0]).c_str());
    // reader->readSol((stream->substreams[1]->filenames[0]).c_str());

    std::vector<char> time_vec_s;
    std::vector<char> time_vec_e;

    Debug("[MPASOReader]::appendSolData::timestep = %d", timestep);
    copyFromNdarray_Double(gs.get(), "bottomDepth", reader->cellBottomDepth_vec);
    copyFromNdarray_Double(gs.get(), "seaSurfaceHeight", reader->cellSurfaceHeight_vec);
    copyFromNdarray_Double(gs.get(), "velocityZonal", reader->cellZonalVelocity_vec);
    copyFromNdarray_Double(gs.get(), "velocityMeridional", reader->cellMeridionalVelocity_vec);
    copyFromNdarray_Double(gs.get(), "layerThickness", reader->cellLayerThickness_vec);
    copyFromNdarray_Double(gs.get(), "zTop", reader->cellZTop_vec);
    copyFromNdarray_Double(gs.get(), "normalVelocity", reader->cellNormalVelocity_vec);
    copyFromNdarray_Char(gs.get(), "xtime", time_vec_s);
    reader->mTimeStamp = std::string(time_vec_s.begin(), time_vec_s.end());


    if (reader->cellSurfaceHeight_vec.size() != 0)
    {
        reader->mVertLevels = reader->cellLayerThickness_vec.size() / reader->cellSurfaceHeight_vec.size();
        reader->mVertLevelsP1 = reader->mVertLevels + 1;
    }
    else if (reader->cellBottomDepth_vec.size() != 0 && reader->cellSurfaceHeight_vec.size() == 0)
    {
        reader->mVertLevels = reader->cellLayerThickness_vec.size() / reader->cellBottomDepth_vec.size();
        reader->mVertLevelsP1 = reader->mVertLevels + 1;
    }
        
    
    Debug("%-40s = \t [ %s ]", "[MPASOReader]:: mTimeStamp", reader->mTimeStamp.c_str());
    Debug("%-40s = \t [ %d ]", "[MPASOReader]:: mVertLevels", reader->mVertLevels);
    Debug("%-40s = \t [ %d ]", "[MPASOReader]:: mVertLevelsP1", reader->mVertLevelsP1);

}

void MPASOReader::copyFromNdarray_Int(ftk::ndarray_group* g, std::string value, std::vector<size_t>& vec)
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

void MPASOReader::copyFromNdarray_Vec2(ftk::ndarray_group* g, std::string xValue, std::string yValue, std::vector<vec2>& vec, std::string name)
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

void MPASOReader::copyFromNdarray_Vec3(ftk::ndarray_group* g, std::string xValue, std::string yValue, std::string zValue, std::vector<vec3>& vec, std::string name)
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
        Error("[MPASOReader]::Missing data in ndarray_group for %s", name.c_str());
    }
}


void MPASOReader::copyFromNdarray_Double(ftk::ndarray_group* g, std::string value, std::vector<double>& vec)
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

void MPASOReader::copyFromNdarray_Char(ftk::ndarray_group* g, std::string value, std::vector<char>& vec)
{
    if (g->has(value))
    {
        Debug("[Ndarray]::%s found [✓]", value.c_str());
        auto tmp_get = g->get(value);
        auto tmp_ptr = std::dynamic_pointer_cast<ftk::ndarray<char>>(g->get(value));
        if (tmp_ptr) // Check if the cast was successful
        {
            auto tmp_vec = tmp_ptr->std_vector();
            vec.resize(tmp_vec.size());
            for (auto i = 0; i < tmp_vec.size(); i++)
                vec[i] = static_cast<double>(tmp_vec[i]);
            Debug("[Ndarray]::loading _vec %-30s= \t [ %-8d ] \t type = [ char ]",
                value.c_str(),
                vec.size());
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

void MPASOReader::copyFromNdarray_Float(ftk::ndarray_group* g, std::string value, std::vector<float>& vec)
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


void MPASOReader::readFromBlock_Vec3(const std::string& filename, std::vector<vec3>& vec)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "[MPASOReader]::Error: Unable to open file " << filename << std::endl;
        return;
    }

    vec.clear();
    int dataSize;
    file.read(reinterpret_cast<char*>(&dataSize), sizeof(int));
    vec.resize(dataSize);
    for (std::size_t i = 0; i < dataSize; ++i)
    {
        double x, y, z;
        file.read(reinterpret_cast<char*>(&x), sizeof(double));
        file.read(reinterpret_cast<char*>(&y), sizeof(double));
        file.read(reinterpret_cast<char*>(&z), sizeof(double));
        vec3 data = vec3{x, y, z};
        vec[i] = data;
    }
    file.close();
}

void MPASOReader::readFromBlock_Int(const std::string& filename, std::vector<size_t>& vec)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "[MPASOReader]::Error: Unable to open file " << filename << std::endl;
        return;
    }

    vec.clear();
    int dataSize;
    file.read(reinterpret_cast<char*>(&dataSize), sizeof(int));
    vec.resize(dataSize);
    for (std::size_t i = 0; i < dataSize; ++i)
    {
        int value;
        file.read(reinterpret_cast<char*>(&value), sizeof(int));
        size_t value_t = static_cast<size_t>(value);
        vec[i] = value_t;
    }
    file.close();
}

void MPASOReader::readFromBlock_IntBasedK(const std::string& filename, std::vector<size_t>& vec, int K)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "[MPASOReader]::Error: Unable to open file " << filename << std::endl;
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
    for (int i = 0; i < dataSize; i++)
    {
        for (int j = 0; j < k; j++) {
            int localVertexId;
            file.read(reinterpret_cast<char*>(&localVertexId), sizeof(int));
            vec[i * k + j] = static_cast<size_t>(localVertexId);
        }
    }
    file.close();
    K = k;
    Debug("[MPASOReader]::Loaded %s with %d entries and %d components each", filename.c_str(), dataSize, k);
}