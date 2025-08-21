#include "Core/MPASOReader.h"
using namespace MOPS;
#define NC_CHECK(err) \
    if ((err) != NC_NOERR) { \
        std::cerr << "NetCDF error: " << nc_strerror(err) << std::endl; \
        std::exit(1); \
    }

MPASOReader::MPASOReader(const std::string& path) 
{
	this->path = path;
	// ncFile = std::make_unique<netCDF::NcFile>(path, netCDF::NcFile::read);
    int err = nc_open(path.c_str(), NC_NOWRITE, &ncid);
    NC_CHECK(err);


    mCellsSize = -1;
    mMaxEdgesSize = -1;
    mVertexSize = -1;
    mTimesteps = -1;
    mVertLevels = -1;
    mVertLevelsP1 = -1;
 
}

MPASOReader::~MPASOReader()
{
    int err = nc_close(ncid);
    if (err != NC_NOERR) {
        std::cerr << "Error closing netCDF file: " << nc_strerror(err) << std::endl;
    }
}


void MPASOReader::readData()
{
    int err, dimid;
    size_t tmp;

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

   

    Debug("%-40s = \t [ %d ]", "[MPASOReader]::loading mCellsSize", mCellsSize);
    Debug("%-40s = \t [ %d ]", "[MPASOReader]::loading mEdgesSize", mEdgesSize);
    Debug("%-40s = \t [ %d ]", "[MPASOReader]::loading mMaxEdgesSize", mMaxEdgesSize);
    Debug("%-40s = \t [ %d ]", "[MPASOReader]::loading mVertexSize", mVertexSize);

}

void MPASOReader::readSol(const int& timestep)
{
    int err, dimid;
    size_t tmp;

   
    err = nc_inq_dimid(ncid, "Time", &dimid);
    NC_CHECK(err);
    err = nc_inq_dimlen(ncid, dimid, &tmp);
    NC_CHECK(err);
    mTimesteps = static_cast<int>(tmp);

    err = nc_inq_dimid(ncid, "nVertLevels", &dimid);
    NC_CHECK(err);
    err = nc_inq_dimlen(ncid, dimid, &tmp);
    NC_CHECK(err);
    mVertLevels = static_cast<int>(tmp);

    Debug("%-40s = \t [ %d ]", "[MPASOReader]::loading mTimesteps", mTimesteps);
    Debug("%-40s = \t [ %d ]", "[MPASOReader]::loading mVertLevels", mVertLevels);


    // err = nc_inq_dimid(ncid, "nVertLevelsP1", &dimid);
    // NC_CHECK(err);
    // err = nc_inq_dimlen(ncid, dimid, &tmp);
    // NC_CHECK(err);
    // mVertLevelsP1 = static_cast<int>(tmp);

    currentTimestep = timestep;
}


/*
void MPASOReader::loadingData()
{
    // get size
    mCellsSize = ncFile.get()->getDim("nCells").getSize();
    mEdgesSize = ncFile.get()->getDim("nEdges").getSize();
    mMaxEdgesSize = ncFile.get()->getDim("maxEdges").getSize();
    mVertexSize = ncFile.get()->getDim("nVertices").getSize();
    mTimesteps = ncFile.get()->getDim("Time").getSize();
    mVertLevels = ncFile.get()->getDim("nVertLevels").getSize();
    mVertLevelsP1 = ncFile.get()->getDim("nVertLevelsP1").getSize();
    
    // loading data
    loadingVec2(mVertexSize, vertexLatLon_vec, "latVertex", "lonVertex");
    loadingVec3(mVertexSize, vertexCoord_vec, "xVertex", "yVertex", "zVertex");
    loadingVec3(mCellsSize, cellCoord_vec, "xCell", "yCell", "zCell");
    loadingVec3(mEdgesSize, edgeCoord_vec, "xEdge", "yEdge", "zEdge");
    loadingInt(mCellsSize * mMaxEdgesSize, verticesOnCell_vec, "verticesOnCell");
    loadingInt(mVertexSize * 3, cellsOnVertex_vec, "cellsOnVertex");
    loadingInt(mCellsSize * mMaxEdgesSize, cellsOnCell_vec, "cellsOnCell");
    loadingInt(mCellsSize, numberVertexOnCell_vec, "nEdgesOnCell");
    loadingInt(mEdgesSize * 2, cellsOnEdge_vec, "cellsOnEdge");
    loadingInt(mCellsSize * mMaxEdgesSize, edgesOnCell_vec, "edgesOnCell");

    Debug("[MPASOReader]::loading mCellsSize = \t [ %d ]", mCellsSize);
    Debug("[MPASOReader]::loading mEdgesSize = \t [ %d ]", mEdgesSize);
    Debug("[MPASOReader]::loading mMaxEdgesSize = \t [ %d ]", mMaxEdgesSize);
    Debug("[MPASOReader]::loading mVertexSize = \t [ %d ]", mVertexSize);
    Debug("[MPASOReader]::loading mTimesteps = \t [ %d ]", mTimesteps);
    Debug("[MPASOReader]::loading mVertLevels = \t [ %d ]", mVertLevels);
    Debug("[MPASOReader]::loading mVertLevelsP1 = \t [ %d ]", mVertLevelsP1);
}

void MPASOReader::loadingVelocity(const int& timestep)
{

    // get size
    mCellsSize = ncFile.get()->getDim("nCells").getSize();
    mEdgesSize = ncFile.get()->getDim("nEdges").getSize();
    mMaxEdgesSize = ncFile.get()->getDim("maxEdges").getSize();
    mVertexSize = ncFile.get()->getDim("nVertices").getSize();
    mTimesteps = ncFile.get()->getDim("Time").getSize();
    mVertLevels = ncFile.get()->getDim("nVertLevels").getSize();
    mVertLevelsP1 = ncFile.get()->getDim("nVertLevelsP1").getSize();

    //std::cout << "Finished pre information loading" << std::endl;

    // loading velocity
    loadingDouble(mCellsSize, cellBottomDepth_vec, "bottomDepth");
    ReadNormalVelocity(timestep, cellNormalVelocity_vec);
    //ReadMeridionalVelocity(timestep, cellMeridionalVelocity_vec);
    //ReadZonalVelocity(timestep, cellZonalVelocity_vec);
    ReadLayerThickness(timestep, cellLayerThickness_vec);



    //ReadVelocity(timestep, cellVelocity_vec);
    //ReadVertVelocityTop(timestep, cellVertVelocity_vec);
    //ReadLayerThickness(timestep, cellLayerThickness_vec);
    ReadZTop(timestep, cellZTop_vec);

    currentTimestep = timestep;
}

void MPASOReader::loadingInt(const int size, std::vector<size_t>& data, std::string xValue)
{
    

    if (!data.empty()) data.clear();
    data.resize(size);

    auto Var = ncFile.get()->getVar(xValue);
    auto Value = std::make_unique<int[]>(size);
    Var.getVar(Value.get());

    for (auto i = 0; i < data.size(); ++i)
        data[i] = Value[i];
}


void MPASOReader::loadingDouble(const int size, std::vector<double>& data, std::string xValue)
{
    
    if (!data.empty()) data.clear();
    data.resize(size);

    auto Var = ncFile.get()->getVar(xValue);
    auto Value = std::make_unique<double[]>(size);
    Var.getVar(Value.get());

    for (auto i = 0; i < data.size(); ++i)
        data[i] = Value[i];
}

void MPASOReader::loadingVec2(const int size, std::vector<vec2>& data, std::string xValue, std::string yValue)
{
    //auto nVerticesSize = (mVertexSize != -1) ? mVertexSize : ncFile->getDim("nVertices").getSize();

    if (!data.empty()) data.clear();
    data.resize(size);

    auto xVar = ncFile.get()->getVar(xValue);
    auto yVar = ncFile.get()->getVar(yValue);

    auto x_Value = std::make_unique<double[]>(size);
    auto y_Value = std::make_unique<double[]>(size);
    
    xVar.getVar(x_Value.get());
    yVar.getVar(y_Value.get());


    for (size_t i = 0; i < size; ++i)
    {
        data[i] = vec2(x_Value[i], y_Value[i]);
    }
}

void MPASOReader::loadingVec3(const int size, std::vector<vec3>& data, std::string xValue, std::string yValue, std::string zVale)
{
    //auto nVerticesSize = (mVertexSize != -1) ? mVertexSize : ncFile->getDim("nVertices").getSize();
    
    if (!data.empty()) data.clear();
    data.resize(size);

    auto xVar = ncFile.get()->getVar(xValue);
    auto yVar = ncFile.get()->getVar(yValue);
    auto zVar = ncFile.get()->getVar(zVale);

    auto x_Value = std::make_unique<double[]>(size);
    auto y_Value = std::make_unique<double[]>(size);
    auto z_Value = std::make_unique<double[]>(size);
    xVar.getVar(x_Value.get());
    yVar.getVar(y_Value.get());
    zVar.getVar(z_Value.get());
   
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = vec3(x_Value[i], y_Value[i], z_Value[i]);
    }
}



void MPASOReader::ReadVelocity(const int timestep, std::vector<vec3>& data)
{
    Debug(("[MPASOReader]::loading Cell Velocity at t = " + std::to_string(timestep)).c_str());

    auto nCellsSize     = (mCellsSize != -1)    ? mCellsSize    : ncFile.get()->getDim("nCells").getSize();
    auto nVertLevels    = (mVertLevels != -1)   ? mVertLevels   : ncFile.get()->getDim("nVertLevels").getSize();
    auto nTimesteps     = (mTimesteps != -1)    ? mTimesteps    : ncFile.get()->getDim("Time").getSize();

    std::string velX_str = "velocityX";
    std::string velY_str = "velocityY";
    std::string velZ_str = "velocityZ";

    std::vector<netCDF::NcDim> dims;
    std::vector<size_t> start;
    std::vector<size_t> count;
    double* cellVel_tmp;

    auto velX = ncFile.get()->getVar(velX_str.c_str());
    auto velY = ncFile.get()->getVar(velY_str.c_str());
    auto velZ = ncFile.get()->getVar(velZ_str.c_str());
    dims = velX.getDims();

    auto cellNodeNum = nCellsSize * nVertLevels;
    start.push_back(timestep); start.push_back(0); start.push_back(0);
    count.push_back(1); count.push_back(nCellsSize); count.push_back(nVertLevels);

    cellVel_tmp = new double[cellNodeNum];
    if (!data.empty()) data.clear();
    data.resize(cellNodeNum);


    velX.getVar(start, count, cellVel_tmp);
    for (int i = 0; i < cellNodeNum; i++)
        data[i].x() = cellVel_tmp[i];
    velY.getVar(start, count, cellVel_tmp);
    for (int i = 0; i < cellNodeNum; i++)
        data[i].y() = cellVel_tmp[i];
    velZ.getVar(start, count, cellVel_tmp);
    for (int i = 0; i < cellNodeNum; i++)
        data[i].z() = cellVel_tmp[i];

    delete[] cellVel_tmp;
}

void MPASOReader::ReadLayerThickness(const int timestep, std::vector<double>& data)
{
    Debug(("[MPASOReader]::loading Layer Thickness at t = " + std::to_string(timestep)).c_str());
    

    std::vector<size_t> start;
    std::vector<size_t> count;
    double* layerThickness_tmp;
    std::string lt_str = "layerThickness";

    auto nCellsSize     = (mCellsSize != -1)    ? mCellsSize    : ncFile.get()->getDim("nCells").getSize();
    auto nVertLevels    = (mVertLevels != -1)   ? mVertLevels   : ncFile.get()->getDim("nVertLevels").getSize();
    auto nTimesteps     = (mTimesteps != -1)    ? mTimesteps    : ncFile.get()->getDim("Time").getSize();

    
    auto lt = ncFile.get()->getVar(lt_str.c_str());
    
    start.push_back(timestep); start.push_back(0); start.push_back(0);
    count.push_back(1); count.push_back(nCellsSize); count.push_back(nVertLevels);
    
    layerThickness_tmp = new double[nCellsSize * nVertLevels];
    lt.getVar(start, count, layerThickness_tmp);

   
    if (!data.empty()) data.clear();
    data.resize(nCellsSize * nVertLevels);


    for(int i = 0; i < nCellsSize * nVertLevels; i++)
        data[i] = layerThickness_tmp[i];

    delete[] layerThickness_tmp;
   
    
}

void MPASOReader::ReadNormalVelocity(const int timestep, std::vector<double>& data)
{
    Debug(("[MPASOReader]::loading Normal Velocity at t = " + std::to_string(timestep)).c_str());

    std::string ztop_str = "normalVelocity";
    
    double* zTop_tmp;
    std::vector<size_t> start;
    std::vector<size_t> count;
    auto ztop = ncFile.get()->getVar(ztop_str.c_str());



    auto nEdges         = (mEdgesSize != -1)    ? mEdgesSize    : ncFile.get()->getDim("nEdges").getSize();
    auto nVertLevels    = (mVertLevels != -1)   ? mVertLevels   : ncFile.get()->getDim("nVertLevels").getSize();
    auto nTimesteps     = (mTimesteps != -1)    ? mTimesteps    : ncFile.get()->getDim("Time").getSize();


    start.push_back(timestep); start.push_back(0); start.push_back(0);
    count.push_back(1); count.push_back(nEdges); count.push_back(nVertLevels);
    
    zTop_tmp = new double[nEdges * nVertLevels];
    ztop.getVar(start, count, zTop_tmp);

    if(!data.empty()) data.clear();
    data.resize(nEdges * nVertLevels);
    int count_zz = 0, count_ff = 0;
    for(int i = 0; i < nEdges * nVertLevels; i++)
    {

        data[i] = zTop_tmp[i];
            
    }
   
    delete[] zTop_tmp;
}

void MPASOReader::ReadZTop(const int timestep, std::vector<double>& data)
{
    Debug(("[MPASOReader]::loading Z Top at t = " + std::to_string(timestep)).c_str());

    std::string ztop_str = "zTop";
    
    double* zTop_tmp;
    std::vector<size_t> start;
    std::vector<size_t> count;
    auto ztop = ncFile.get()->getVar(ztop_str.c_str());

    double invalidValue = -2.0;

    auto nCellsSize     = (mCellsSize != -1)    ? mCellsSize    : ncFile.get()->getDim("nCells").getSize();
    auto nVertLevels    = (mVertLevels != -1)   ? mVertLevels   : ncFile.get()->getDim("nVertLevels").getSize();
    auto nTimesteps     = (mTimesteps != -1)    ? mTimesteps    : ncFile.get()->getDim("Time").getSize();


    start.push_back(timestep); start.push_back(0); start.push_back(0);
    count.push_back(1); count.push_back(nCellsSize); count.push_back(nVertLevels);
    
    zTop_tmp = new double[nCellsSize * nVertLevels];
    ztop.getVar(start, count, zTop_tmp);

    if(!data.empty()) data.clear();
    data.resize(nCellsSize * nVertLevels);
    int count_zz = 0, count_ff = 0;
    for(int i = 0; i < nCellsSize * nVertLevels; i++)
    {
       

        
        data[i] = zTop_tmp[i];
            
    }
    
    delete[] zTop_tmp;
}

void MPASOReader::ReadVertVelocityTop(const int timestep, std::vector<double>& data)
{
    Debug(("[MPASOReader]::loading Vert Velocity Top at t = " + std::to_string(timestep)).c_str());

    std::string vertvel_str = "vertVelocityTop";
  
    std::vector<size_t> start;
    std::vector<size_t> count;
    double* vertVelocity_tmp;

    auto vertvel = ncFile.get()->getVar(vertvel_str.c_str());


    auto nCellsSize         = (mCellsSize != -1)        ? mCellsSize        : ncFile.get()->getDim("nCells").getSize();
    auto nVertLevels        = (mVertLevels != -1)       ? mVertLevels       : ncFile.get()->getDim("nVertLevels").getSize();
    auto nVertLevelsP1      = (mVertLevelsP1 != -1)     ? mVertLevelsP1     : ncFile.get()->getDim("nVertLevelsP1").getSize();
    auto nTimesteps         = (mTimesteps != -1)        ? mTimesteps        : ncFile.get()->getDim("Time").getSize();

   
    start.push_back(timestep); start.push_back(0); start.push_back(0);
    count.push_back(1); count.push_back(nCellsSize); count.push_back(nVertLevelsP1);

    vertVelocity_tmp = new double[nCellsSize * nVertLevelsP1];
    vertvel.getVar(start, count, vertVelocity_tmp);

    if(!data.empty()) data.clear();
    data.resize(nCellsSize * nVertLevelsP1);
    for(int cellidx = 0; cellidx < nCellsSize; cellidx++)
     {
        int offset = cellidx * nVertLevels;
        int offsetp1 = cellidx * nVertLevelsP1;
        for(int vLevel = 0; vLevel < nVertLevels; vLevel++)
        {
            data[offsetp1 + vLevel] = vertVelocity_tmp[offset + vLevel];
        }
    }

    delete[] vertVelocity_tmp;
}


void MPASOReader::ReadMeridionalVelocity(const int timestep, std::vector<double>& data)
{
    Debug(("[MPASOReader]::loading Surface Meridional Velocity Top at t = " + std::to_string(timestep)).c_str());

    std::string vertvel_str = "surfaceVelocityMeridional";

    std::vector<size_t> start;
    std::vector<size_t> count;
    double* vertVelocity_tmp;

    auto vertvel = ncFile.get()->getVar(vertvel_str.c_str());
    auto nCellsSize         = (mCellsSize != -1)        ? mCellsSize        : ncFile.get()->getDim("nCells").getSize();
    
    start.push_back(timestep); start.push_back(0); start.push_back(0);
    count.push_back(1); count.push_back(nCellsSize); 

    vertVelocity_tmp = new double[nCellsSize];
    vertvel.getVar(start, count, vertVelocity_tmp);

    if(!data.empty()) data.clear();
    data.resize(nCellsSize);
    for(int cellidx = 0; cellidx < nCellsSize; cellidx++)
     {
        int offset = cellidx ;
        int offsetp1 = cellidx ;
        for(int vLevel = 0; vLevel < 1; vLevel++)
        {
            data[offsetp1 + vLevel] = vertVelocity_tmp[offset + vLevel];
        }
    }
    delete[] vertVelocity_tmp;
}

void MPASOReader::ReadZonalVelocity(const int timestep, std::vector<double>& data)
{
    Debug(("[MPASOReader]::loading Surface Zonal Velocity Top at t = " + std::to_string(timestep)).c_str());

    std::string vertvel_str = "surfaceVelocityZonal";

    std::vector<size_t> start;
    std::vector<size_t> count;
    double* vertVelocity_tmp;

    auto vertvel = ncFile.get()->getVar(vertvel_str.c_str());
    auto nCellsSize         = (mCellsSize != -1)        ? mCellsSize        : ncFile.get()->getDim("nCells").getSize();
    
    start.push_back(timestep); start.push_back(0); start.push_back(0);
    count.push_back(1); count.push_back(nCellsSize); 

    vertVelocity_tmp = new double[nCellsSize];
    vertvel.getVar(start, count, vertVelocity_tmp);

    if(!data.empty()) data.clear();
    data.resize(nCellsSize);
    for(int cellidx = 0; cellidx < nCellsSize; cellidx++)
    {
        int offset = cellidx ;
        int offsetp1 = cellidx ;
        for(int vLevel = 0; vLevel < 1; vLevel++)
        {
            data[offsetp1 + vLevel] = vertVelocity_tmp[offset + vLevel];
        }
    }
    delete[] vertVelocity_tmp;
}
*/