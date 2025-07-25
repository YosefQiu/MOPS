#include "Core/MPASOVisualizer.h"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#if MOPS_VTK
#include "IO/VTKFileManager.hpp"
#endif
#include "SYCL/SYCLKernel.h"


#define CHECK_VECTOR(vec, name) \
    if (vec.data() == nullptr || vec.size() == 0) { \
        std::cerr << "[Warning] Vector '" << name << "' is empty or nullptr!" << std::endl; \
    }

using namespace MOPS;

void MPASOVisualizer::VisualizeFixedLayer(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, sycl::queue& sycl_Q)
{
    int width = config->imageSize.x();
    int height = config->imageSize.y();
    auto minLat = config->LatRange.x();
    auto maxLat = config->LatRange.y();
    auto minLon = config->LonRange.x();
    auto maxLon = config->LonRange.y();
    auto fixed_layer = config->FixedLayer;

    std::vector<size_t> grid_info_vec;
    // tbl
    // 0 : mCellsSize
    // 1 : mEdgesSize
    // 2 : mMaxEdgesSize
    // 3 : mVertexSize
    // 4 : mVertLevels
    // 5 : mVertLevelsP1
    grid_info_vec.push_back(mpasoF->mGrid->mCellsSize);
    grid_info_vec.push_back(mpasoF->mGrid->mEdgesSize);
    grid_info_vec.push_back(mpasoF->mGrid->mMaxEdgesSize);
    grid_info_vec.push_back(mpasoF->mGrid->mVertexSize);
    grid_info_vec.push_back(mpasoF->mGrid->mVertLevels);
    grid_info_vec.push_back(mpasoF->mGrid->mVertLevelsP1);


    std::vector<int> cell_id_vec; 
    cell_id_vec.resize(width * height); 
    SYCLKernel::SearchKDTree(cell_id_vec.data(), mpasoF->mGrid.get(), width, height, minLat, maxLat, minLon, maxLon);
    Debug("[MPASOVisualizer]::Finished KD Tree Search....");
    

    
#pragma region sycl_buffer
    sycl::buffer<int, 1> width_buf(&width, 1);
    sycl::buffer<int, 1> height_buf(&height, 1);
    sycl::buffer<double, 1> minLat_buf(&minLat, 1);
    sycl::buffer<double, 1> maxLat_buf(&maxLat, 1);
    sycl::buffer<double, 1> minLon_buf(&minLon, 1);
    sycl::buffer<double, 1> maxLon_buf(&maxLon, 1);
    sycl::buffer<double, 1> img_buf(img->mPixels.data(), sycl::range<1>(img->mPixels.size()));

    sycl::buffer<int, 1> cellID_buf(cell_id_vec.data(), sycl::range<1>(cell_id_vec.size()));
    sycl::buffer<vec3, 1> vertexCoord_buf(mpasoF->mGrid->vertexCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->vertexCoord_vec.size())); // CELL 顶点坐标
    sycl::buffer<vec3, 1> cellCoord_buf(mpasoF->mGrid->cellCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->cellCoord_vec.size()));       // CELL 中心坐标
    sycl::buffer<size_t, 1> numberVertexOnCell_buf(mpasoF->mGrid->numberVertexOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->numberVertexOnCell_vec.size())); // CELL 有几个顶点
    sycl::buffer<size_t, 1> verticesOnCell_buf(mpasoF->mGrid->verticesOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->verticesOnCell_vec.size()));             // 
    sycl::buffer<size_t, 1> cellsOnVertex_buf(mpasoF->mGrid->cellsOnVertex_vec.data(), sycl::range<1>(mpasoF->mGrid->cellsOnVertex_vec.size()));
    sycl::buffer<size_t, 1> grid_info_buf(grid_info_vec.data(), sycl::range<1>(grid_info_vec.size()));

    sycl::buffer<vec3, 1> cellCenterVelocity_buf(mpasoF->mSol_Front->cellCenterVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellCenterVelocity_vec.size()));
    sycl::buffer<vec3, 1> cellVertexVelocity_buf(mpasoF->mSol_Front->cellVertexVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexVelocity_vec.size()));
    sycl::buffer<double, 1> cellVertexZTop_buf(mpasoF->mSol_Front->cellVertexZTop_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexZTop_vec.size()));
    sycl::buffer<double, 1> cellCenterZonalVelocity_buf(mpasoF->mSol_Front->cellZonalVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellZonalVelocity_vec.size()));
    sycl::buffer<double, 1> cellCenterMeridionalVelocity_buf(mpasoF->mSol_Front->cellMeridionalVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellMeridionalVelocity_vec.size()));
#pragma endregion sycl_buffer

    sycl_Q.submit([&](sycl::handler& cgh) {

#pragma region sycl_acc
        auto width_acc = width_buf.get_access<sycl::access::mode::read>(cgh);
        auto height_acc = height_buf.get_access<sycl::access::mode::read>(cgh);
        auto minLat_acc = minLat_buf.get_access<sycl::access::mode::read>(cgh);
        auto maxLat_acc = maxLat_buf.get_access<sycl::access::mode::read>(cgh);
        auto minLon_acc = minLon_buf.get_access<sycl::access::mode::read>(cgh);
        auto maxLon_acc = maxLon_buf.get_access<sycl::access::mode::read>(cgh);
        auto img_acc = img_buf.get_access<sycl::access::mode::read_write>(cgh);

        auto acc_cellID_buf = cellID_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_vertexCoord_buf = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnCell_buf = verticesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnVertex_buf = cellsOnVertex_buf.get_access<sycl::access::mode::read>(cgh);
        int grid_cell_size = mpasoF->mGrid->mCellsSize;
        int grid_max_edge = mpasoF->mGrid->mMaxEdgesSize;
        auto acc_grid_info_buf = grid_info_buf.get_access<sycl::access::mode::read>(cgh);

        auto acc_cellCenterVelocity_buf = cellCenterVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexVelocity_buf = cellVertexVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexZTop_buf = cellVertexZTop_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCenterZonalVelocity_buf = cellCenterZonalVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCenterMeridionalVelocity_buf = cellCenterMeridionalVelocity_buf.get_access<sycl::access::mode::read>(cgh);
#pragma endregion sycl_acc

        sycl::range<2> global_range((height + 7) / 8 * 8, (width + 7) / 8 * 8);
        sycl::range<2> local_range(8, 8);  

        cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), [=](sycl::nd_item<2> idx) {
            int height_index = idx.get_global_id(0);
            int width_index = idx.get_global_id(1);
            int global_id = height_index * width_acc[0] + width_index;
                
            if(height_index < height && width_index < width)
            {
                const int CELL_SIZE = (int)acc_grid_info_buf[0];
                const int max_edge = (int)acc_grid_info_buf[2];
                const int MAX_VERTEX_NUM = 20;
                const int NEIGHBOR_NUM = 3;
                const int TOTAY_ZTOP_LAYER = 80;
                const int VERTLEVELS = 80;
                auto double_nan = std::numeric_limits<double>::quiet_NaN();
                vec3 vec3_nan = { double_nan, double_nan, double_nan };

                // CalcPosition&CellID
                vec2 current_pixel = { height_index, width_index };
                CartesianCoord current_position;
                SphericalCoord current_latlon_r;
                GeoConverter::convertPixelToLatLonToRadians(width_acc[0], height_acc[0], minLat_acc[0], maxLat_acc[0], minLon_acc[0], maxLon_acc[0], current_pixel, current_latlon_r);
                GeoConverter::convertRadianLatLonToXYZ(current_latlon_r, current_position);
                int cell_id = acc_cellID_buf[global_id];


                // Determine whether it is on the mainland 判断是否在大陆上
                // 1.1 Calculate how many vertices are in this cell.
                auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                auto nan = std::numeric_limits<size_t>::max();
                //1.2 Find all candidate vertices
                size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
                SYCLKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx, MAX_VERTEX_NUM, max_edge, acc_verticesOnCell_buf);
                bool is_land = SYCLKernel::IsInOcean(cell_id, max_edge, current_position, acc_numberVertexOnCell_buf, acc_verticesOnCell_buf, acc_vertexCoord_buf);
                if (!is_land)
                {
                    SetPixel(img_acc, width_acc[0], height_acc[0], height_index, width_index, vec3_nan);
                    return;
                }                               

                vec3 imgValue = vec3_nan;

                if (true)//config->PositionType == CalcPositionType::kPoint //TODO
                {
                    //  Calculate the (velocity , coordinates and zTOP) of each Cell vertex.
                    vec3 current_cell_vertices_velocity[MAX_VERTEX_NUM];
                    vec3 current_verteices_positions[MAX_VERTEX_NUM];
                    double current_cell_vertices_ztop[MAX_VERTEX_NUM];
                    //TODO
                    vec3 tmp_center_velocity = acc_cellCenterVelocity_buf[cell_id * VERTLEVELS + fixed_layer];
                    for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
                    {
                        auto VID = current_cell_vertices_idx[v_idx];
                        auto vel = acc_cellVertexVelocity_buf[VID * VERTLEVELS + fixed_layer];
                        auto ztop = acc_cellVertexZTop_buf[VID * VERTLEVELS + fixed_layer];
                        auto pos = acc_vertexCoord_buf[VID];
                        current_cell_vertices_velocity[v_idx] = vel;
                        current_cell_vertices_ztop[v_idx] = ztop;
                        current_verteices_positions[v_idx] = pos;
                    }
                    //  Set the non-existent vertex to NaN.
                    for (auto v_idx = current_cell_vertices_number; v_idx < MAX_VERTEX_NUM; ++v_idx)
                    {
                        current_cell_vertices_velocity[v_idx] = vec3_nan;
                        current_verteices_positions[v_idx] = vec3_nan;
                        current_cell_vertices_ztop[v_idx] = double_nan;
                    }
                    //  Calculate the speed of the current point using the Wachspress coordinates parameter.
                    double current_cell_weight[MAX_VERTEX_NUM];
                    Interpolator::CalcPolygonWachspress(current_position, current_verteices_positions, current_cell_weight, current_cell_vertices_number);
                    for (auto v_idx = current_cell_vertices_number; v_idx < MAX_VERTEX_NUM; ++v_idx)
                    {
                        current_cell_weight[v_idx] = double_nan;
                    }
                    //TODO
                    vec3 current_point_velocity = { 0.0, 0.0, 0.0 };
                    for (auto k = 0; k < current_cell_vertices_number; ++k)
                    {
                        current_point_velocity.x() += current_cell_weight[k] * current_cell_vertices_velocity[k].x();
                        current_point_velocity.y() += current_cell_weight[k] * current_cell_vertices_velocity[k].y();
                        current_point_velocity.z() += current_cell_weight[k] * current_cell_vertices_velocity[k].z();
                    }
                    double zional_velocity, merminoal_velicity;
                    GeoConverter::convertXYZVelocityToENU(current_position, current_point_velocity, zional_velocity, merminoal_velicity);
                    vec3 current_point_velocity_enu = { zional_velocity, merminoal_velicity, 0.0 };
                    imgValue = current_point_velocity_enu;              
                }
                    
                SetPixel(img_acc, width, height, height_index, width_index, imgValue);
            } 
        });
    });
    sycl_Q.wait();
   
}


void MPASOVisualizer::VisualizeFixedDepth(MPASOField* mpasoF, VisualizationSettings* config, std::vector<ImageBuffer<double>>& img_vec, sycl::queue& sycl_Q)
{
    int width = config->imageSize.x();
    int height = config->imageSize.y();
    auto minLat = config->LatRange.x();
    auto maxLat = config->LatRange.y();
    auto minLon = config->LonRange.x();
    auto maxLon = config->LonRange.y();
    auto fixed_depth = -config->FixedDepth;

    std::vector<size_t> grid_info_vec;
    // tbl
    // 0 : mCellsSize
    // 1 : mEdgesSize
    // 2 : mMaxEdgesSize
    // 3 : mVertexSize
    // 4 : mVertLevels
    // 5 : mVertLevelsP1
    grid_info_vec.push_back(mpasoF->mGrid->mCellsSize);
    grid_info_vec.push_back(mpasoF->mGrid->mEdgesSize);
    grid_info_vec.push_back(mpasoF->mGrid->mMaxEdgesSize);
    grid_info_vec.push_back(mpasoF->mGrid->mVertexSize);
    grid_info_vec.push_back(mpasoF->mGrid->mVertLevels);
    // grid_info_vec.push_back(mpasoF->mGrid->mVertLevelsP1);

    std::vector<int> cell_id_vec; 
    cell_id_vec.resize(width * height); 
    SYCLKernel::SearchKDTree(cell_id_vec.data(), mpasoF->mGrid.get(), width, height, minLat, maxLat, minLon, maxLon);

    Debug("[MPASOVisualizer]::Finished KD Tree Search....");

#pragma region sycl_buffer
    sycl::buffer<int, 1> width_buf(&width, 1);
    sycl::buffer<int, 1> height_buf(&height, 1);
    sycl::buffer<double, 1> minLat_buf(&minLat, 1);
    sycl::buffer<double, 1> maxLat_buf(&maxLat, 1);
    sycl::buffer<double, 1> minLon_buf(&minLon, 1);
    sycl::buffer<double, 1> maxLon_buf(&maxLon, 1);
    sycl::buffer<double, 1> depth_buf(&fixed_depth, 1);
    std::vector<sycl::buffer<double, 1>> img_bufs;
    for (auto& img : img_vec)
        img_bufs.emplace_back(img.mPixels.data(), sycl::range<1>(img.mPixels.size()));

    sycl::buffer<int, 1> cellID_buf(cell_id_vec.data(), sycl::range<1>(cell_id_vec.size()));
    sycl::buffer<vec3, 1> vertexCoord_buf(mpasoF->mGrid->vertexCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->vertexCoord_vec.size())); // CELL 顶点坐标
    sycl::buffer<vec3, 1> cellCoord_buf(mpasoF->mGrid->cellCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->cellCoord_vec.size()));       // CELL 中心坐标
    sycl::buffer<size_t, 1> numberVertexOnCell_buf(mpasoF->mGrid->numberVertexOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->numberVertexOnCell_vec.size())); // CELL 有几个顶点
    sycl::buffer<size_t, 1> verticesOnCell_buf(mpasoF->mGrid->verticesOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->verticesOnCell_vec.size()));             // 
    sycl::buffer<size_t, 1> cellsOnVertex_buf(mpasoF->mGrid->cellsOnVertex_vec.data(), sycl::range<1>(mpasoF->mGrid->cellsOnVertex_vec.size()));
    sycl::buffer<size_t, 1> grid_info_buf(grid_info_vec.data(), sycl::range<1>(grid_info_vec.size()));

    sycl::buffer<vec3, 1> cellVertexVelocity_buf(mpasoF->mSol_Front->cellVertexVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexVelocity_vec.size()));
    sycl::buffer<double, 1> cellVertexZTop_buf(mpasoF->mSol_Front->cellVertexZTop_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexZTop_vec.size()));

    
    std::vector<std::string> attr_names;
    std::vector<sycl::buffer<double, 1>> attr_bufs;

    for (const auto& [name, vec] : mpasoF->mSol_Front->mDoubleAttributes_CtoV)
    {
        attr_names.push_back(name);
        attr_bufs.emplace_back(vec.data(), sycl::range<1>(vec.size()));
    }

    CHECK_VECTOR(img_vec[0].mPixels, "img_vec[0].mPixels");
    CHECK_VECTOR(cell_id_vec, "cell_id_vec");
    CHECK_VECTOR(mpasoF->mGrid->vertexCoord_vec, "vertexCoord_vec");
    CHECK_VECTOR(mpasoF->mGrid->cellCoord_vec, "cellCoord_vec");
    CHECK_VECTOR(mpasoF->mGrid->numberVertexOnCell_vec, "numberVertexOnCell_vec");
    CHECK_VECTOR(mpasoF->mGrid->verticesOnCell_vec, "verticesOnCell_vec");
    CHECK_VECTOR(mpasoF->mGrid->cellsOnVertex_vec, "cellsOnVertex_vec");
    CHECK_VECTOR(grid_info_vec, "grid_info_vec");
    CHECK_VECTOR(mpasoF->mSol_Front->cellVertexVelocity_vec, "cellVertexVelocity_vec");
    CHECK_VECTOR(mpasoF->mSol_Front->cellVertexZTop_vec, "cellVertexZTop_vec");

    for (const auto& [name, vec] : mpasoF->mSol_Front->mDoubleAttributes_CtoV) {
        CHECK_VECTOR(vec, name);
    }

    
#pragma endregion sycl_buffer

    sycl_Q.submit([&](sycl::handler& cgh)  {
#pragma region sycl_acc
        constexpr int MAX_OUTPUTS = 8;  // max img_vec size
        constexpr int MAX_ATTRS = 8;    // max attribute size
        auto width_acc = width_buf.get_access<sycl::access::mode::read>(cgh);
        auto height_acc = height_buf.get_access<sycl::access::mode::read>(cgh);
        auto minLat_acc = minLat_buf.get_access<sycl::access::mode::read>(cgh);
        auto maxLat_acc = maxLat_buf.get_access<sycl::access::mode::read>(cgh);
        auto minLon_acc = minLon_buf.get_access<sycl::access::mode::read>(cgh);
        auto maxLon_acc = maxLon_buf.get_access<sycl::access::mode::read>(cgh);
        auto depth_acc = depth_buf.get_access<sycl::access::mode::read>(cgh);
        // auto img_acc = img_buf.get_access<sycl::access::mode::read_write>(cgh);
        std::array<sycl::accessor<double, 1>, MAX_OUTPUTS> img_accs;
        int img_count = img_bufs.size();
        for (int i = 0; i < img_count; ++i) 
        {
            img_accs[i] = img_bufs[i].get_access<sycl::access::mode::read_write>(cgh);
        }


        auto acc_cellID_buf = cellID_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_vertexCoord_buf = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnCell_buf = verticesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnVertex_buf = cellsOnVertex_buf.get_access<sycl::access::mode::read>(cgh);
        auto grid_cell_size = mpasoF->mGrid->mCellsSize;
        auto acc_grid_info_buf = grid_info_buf.get_access<sycl::access::mode::read>(cgh);

        auto acc_cellVertexVelocity_buf = cellVertexVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexZTop_buf = cellVertexZTop_buf.get_access<sycl::access::mode::read>(cgh);

        std::array<sycl::accessor<double, 1, sycl::access::mode::read>, MAX_ATTRS> acc_attr_bufs;
        int attr_count = attr_bufs.size();

        for (int i = 0; i < attr_count; ++i) {
            acc_attr_bufs[i] = attr_bufs[i].get_access<sycl::access::mode::read>(cgh);
        }


#pragma endregion sycl_acc
        
    sycl::stream out(1024, 256, cgh);
    sycl::range<2> global_range((height + 7) / 8 * 8, (width + 7) / 8 * 8);
    sycl::range<2> local_range(8, 8);
                        
    cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), [=](sycl::nd_item<2> idx) {

        int height_index = idx.get_global_id(0);
        int width_index = idx.get_global_id(1);
        int global_id = height_index * width_acc[0] + width_index;
                        
        if(height_index < height && width_index < width)
        {
            const int CELL_SIZE = (int)acc_grid_info_buf[0];
            const int max_edge = (int)acc_grid_info_buf[2];
            const int MAX_VERTEX_NUM = 20;
            const int NEIGHBOR_NUM = 3;
            const int TOTAY_ZTOP_LAYER = (int)acc_grid_info_buf[4];
            const int MAX_VERTLEVELS = 100;
            const int VERTLEVELS = (int)acc_grid_info_buf[4];
            const double DEPTH = depth_acc[0];
            auto nan = std::numeric_limits<size_t>::max();
            auto double_nan = std::numeric_limits<double>::quiet_NaN();
            vec3 vec3_nan = { double_nan, double_nan, double_nan };

            //CalcPosition&CellID
            vec2 current_pixel = { height_index, width_index };
            CartesianCoord current_position;
            SphericalCoord current_latlon_r;
            GeoConverter::convertPixelToLatLonToRadians(width_acc[0], height_acc[0], minLat_acc[0], maxLat_acc[0], minLon_acc[0], maxLon_acc[0], current_pixel, current_latlon_r);
            GeoConverter::convertRadianLatLonToXYZ(current_latlon_r, current_position);
            int cell_id = acc_cellID_buf[global_id];

            // Determine whether it is on the mainland 判断是否在大陆上
            // 1.1 Calculate how many vertices are in this cell.
            auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
            // 1.2 Find all candidate vertices
            size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
            SYCLKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx, MAX_VERTEX_NUM, max_edge, acc_verticesOnCell_buf);
            bool is_land = SYCLKernel::IsInOcean(cell_id, max_edge, current_position, acc_numberVertexOnCell_buf, acc_verticesOnCell_buf, acc_vertexCoord_buf);
            if (!is_land)
            {
                for (int i = 0; i < img_count; ++i)
                {
                    SetPixel(img_accs[i], width_acc[0], height_acc[0], height_index, width_index, vec3_nan);
                }
                return;
            }
         
            vec3 imgValue = vec3_nan;
            double current_point_ztop_vec[MAX_VERTLEVELS];

            vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
            double current_cell_vertex_weight[MAX_VERTEX_NUM];
            bool rc = SYCLKernel::GetCellVertexPos(current_cell_vertex_pos, current_cell_vertices_idx, MAX_VERTEX_NUM, current_cell_vertices_number, acc_vertexCoord_buf);
            if (!rc)
            {
                out << "[ERROR]:: GetCellVertexPos Failed....(VLA < current_cell_vertices_number)" << sycl::endl;
                return;
            }
            // washpress
            Interpolator::CalcPolygonWachspress(current_position, current_cell_vertex_pos, current_cell_vertex_weight, current_cell_vertices_number);
                      
            for (auto k = 0; k < VERTLEVELS; ++k)
            {
                double current_point_ztop_in_layer = 0.0;
                // Get the ztop of each vertex, and the non-existent vertex is set to NaN.
                double current_cell_vertex_ztop[MAX_VERTEX_NUM];
                for (auto v_idx = 0; v_idx < current_cell_vertices_number; v_idx++)
                {
                    auto VID = current_cell_vertices_idx[v_idx];
                    double ztop = acc_cellVertexZTop_buf[VID * TOTAY_ZTOP_LAYER + k];
                    current_cell_vertex_ztop[v_idx] = ztop;
                }
                for (auto v_idx = current_cell_vertices_number; v_idx < MAX_VERTEX_NUM; ++v_idx)
                {
                    current_cell_vertex_ztop[v_idx] = double_nan;
                }
                    
                // Calculate the ZTOP of the current point
                for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
                {
                    current_point_ztop_in_layer += current_cell_vertex_weight[v_idx] * current_cell_vertex_ztop[v_idx];
                }
                current_point_ztop_vec[k] = current_point_ztop_in_layer;
            }
            for (auto k = VERTLEVELS; k < MAX_VERTLEVELS; k++)
            {
                current_point_ztop_vec[k] = double_nan;
            }

            const double EPSILON = 1e-6;
            int local_layer = 0;
            for (int k = 1; k < VERTLEVELS; ++k)
            {
                if (DEPTH <= current_point_ztop_vec[k - 1] - EPSILON && DEPTH >= current_point_ztop_vec[k] + EPSILON)
                {
                    local_layer = k;
                    break;
                }
            }
              
            vec3 current_point_velocity_enu = {0.0, 0.0, 0.0};
            vec3 current_point_attr_value = {0.0, 0.0, 0.0};
            if (local_layer == 0)
            {
                imgValue = vec3_nan;
                for (int i = 0; i < img_count; ++i)
                {
                    SetPixel(img_accs[i], width, height, height_index, width_index, imgValue);
                }
                return;
            }
            else
            {
                double attr_value_vec[MAX_ATTRS];
                auto layer = local_layer;
                double ztop_layer1, ztop_layer2;
                ztop_layer1 = current_point_ztop_vec[layer];
                ztop_layer2 = current_point_ztop_vec[layer - 1];
                double t = (sycl::fabs(DEPTH) - sycl::fabs(ztop_layer1)) / (sycl::fabs(ztop_layer2) - sycl::fabs(ztop_layer1));
                double current_point_ztop;
                current_point_ztop = t * ztop_layer1 + (1 - t) * ztop_layer2;

                vec3 final_vel;
                // Calculate the speed of these two layers.
                vec3 vertex_vel1[MAX_VERTEX_NUM];
                vec3 vertex_vel2[MAX_VERTEX_NUM];
                vec3 current_point_vel1 = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, TOTAY_ZTOP_LAYER, layer, acc_cellVertexVelocity_buf);
                
                vec3 current_point_vel2 = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                    MAX_VERTEX_NUM, current_cell_vertices_number, TOTAY_ZTOP_LAYER, layer - 1, acc_cellVertexVelocity_buf);
                
                final_vel.x() = t * current_point_vel2.x() + (1 - t) * current_point_vel1.x();
                final_vel.y() = t * current_point_vel2.y() + (1 - t) * current_point_vel1.y();
                final_vel.z() = t * current_point_vel2.z() + (1 - t) * current_point_vel1.z();
                        
                double zional_velocity, merminoal_velicity;
                GeoConverter::convertXYZVelocityToENU(current_position, final_vel, zional_velocity, merminoal_velicity);
                double total_velocity = sycl::sqrt(zional_velocity * zional_velocity + merminoal_velicity * merminoal_velicity);
                current_point_velocity_enu = {zional_velocity, merminoal_velicity, total_velocity};

                for (int attr_idx = 0; attr_idx < attr_count; ++attr_idx)
                {
                    double attr_value = SYCLKernel::CalcAttribute(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, TOTAY_ZTOP_LAYER, layer, acc_attr_bufs[attr_idx]);
                    attr_value_vec[attr_idx] = attr_value;
                    
                }
                current_point_attr_value = {attr_value_vec[0], attr_value_vec[1], 0.0};
            }
                
            SetPixel(img_accs[0], width, height, height_index, width_index, current_point_velocity_enu);
            SetPixel(img_accs[1], width, height, height_index, width_index, current_point_attr_value);
        }

        });
    });

    sycl_Q.wait();

}




//TODO: Temporarily unavailable, it will be released later.
[[deprecated]]
void MPASOVisualizer::VisualizeFixedLatitude(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, sycl::queue& sycl_Q)
{
/*
    int width = config->imageSize.x();
    int height = config->imageSize.y();
    auto minDepth = config->DepthRange.x();
    auto maxDepth = config->DepthRange.y();
    auto minLon = config->LonRange.x();
    auto maxLon = config->LonRange.y();
    auto fixed_lat = config->FixedLatitude;


   std::vector<int> cell_id_vec;
   cell_id_vec.resize(width * height);
   
   float i_step = (maxDepth - minDepth) / height;
   float j_step = (maxLon - minLon) / width;
   for (float i = minDepth; i < maxDepth; i = i + i_step)
   {
       for (float j = minLon; j < maxLon; j = j + j_step)
       {
           auto Lat = fixed_lat;
           auto Lon = j;
           SphericalCoord latlon_r = vec2(Lat * (M_PI / 180.0f), Lon * (M_PI / 180.0f));
           CartesianCoord current_position;
           GeoConverter::convertRadianLatLonToXYZ(latlon_r, current_position);
           int cell_id_value = -1;
           mpasoF->mGrid->searchKDT(current_position, cell_id_value);
           int global_id = i * width + j;
           cell_id_vec[global_id] = cell_id_value;
       }
   }

   Debug("MPASOVisualizer::Finished KD Tree Search....");


    auto double_nan = std::numeric_limits<double>::quiet_NaN();
    vec3 vec3_nan = { double_nan, double_nan, double_nan };

    float i_step = (maxDepth - minDepth) / height;
    float j_step = (maxLon - minLon) / width;
    for (float i = minDepth; i < maxDepth; i = i + i_step)
    {
        for (float j = minLon; j < maxLon; j = j + j_step)
        {
            // 0. convert Lat, j to xyz
            auto Lon = j;
            auto Lat = fixed_lat;
            vec2 latlon_r = vec2(Lat * (M_PI / 180.0f), Lon * (M_PI / 180.0f));
            vec3 position;
            GeoConverter::convertRadianLatLonToXYZ(latlon_r, position);

            auto r = 6371.01 * 1000.0;
            auto tmp_r = i;
            auto current_r = r - tmp_r;
            double DEPTH = -tmp_r;

            // 1. 判断在哪个cell
            int cell_id = -1;
            mpasoF->calcInWhichCells(position, cell_id);

            // 2. 判断是否在大陆上
            std::vector<size_t> current_cell_vertices_idx;
            bool is_land = mpasoF->isOnOcean(position, cell_id, current_cell_vertices_idx);
            if (is_land)
            {
                int height_idx = (i - minDepth) / i_step;
                int width_idx = (j - minLon) / j_step;
                img->setPixel(height_idx, width_idx, vec3_nan);
                is_land = false; // 重置为默认值以供下一个像素点使用
                continue;
            }

            // 这个点在Cell上
            auto current_cell_vertices_number = mpasoF->mGrid->numberVertexOnCell_vec[cell_id];
            vec3 current_cell_vertex_pos[8];
            double current_cell_vertex_weight[8];
            std::vector<double> current_point_ztop_vec; current_point_ztop_vec.resize(80);
            for (auto v_idx = 0; v_idx < current_cell_vertices_number; v_idx++)
            {
                auto VID = current_cell_vertices_idx[v_idx];
                vec3 pos = mpasoF->mGrid->vertexCoord_vec[VID];
                current_cell_vertex_pos[v_idx] = pos;
            }
            // washpress
            Interpolator::CalcPolygonWachspress(position, current_cell_vertex_pos, current_cell_vertex_weight, current_cell_vertices_number);

            for (auto k = 0; k < 80; ++k)
            {
                double current_point_ztop_in_layer = 0.0;
                // 获取每个顶点的ztop
                double current_cell_vertex_ztop[8];
                for (auto v_idx = 0; v_idx < current_cell_vertices_number; v_idx++)
                {
                    auto VID = current_cell_vertices_idx[v_idx];
                    double ztop = mpasoF->mSol_Front->cellVertexZTop_vec[VID * 80 + k];
                    current_cell_vertex_ztop[v_idx] = ztop;
                }

                // 计算当前点的ZTOP
                for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
                {
                    current_point_ztop_in_layer += current_cell_vertex_weight[v_idx] * current_cell_vertex_ztop[v_idx];
                }
                current_point_ztop_vec[k] = current_point_ztop_in_layer;
            }

            int layer = -1;
            const double EPSILON = 1e-6;  // 设定一个小的容忍度

            for (size_t k = 1; k < current_point_ztop_vec.size(); ++k)
            {
                if (DEPTH <= current_point_ztop_vec[current_point_ztop_vec.size() - 1] - EPSILON)
                {
                    layer = -1;
                    break;
                }

                if (DEPTH <= current_point_ztop_vec[k - 1] - EPSILON && DEPTH >= current_point_ztop_vec[k] + EPSILON)
                {
                    layer = k;
                    break;
                }
            }

            if (layer == -1)
            {
                if (DEPTH < current_point_ztop_vec[current_point_ztop_vec.size() - 1] + EPSILON)
                {
                    int height_idx = (i - minDepth) / i_step;
                    int width_idx = (j - minLon) / j_step;
                    img->setPixel(height_idx, width_idx, vec3_nan);
                    continue;
                }
                else
                {
                    int height_idx = (i - minDepth) / i_step;
                    int width_idx = (j - minLon) / j_step;
                    img->setPixel(height_idx, width_idx, vec3_nan);
                    continue;
                }
            }

            double ztop_layer1, ztop_layer2;
            ztop_layer1 = current_point_ztop_vec[layer];
            ztop_layer2 = current_point_ztop_vec[layer - 1];
            double t = (sycl::fabs(DEPTH) - sycl::fabs(ztop_layer1)) / (sycl::fabs(ztop_layer2) - sycl::fabs(ztop_layer1));
            double current_point_ztop;
            current_point_ztop = t * ztop_layer1 + (1 - t) * ztop_layer2;
            //imgValue = { current_point_ztop , current_point_ztop , current_point_ztop };
            vec3 final_vel;
            // 算出 这两个layer的速度
            vec3 vertex_vel1[8];
            vec3 vertex_vel2[8];
            vec3 current_point_vel1 = { 0.0, 0.0, 0.0 };
            vec3 current_point_vel2 = { 0.0, 0.0, 0.0 };
            for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
            {
                auto VID = current_cell_vertices_idx[v_idx];
                vec3 vel1 = mpasoF->mSol_Front->cellVertexVelocity_vec[VID * 80 + layer];
                vec3 vel2 = mpasoF->mSol_Front->cellVertexVelocity_vec[VID * 80 + layer - 1];
                vertex_vel1[v_idx] = vel1;
                vertex_vel2[v_idx] = vel2;
            }
            for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
            {
                current_point_vel1.x() += current_cell_vertex_weight[v_idx] * vertex_vel1[v_idx].x(); // layer
                current_point_vel1.y() += current_cell_vertex_weight[v_idx] * vertex_vel1[v_idx].y();
                current_point_vel1.z() += current_cell_vertex_weight[v_idx] * vertex_vel1[v_idx].z();

                current_point_vel2.x() += current_cell_vertex_weight[v_idx] * vertex_vel2[v_idx].x(); //layer - 1
                current_point_vel2.y() += current_cell_vertex_weight[v_idx] * vertex_vel2[v_idx].y();
                current_point_vel2.z() += current_cell_vertex_weight[v_idx] * vertex_vel2[v_idx].z();
            }
            final_vel.x() = t * current_point_vel2.x() + (1 - t) * current_point_vel1.x();
            final_vel.y() = t * current_point_vel2.y() + (1 - t) * current_point_vel1.y();
            final_vel.z() = t * current_point_vel2.z() + (1 - t) * current_point_vel1.z();

            vec3 imgValue = final_vel;
            double zional_velocity, merminoal_velicity;
            GeoConverter::convertXYZVelocityToENU(position, final_vel, zional_velocity, merminoal_velicity);
            vec3 current_point_velocity_enu = { zional_velocity, merminoal_velicity, 0.0 };
            imgValue = current_point_velocity_enu;
            


            img->setPixel(i, j, current_point_velocity_enu);
        }
    }

    double latSpacing = (config->DepthRange.y() - config->DepthRange.x()) / (height - 1);
    //double lonSpacing = (config->LonRange.y() - config->LonRange.x()) / (width - 1);
    double lonSpacing = 1000.0 / (width - 1); // 调整后的范围

    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
    imageData->SetDimensions(width, height, 1);
    imageData->AllocateScalars(VTK_DOUBLE, 3);
    //imageData->SetOrigin(config->LonRange.x(), config->LatRange.x(), config->FixedLatitude);  // 设置数据的起始位置
    //imageData->SetSpacing(lonSpacing, latSpacing, config->FixedLatitude);  // 设置每个像素的物理尺寸

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            auto pixel = img->getPixel(i, j);
            double* pixelData = static_cast<double*>(imageData->GetScalarPointer(j, i, 0)); 
            pixelData[0] = pixel.x();
            pixelData[1] = pixel.y();
            pixelData[2] = pixel.z();
        }
    }

    

    vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
    writer->SetFileName("ouput.vti");
    writer->SetInputData(imageData);
    writer->Write();
*/
}


void MPASOVisualizer::GenerateSamplePoint(std::vector<CartesianCoord>& points, SamplingSettings* config)
{
    auto minLat = config->getLatitudeRange().x(); auto maxLat = config->getLatitudeRange().y();
    auto minLon = config->getLongitudeRange().x(); auto maxLon = config->getLongitudeRange().y();

    double i_step = (maxLat - minLat) / static_cast<double>(config->getSampleRange().x() - 1);
    double j_step = (maxLon - minLon) / static_cast<double>(config->getSampleRange().y() - 1);

    for (double i = minLat ; i < maxLat; i += i_step)
    {
        for (double j = minLon; j < maxLon; j += j_step)
        {
            CartesianCoord p = { j, i, config->getDepth() };
            points.push_back(p);
        }
    }


    Debug("Generate %d sample points in [ %f, %f ] -> [ %f, %f ]", points.size(), minLat, minLon, maxLat, maxLon);

    for (auto i = 0; i < points.size(); i++)
    {
        vec3 get_points = points[i];
        SphericalCoord latlon_d; SphericalCoord latlon_r; CartesianCoord position;
        latlon_d.x() = get_points.y(); latlon_d.y() = get_points.x();
        GeoConverter::convertDegreeToRadian(latlon_d, latlon_r);
        GeoConverter::convertRadianLatLonToXYZ(latlon_r, position);
        points[i].x() = position.x(); points[i].y() = position.y(); points[i].z() = position.z();
    }
}

void MPASOVisualizer::GenerateSamplePointAtCenter(std::vector<CartesianCoord>& points, SamplingSettings* config)
{
    if (config->isAtCellCenter() == false) return;
    Debug("Generate %d sample points At Cell Center]", points.size());

}


//TODO: Temporarily unavailable, it will be released later.
[[deprecated]]
void MPASOVisualizer::GenerateGaussianSpherePoints(std::vector<CartesianCoord>& points, SamplingSettings* config, int numPoints, double meanLat, double meanLon, double stdDev)
{
    auto minLat = config->getLatitudeRange().x(); auto maxLat = config->getLatitudeRange().y();
    auto minLon = config->getLongitudeRange().x(); auto maxLon = config->getLongitudeRange().y();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> latDist(meanLat, stdDev); // 纬度高斯分布
    std::normal_distribution<double> lonDist(meanLon, stdDev); // 经度高斯分布

    for (int i = 0; i < numPoints; ++i) {
        double lat, lon;

        // 生成符合范围的纬度和经度
        do {
            lat = latDist(gen);
        } while (lat < minLat || lat > maxLat);

        do {
            lon = lonDist(gen);
        } while (lon < minLon || lon > maxLon);

        // 转换为笛卡尔坐标
        SphericalCoord latlon_d = { lat, lon };
        SphericalCoord latlon_r;
        CartesianCoord position;
        GeoConverter::convertDegreeToRadian(latlon_d, latlon_r);
        GeoConverter::convertRadianLatLonToXYZ(latlon_r, position);

        points.push_back(position);
    }
    Debug("Generate %d sample points in [ %f, %f ] -> [ %f ]", points.size(), meanLat, meanLon, stdDev);
}


[[deprecated]]
vec3 computeRotationAxis(const vec3& position, const vec3& velocity)
{
    vec3 axis;
    axis.x() = position.y() * velocity.z() - position.z() * velocity.y();
    axis.y() = position.z() * velocity.x() - position.x() * velocity.z();
    axis.z() = position.x() * velocity.y() - position.y() * velocity.x();
    return axis;
}
[[deprecated]]
double magnitude(const vec3& v)
{
    return sycl::sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
}
[[deprecated]]
vec3 normalize(const vec3& v)
{
    double length = sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
    vec3 normalized = { v.x() / length, v.y() / length, v.z() / length };
    return normalized;
}
[[deprecated]]
void rotateAroundAxis(const vec3& point, const vec3& axis, double theta_rad, double& x, double& y, double& z)
{
    // 传入的是弧度
    double thetaRad = theta_rad;
    double cosTheta = sycl::cos(thetaRad);
    double sinTheta = sycl::sin(thetaRad);
    vec3 u = normalize(axis);

    vec3 rotated;
    rotated.x() = (cosTheta + u.x() * u.x() * (1.0 - cosTheta)) * point.x() +
        (u.x() * u.y() * (1.0 - cosTheta) - u.z() * sinTheta) * point.y() +
        (u.x() * u.z() * (1.0 - cosTheta) + u.y() * sinTheta) * point.z();

    rotated.y() = (u.y() * u.x() * (1.0 - cosTheta) + u.z() * sinTheta) * point.x() +
        (cosTheta + u.y() * u.y() * (1.0 - cosTheta)) * point.y() +
        (u.y() * u.z() * (1.0 - cosTheta) - u.x() * sinTheta) * point.z();

    rotated.z() = (u.z() * u.x() * (1.0 - cosTheta) - u.y() * sinTheta) * point.x() +
        (u.z() * u.y() * (1.0 - cosTheta) + u.x() * sinTheta) * point.y() +
        (cosTheta + u.z() * u.z() * (1.0 - cosTheta)) * point.z();

    x = rotated.x();
    y = rotated.y();
    z = rotated.z();
}


[[deprecated]]
bool isClose(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) < eps;
}
[[deprecated]]
bool isCloseVec3(const CartesianCoord& a, const CartesianCoord& b, double eps = 1e-6) {
    return isClose(a.x(), b.x(), eps) && isClose(a.y(), b.y(), eps) && isClose(a.z(), b.z(), eps);
}
[[deprecated]]
bool compareTrajectoryLines(const std::vector<TrajectoryLine>& a, const std::vector<TrajectoryLine>& b) {
    if (a.size() != b.size()) {
        std::cout << "[Compare] Line vector size mismatch: " << a.size() << " vs " << b.size() << std::endl;
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        const auto& la = a[i];
        const auto& lb = b[i];

        if (la.lineID != lb.lineID) {
            std::cout << "[Compare] LineID mismatch at index " << i << ": " << la.lineID << " vs " << lb.lineID << std::endl;
            return false;
        }

        if (la.points.size() != lb.points.size()) {
            std::cout << "[Compare] Points size mismatch at line " << i << ": " << la.points.size() << " vs " << lb.points.size() << std::endl;
            return false;
        }

        for (size_t j = 0; j < la.points.size(); ++j) {
            if (!isCloseVec3(la.points[j], lb.points[j])) {
                std::cout << "[Compare] Point mismatch at line " << i << ", point " << j << std::endl;
                return false;
            }
        }

        if (!isCloseVec3(la.lastPoint, lb.lastPoint)) {
            std::cout << "[Compare] LastPoint mismatch at line " << i << std::endl;
            return false;
        }

        if (!isClose(la.duration, lb.duration)) {
            std::cout << "[Compare] Duration mismatch at line " << i << ": " << la.duration << " vs " << lb.duration << std::endl;
            return false;
        }

        if (!isClose(la.timestamp, lb.timestamp)) {
            std::cout << "[Compare] Timestamp mismatch at line " << i << ": " << la.timestamp << " vs " << lb.timestamp << std::endl;
            return false;
        }
    }

    std::cout << "[Compare] All trajectory lines are identical!" << std::endl;
    return true;
}

std::vector<TrajectoryLine> MPASOVisualizer::StreamLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, sycl::queue& sycl_Q)
{
    
    std::vector<vec3> stable_points = points; 

    std::vector<vec3> update_points;
    std::vector<vec3> update_vels;
    if (!update_points.empty()) update_points.clear();
    update_points.resize(stable_points.size() * (config->simulationDuration / config->recordT));
    if (!update_vels.empty()) update_vels.clear();
    update_vels.resize(stable_points.size() * (config->simulationDuration / config->recordT));

    std::cout << "points.size = " << stable_points.size() 
          << ", default_cell_id.size = " << default_cell_id.size() << std::endl;

    std::vector<TrajectoryLine> trajectory_lines;
    trajectory_lines.resize(stable_points.size());
    tbb::parallel_for(size_t(0), stable_points.size(), [&](size_t i) {
        trajectory_lines[i].lineID = i;
        trajectory_lines[i].points.push_back(stable_points[i]);
        trajectory_lines[i].lastPoint = stable_points[i];
        trajectory_lines[i].duration = config->simulationDuration;
        trajectory_lines[i].timestamp = config->deltaT;
        trajectory_lines[i].depth = config->depth;
    });

   
    std::vector<size_t> grid_info_vec;
    // tbl
    // 0 : mCellsSize
    // 1 : mEdgesSize
    // 2 : mMaxEdgesSize
    // 3 : mVertexSize
    // 4 : mVertLevels
    // 5 : mVertLevelsP1
    grid_info_vec.push_back(mpasoF->mGrid->mCellsSize);
    grid_info_vec.push_back(mpasoF->mGrid->mEdgesSize);
    grid_info_vec.push_back(mpasoF->mGrid->mMaxEdgesSize);
    grid_info_vec.push_back(mpasoF->mGrid->mVertexSize);
    grid_info_vec.push_back(mpasoF->mGrid->mVertLevels);
    // grid_info_vec.push_back(mpasoF->mGrid->mVertLevelsP1);

#pragma region sycl_buffer_grid
    sycl::buffer<vec3, 1> vertexCoord_buf(mpasoF->mGrid->vertexCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->vertexCoord_vec.size())); // CELL 顶点坐标
    sycl::buffer<vec3, 1> cellCoord_buf(mpasoF->mGrid->cellCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->cellCoord_vec.size()));       // CELL 中心坐标
    sycl::buffer<size_t, 1> numberVertexOnCell_buf(mpasoF->mGrid->numberVertexOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->numberVertexOnCell_vec.size())); // CELL 有几个顶点
    sycl::buffer<size_t, 1> verticesOnCell_buf(mpasoF->mGrid->verticesOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->verticesOnCell_vec.size()));             // 
    sycl::buffer<size_t, 1> cellsOnVertex_buf(mpasoF->mGrid->cellsOnVertex_vec.data(), sycl::range<1>(mpasoF->mGrid->cellsOnVertex_vec.size()));
    sycl::buffer<size_t, 1> cells_onCell_buf(mpasoF->mGrid->cellsOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->cellsOnCell_vec.size()));
    sycl::buffer<size_t, 1> grid_info_buf(grid_info_vec.data(), sycl::range<1>(grid_info_vec.size()));
#pragma endregion   sycl_buffer_grid


#pragma region sycl_buffer_velocity
    sycl::buffer<vec3, 1> cellVertexVelocity_buf(mpasoF->mSol_Front->cellVertexVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexVelocity_vec.size()));
    sycl::buffer<double, 1> cellVertexZTop_buf(mpasoF->mSol_Front->cellVertexZTop_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexZTop_vec.size()));
#pragma endregion sycl_buffer_velocity   

    sycl::buffer<int, 1> cellID_buf(default_cell_id.data(), sycl::range<1>(default_cell_id.size()));
    sycl::buffer<vec3> wirte_points_buf(update_points.data(), sycl::range<1>(update_points.size()));
    sycl::buffer<vec3> write_vels_buf(update_vels.data(), sycl::range<1>(update_vels.size()));
    sycl::buffer<vec3> sample_points_buf(stable_points.data(), sycl::range<1>(stable_points.size()));
    sycl_Q.submit([&](sycl::handler& cgh) 
    {

#pragma region sycl_acc_grid
        auto acc_cellID_buf = cellID_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_vertexCoord_buf = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnCell_buf = verticesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnVertex_buf = cellsOnVertex_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cells_onCell_buf = cells_onCell_buf.get_access<sycl::access::mode::read>(cgh);
        int grid_cell_size = mpasoF->mGrid->mCellsSize;
        auto acc_grid_info_buf = grid_info_buf.get_access<sycl::access::mode::read>(cgh);
#pragma endregion sycl_acc_grid

#pragma region sycl_acc_velocity
        auto acc_cellVertexVelocity_buf = cellVertexVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexZTop_buf = cellVertexZTop_buf.get_access<sycl::access::mode::read>(cgh);
#pragma endregion sycl_acc_velocity

        auto acc_def_cell_id_buf = cellID_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_wirte_points_buf = wirte_points_buf.get_access<sycl::access::mode::write>(cgh);
        auto acc_write_vels_buf = write_vels_buf.get_access<sycl::access::mode::write>(cgh);
        auto acc_sample_points_buf = sample_points_buf.get_access<sycl::access::mode::read_write>(cgh);

        sycl::stream out(2048, 256, cgh);
        int times = config->simulationDuration / config->deltaT;
        int each_points_size = config->simulationDuration / config->recordT;
        int recordT = config->recordT;
        int deltaT = config->deltaT;
        float config_depth = config->depth;

        
       
        cgh.parallel_for(sycl::range<1>(points.size()), [=](sycl::item<1> item) 
        {
            int global_id = item[0];
            // load from grid info
            const int ACTUALL_CELL_SIZE         = (int)acc_grid_info_buf[0];
            const int ACTUALL_EDGE_SIZE         = (int)acc_grid_info_buf[1];
            const int ACTUALL_MAX_EDGE_SIZE     = (int)acc_grid_info_buf[2];
            const int ACTUALL_VERTEX_SIZE       = (int)acc_grid_info_buf[3];
            const int ACTUALL_ZTOP_LAYER        = (int)acc_grid_info_buf[4];
            // default
            const int MAX_VERTEX_NUM            = 20;
            const int MAX_CELL_NEIGHBOR_NUM     = 21;
            const int MAX_VERTLEVELS            = 80;
            const int NEIGHBOR_NUM              = 3;
            // const int TOTAY_ZTOP_LAYER          = 60;
            // const int VERTLEVELS                = 60;
            double fixed_depth                  = -1.0f * config_depth;

            double runTime = 0.0;
            int save_times = 0;
            bool bFirstLoop = true;
            bool bFirstVel = true;
            int cell_id_vec[MAX_VERTEX_NUM];
            int base_idx = global_id * each_points_size;
            int update_points_idx = 0;

            // 1. 获取point position
            vec3 position; 
            int cell_id = -1;
            vec3 new_position;
            double pos_x, pos_y, pos_z;
            int cell_neig_vec[MAX_CELL_NEIGHBOR_NUM];


            auto CalcVelocityAt = [&](const vec3& pos, int& cid) -> vec3 {
                // 判断是否在大陆上
                vec3 position = pos;
                int cell_id = cid;
                vec3 current_position = position;
                auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
                SYCLKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx, MAX_VERTEX_NUM, ACTUALL_MAX_EDGE_SIZE, acc_verticesOnCell_buf);
                bool is_land = SYCLKernel::IsInOcean(cell_id, ACTUALL_MAX_EDGE_SIZE, current_position, acc_numberVertexOnCell_buf, acc_verticesOnCell_buf, acc_vertexCoord_buf);
                if (!is_land)
                {
                    // out << "DEBUG: Particle " << global_id << " hit land at cell " << cell_id << sycl::endl;
                    return vec3(0.0, 0.0, 0.0);
                }

                
                // 计算当前点 的ZTOP
                double current_point_ztop_vec[MAX_VERTLEVELS];

                vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
                double current_cell_vertex_weight[MAX_VERTEX_NUM];
                bool rc = SYCLKernel::GetCellVertexPos(current_cell_vertex_pos, current_cell_vertices_idx, MAX_VERTEX_NUM, current_cell_vertices_number, acc_vertexCoord_buf);
                if (!rc)
                {
                    out << "[ERROR]:: GetCellVertexPos Failed....(VLA < current_cell_vertices_number)" << sycl::endl;
                    return vec3(0.0, 0.0, 0.0);
                }
                // washpress.
                Interpolator::CalcPolygonWachspress(current_position, current_cell_vertex_pos, current_cell_vertex_weight, current_cell_vertices_number);

                int ztop_range = acc_cellVertexZTop_buf.get_range()[0];
                int vel_range = acc_cellVertexVelocity_buf.get_range()[0];

                for (auto k = 0; k < ACTUALL_ZTOP_LAYER; ++k)
                {
                    double current_point_ztop_in_layer = 0.0;
                    // 获取每个顶点的ztop
                    double current_cell_vertex_ztop[MAX_VERTEX_NUM];
                    for (auto v_idx = 0; v_idx < current_cell_vertices_number; v_idx++)
                    {
                        auto VID = current_cell_vertices_idx[v_idx];
                        if (VID < 0 || VID >= acc_vertexCoord_buf.get_range()[0]) {
                            out << "[ERROR] Invalid VID: " << VID << sycl::endl;
                            return vec3(0.0, 0.0, 0.0);
                        }

                        double ztop = acc_cellVertexZTop_buf[VID * ACTUALL_ZTOP_LAYER + k];
                        current_cell_vertex_ztop[v_idx] = ztop;
                    }
                    // 计算当前点的ZTOP
                    for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
                    {
                        current_point_ztop_in_layer += current_cell_vertex_weight[v_idx] * current_cell_vertex_ztop[v_idx];
                    }
                    current_point_ztop_vec[k] = current_point_ztop_in_layer;
                }

                const double EPSILON = 1e-6;
                int local_layer = 0;
                for (auto k = 1; k < ACTUALL_ZTOP_LAYER; ++k)
                {
                    if (fixed_depth <= current_point_ztop_vec[k - 1] - EPSILON && fixed_depth >= current_point_ztop_vec[k] + EPSILON)
                    {
                        local_layer = k;
                        break;
                    }
                }
            
                
                if (local_layer == 0)
                {
                    out << "DEBUG: Particle " << global_id << " depth layer not found. " 
                    << "fixed_depth=" << fixed_depth 
                    << " ztop_surface=" << current_point_ztop_vec[0] 
                    << " ztop_bottom=" << current_point_ztop_vec[ACTUALL_ZTOP_LAYER-1] << sycl::endl;
                    return vec3(0.0, 0.0, 0.0);
                }
                else
                {
                    // 计算当前点的速度
                    double ztop_layer1, ztop_layer2;
                    ztop_layer1 = current_point_ztop_vec[local_layer];
                    ztop_layer2 = current_point_ztop_vec[local_layer - 1];
                    double t = (sycl::fabs(fixed_depth) - sycl::fabs(ztop_layer1)) / (sycl::fabs(ztop_layer2) - sycl::fabs(ztop_layer1));
                    double denominator = sycl::fabs(ztop_layer2) - sycl::fabs(ztop_layer1);
                    if (sycl::fabs(denominator) < 1e-12) {
                        out << "DEBUG: Particle " << global_id << " has zero depth difference: " << denominator << sycl::endl;
                        return vec3(0.0, 0.0, 0.0);
                    }
                    // double current_point_ztop;
                    // current_point_ztop = t * ztop_layer1 + (1 - t) * ztop_layer2;
                    
                    vec3 final_vel;
                    // 算出 这两个layer的速度
                    // vec3 vertex_vel1[MAX_VERTEX_NUM];
                    // vec3 vertex_vel2[MAX_VERTEX_NUM];
                    vec3 current_point_vel1 = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer, acc_cellVertexVelocity_buf);
                
                    vec3 current_point_vel2 = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer - 1, acc_cellVertexVelocity_buf);
                    
                    double vel1_mag = YOSEF_LENGTH(current_point_vel1);
                    double vel2_mag = YOSEF_LENGTH(current_point_vel2);
                        
                    if (vel1_mag < 1e-12 || vel2_mag < 1e-12) {
                            out << "DEBUG: Particle " << global_id << " layer velocities are zero" << sycl::endl;
                            return vec3(0.0, 0.0, 0.0);
                    }

                    final_vel.x() = t * current_point_vel2.x() + (1 - t) * current_point_vel1.x();
                    final_vel.y() = t * current_point_vel2.y() + (1 - t) * current_point_vel1.y();
                    final_vel.z() = t * current_point_vel2.z() + (1 - t) * current_point_vel1.z();

                    vec3 current_velocity = final_vel;

                    double vel_mag = YOSEF_LENGTH(current_velocity);
                    if (vel_mag < 1e-12) {
                        out << "DEBUG: Particle " << global_id << " has zero velocity: " << vel_mag << sycl::endl;
                        return vec3(0.0, 0.0, 0.0);
                    }
                    return current_velocity;
                }
                
            };


            if (bFirstLoop == false) {
                if (cell_id < 0 || cell_id >= acc_numberVertexOnCell_buf.get_range()[0]) {
                    out << "[Error] cell_id out of range: " << cell_id << sycl::endl;
                    return;
                }
            }


            for (auto times_i = 0; times_i < times; times_i++)
            {
                
                runTime += deltaT;
                position = acc_sample_points_buf[global_id];
                int firtst_cell_id = acc_def_cell_id_buf[global_id];
                // 第一次循环
                if (bFirstLoop)
                {
                    bFirstLoop = false;
                    cell_id = firtst_cell_id;
                    
                    // 找到这个CELL有多少个点
                    auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                    SYCLKernel::GetCellNeighborsIdx(cell_id, current_cell_vertices_number, cell_neig_vec, MAX_VERTEX_NUM, ACTUALL_MAX_EDGE_SIZE, acc_cells_onCell_buf);
                    
                    int acc_wirte_pints_idx = base_idx + 0;
                    acc_wirte_points_buf[acc_wirte_pints_idx] = position;
                }
                else
                {
                    // 找到这个CELL有多少个点
                    auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                    double max_len = std::numeric_limits<double>::max();
                    for (auto idx = 0; idx < current_cell_vertices_number + 1; idx++)
                    {
                        // 判断当前点 POS 在 哪个CELL 中  -> get new_cell_id
                        auto CID = cell_neig_vec[idx];
                        if (CID < 0 || CID >= acc_cellCoord_buf.get_range()[0]) continue;
                        vec3 pos = acc_cellCoord_buf[CID];
                        double len = YOSEF_LENGTH(pos - position);
                        if (len < max_len)
                        {
                            max_len = len;
                            cell_id = CID;
                        }
                    }
                    SYCLKernel::GetCellNeighborsIdx(cell_id, current_cell_vertices_number, cell_neig_vec, MAX_VERTEX_NUM, ACTUALL_MAX_EDGE_SIZE, acc_cells_onCell_buf);
                   
                }

                // TEST
                vec3 current_position = position;
                /*
                // 判断是否在大陆上
                vec3 current_position = position;
                auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
                SYCLKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx, MAX_VERTEX_NUM, ACTUALL_MAX_EDGE_SIZE, acc_verticesOnCell_buf);
                bool is_land = SYCLKernel::IsInOcean(cell_id, ACTUALL_MAX_EDGE_SIZE, current_position, acc_numberVertexOnCell_buf, acc_verticesOnCell_buf, acc_vertexCoord_buf);
                if (is_land)
                {
                    continue;
                }

                
                // 计算当前点 的ZTOP
                double current_point_ztop_vec[MAX_VERTLEVELS];

                vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
                double current_cell_vertex_weight[MAX_VERTEX_NUM];
                bool rc = SYCLKernel::GetCellVertexPos(current_cell_vertex_pos, current_cell_vertices_idx, MAX_VERTEX_NUM, current_cell_vertices_number, acc_vertexCoord_buf);
                if (!rc)
                {
                    out << "[ERROR]:: GetCellVertexPos Failed....(VLA < current_cell_vertices_number)" << sycl::endl;
                    return;
                }
                // washpress
                Interpolator::CalcPolygonWachspress(current_position, current_cell_vertex_pos, current_cell_vertex_weight, current_cell_vertices_number);

                int ztop_range = acc_cellVertexZTop_buf.get_range()[0];
                int vel_range = acc_cellVertexVelocity_buf.get_range()[0];

                for (auto k = 0; k < ACTUALL_ZTOP_LAYER; ++k)
                {
                    double current_point_ztop_in_layer = 0.0;
                    // 获取每个顶点的ztop
                    double current_cell_vertex_ztop[MAX_VERTEX_NUM];
                    for (auto v_idx = 0; v_idx < current_cell_vertices_number; v_idx++)
                    {
                        auto VID = current_cell_vertices_idx[v_idx];
                        if (VID < 0 || VID >= acc_vertexCoord_buf.get_range()[0]) {
                            out << "[ERROR] Invalid VID: " << VID << sycl::endl;
                            return;
                        }

                        double ztop = acc_cellVertexZTop_buf[VID * ACTUALL_ZTOP_LAYER + k];
                        current_cell_vertex_ztop[v_idx] = ztop;
                    }
                    // 计算当前点的ZTOP
                    for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
                    {
                        current_point_ztop_in_layer += current_cell_vertex_weight[v_idx] * current_cell_vertex_ztop[v_idx];
                    }
                    current_point_ztop_vec[k] = current_point_ztop_in_layer;
                }

                const double EPSILON = 1e-6;
                int local_layer = 0;
                for (auto k = 1; k < ACTUALL_ZTOP_LAYER; ++k)
                {
                    if (fixed_depth <= current_point_ztop_vec[k - 1] - EPSILON && fixed_depth >= current_point_ztop_vec[k] + EPSILON)
                    {
                        local_layer = k;
                        break;
                    }
                }
            
               
                if (local_layer == 0)
                {
                    continue;
                }
                else
                {
                    // 计算当前点的速度
                    double ztop_layer1, ztop_layer2;
                    ztop_layer1 = current_point_ztop_vec[local_layer];
                    ztop_layer2 = current_point_ztop_vec[local_layer - 1];
                    double t = (sycl::fabs(fixed_depth) - sycl::fabs(ztop_layer1)) / (sycl::fabs(ztop_layer2) - sycl::fabs(ztop_layer1));
                    // double current_point_ztop;
                    // current_point_ztop = t * ztop_layer1 + (1 - t) * ztop_layer2;
                    
                    vec3 final_vel;
                    // 算出 这两个layer的速度
                    // vec3 vertex_vel1[MAX_VERTEX_NUM];
                    // vec3 vertex_vel2[MAX_VERTEX_NUM];
                    vec3 current_point_vel1 = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer, acc_cellVertexVelocity_buf);
                
                    vec3 current_point_vel2 = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer - 1, acc_cellVertexVelocity_buf);
                    
                    final_vel.x() = t * current_point_vel2.x() + (1 - t) * current_point_vel1.x();
                    final_vel.y() = t * current_point_vel2.y() + (1 - t) * current_point_vel1.y();
                    final_vel.z() = t * current_point_vel2.z() + (1 - t) * current_point_vel1.z();

                    vec3 current_velocity = final_vel;
                }
                */

                // Euler method
                // vec3 current_velocity = CalcVelocityAt(current_position, cell_id);
                
                // Runge-Kutta 4th order method
                double r = YOSEF_LENGTH(current_position);
                double dt = static_cast<double>(deltaT);
                vec3 k1 = CalcVelocityAt(position, cell_id);
                double speed_k1 = YOSEF_LENGTH(k1);
                if (speed_k1 < 1e-12) {
                    out << "DEBUG: Particle " << global_id << " has zero velocity at time " << runTime << sycl::endl;
                    return;
                }
                // // p + 0.5 * k1 Δt   → 归一化回球面
                // vec3 p2   = sycl::normalize(position + (dt * 0.5) * k1) * r;
                // int  cid2 = cell_id;
                // vec3 k2   = CalcVelocityAt(p2, cid2);
                // // p + 0.5 * k2 Δt
                // vec3 p3   = sycl::normalize(position + (dt * 0.5) * k2) * r;
                // int  cid3 = cell_id;
                // vec3 k3   = CalcVelocityAt(p3, cid3);
                // // p + 1.0 * k3 Δt
                // vec3 p4   = sycl::normalize(position + dt * k3) * r;
                // int  cid4 = cell_id;
                // vec3 k4   = CalcVelocityAt(p4, cid4);
                // // 组合平均速度
                // vec3 current_velocity = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

                vec3 current_velocity = k1;


                
                vec3 rotationAxis = SYCLKernel::CalcRotationAxis(current_position, current_velocity);
                double speed = YOSEF_LENGTH(current_velocity);
                double theta_rad = (speed * deltaT) / r;
                new_position = SYCLKernel::CalcPositionAfterRotation(current_position, rotationAxis, theta_rad);
                   
                if (bFirstVel)
                {
                     bFirstVel = false;
                     int acc_wirte_pints_idx = base_idx + 0;
                     acc_write_vels_buf[acc_wirte_pints_idx] = current_velocity;
                }

                acc_sample_points_buf[global_id] = new_position;
                    
                if ((int)runTime % recordT == 0)
                {
                    // //save .//TODO
                    int acc_wirte_pints_idx = base_idx + update_points_idx;
                    acc_wirte_points_buf[acc_wirte_pints_idx] = new_position;
                    acc_write_vels_buf[acc_wirte_pints_idx] = current_velocity;
                    update_points_idx = update_points_idx + 1;
                }
                    
                //} 
            }
        });
        
       
    });
    try {
        sycl_Q.wait();
    } catch (sycl::exception const& e) {
        std::cerr << "Caught SYCL exception: " << e.what() << std::endl;
        std::exit(1);
    }

    Debug("[VisualizeTrajectory]::Finished...");
    // auto after_write_p = wirte_points_buf.get_access<sycl::access::mode::read>();
    auto after_write_p = wirte_points_buf.get_host_access(sycl::read_only); // XYZ
    auto after_write_v = write_vels_buf.get_host_access(sycl::read_only); // XYZ
    std::vector<CartesianCoord> last_points;
    

    // update trajectory_lines
    size_t line_idx = 0;
    size_t each_point_size = config->simulationDuration / config->recordT;
    size_t total_points = update_points.size();
    size_t total_lines = trajectory_lines.size();

    tbb::parallel_for(size_t(0), total_lines, [&](size_t line_idx) {
        size_t start_idx = line_idx * each_point_size;
        size_t end_idx = std::min(start_idx + each_point_size, total_points);

        for (size_t i = start_idx; i < end_idx; ++i) {
            vec3 p = after_write_p[i];
            vec3 v = after_write_v[i];
            trajectory_lines[line_idx].points.push_back(p);
            trajectory_lines[line_idx].velocity.push_back(v);

            if (i == end_idx - 1 || i == total_points - 1) {
                trajectory_lines[line_idx].lastPoint = p;
            }
        }
    });

    return trajectory_lines;
}

std::vector<TrajectoryLine> MPASOVisualizer::PathLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, sycl::queue& sycl_Q)
{
    
    std::vector<vec3> stable_points = points; 

    std::vector<vec3> update_points;
    if (!update_points.empty()) update_points.clear();
    update_points.resize(stable_points.size() * (config->simulationDuration / config->recordT));

    std::cout << "points.size = " << stable_points.size() 
          << ", default_cell_id.size = " << default_cell_id.size() << std::endl;

    std::vector<TrajectoryLine> trajectory_lines;
    trajectory_lines.resize(stable_points.size());
    tbb::parallel_for(size_t(0), stable_points.size(), [&](size_t i) {
        trajectory_lines[i].lineID = i;
        trajectory_lines[i].points.push_back(stable_points[i]);
        trajectory_lines[i].lastPoint = stable_points[i];
        trajectory_lines[i].duration = config->simulationDuration;
        trajectory_lines[i].timestamp = config->deltaT;
        trajectory_lines[i].depth = config->depth;
    });

   
    std::vector<size_t> grid_info_vec;
    // tbl
    // 0 : mCellsSize
    // 1 : mEdgesSize
    // 2 : mMaxEdgesSize
    // 3 : mVertexSize
    // 4 : mVertLevels
    // 5 : mVertLevelsP1
    grid_info_vec.push_back(mpasoF->mGrid->mCellsSize);
    grid_info_vec.push_back(mpasoF->mGrid->mEdgesSize);
    grid_info_vec.push_back(mpasoF->mGrid->mMaxEdgesSize);
    grid_info_vec.push_back(mpasoF->mGrid->mVertexSize);
    grid_info_vec.push_back(mpasoF->mGrid->mVertLevels);
    // grid_info_vec.push_back(mpasoF->mGrid->mVertLevelsP1);

#pragma region sycl_buffer_grid
    sycl::buffer<vec3, 1> vertexCoord_buf(mpasoF->mGrid->vertexCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->vertexCoord_vec.size())); // CELL 顶点坐标
    sycl::buffer<vec3, 1> cellCoord_buf(mpasoF->mGrid->cellCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->cellCoord_vec.size()));       // CELL 中心坐标
    sycl::buffer<size_t, 1> numberVertexOnCell_buf(mpasoF->mGrid->numberVertexOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->numberVertexOnCell_vec.size())); // CELL 有几个顶点
    sycl::buffer<size_t, 1> verticesOnCell_buf(mpasoF->mGrid->verticesOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->verticesOnCell_vec.size()));             // 
    sycl::buffer<size_t, 1> cellsOnVertex_buf(mpasoF->mGrid->cellsOnVertex_vec.data(), sycl::range<1>(mpasoF->mGrid->cellsOnVertex_vec.size()));
    sycl::buffer<size_t, 1> cells_onCell_buf(mpasoF->mGrid->cellsOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->cellsOnCell_vec.size()));
    sycl::buffer<size_t, 1> grid_info_buf(grid_info_vec.data(), sycl::range<1>(grid_info_vec.size()));
#pragma endregion   sycl_buffer_grid


#pragma region sycl_buffer_velocity
    sycl::buffer<vec3, 1> cellVertexVelocity_front_buf(mpasoF->mSol_Front->cellVertexVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexVelocity_vec.size()));
    sycl::buffer<double, 1> cellVertexZTop_front_buf(mpasoF->mSol_Front->cellVertexZTop_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexZTop_vec.size()));
    sycl::buffer<vec3, 1> cellVertexVelocity_back_buf(mpasoF->mSol_Back->cellVertexVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Back->cellVertexVelocity_vec.size()));
    sycl::buffer<double, 1> cellVertexZTop_back_buf(mpasoF->mSol_Back->cellVertexZTop_vec.data(), sycl::range<1>(mpasoF->mSol_Back->cellVertexZTop_vec.size()));
#pragma endregion sycl_buffer_velocity   

    sycl::buffer<int, 1> cellID_buf(default_cell_id.data(), sycl::range<1>(default_cell_id.size()));
    sycl::buffer<vec3> wirte_points_buf(update_points.data(), sycl::range<1>(update_points.size()));
    sycl::buffer<vec3> sample_points_buf(stable_points.data(), sycl::range<1>(stable_points.size()));
    sycl_Q.submit([&](sycl::handler& cgh) 
    {

#pragma region sycl_acc_grid
        auto acc_cellID_buf = cellID_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_vertexCoord_buf = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnCell_buf = verticesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnVertex_buf = cellsOnVertex_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cells_onCell_buf = cells_onCell_buf.get_access<sycl::access::mode::read>(cgh);
        int grid_cell_size = mpasoF->mGrid->mCellsSize;
        auto acc_grid_info_buf = grid_info_buf.get_access<sycl::access::mode::read>(cgh);
#pragma endregion sycl_acc_grid

#pragma region sycl_acc_velocity
        auto acc_cellVertexVelocity_front_buf = cellVertexVelocity_front_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexZTop_front_buf = cellVertexZTop_front_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexVelocity_back_buf = cellVertexVelocity_back_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexZTop_back_buf = cellVertexZTop_back_buf.get_access<sycl::access::mode::read>(cgh);
#pragma endregion sycl_acc_velocity

        auto acc_def_cell_id_buf = cellID_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_wirte_points_buf = wirte_points_buf.get_access<sycl::access::mode::write>(cgh);
        auto acc_sample_points_buf = sample_points_buf.get_access<sycl::access::mode::read_write>(cgh);

        sycl::stream out(1024, 256, cgh);
        int n_steps = config->simulationDuration / config->deltaT;
        int each_points_size = config->simulationDuration / config->recordT;
        int recordT = config->recordT;
        int deltaT = config->deltaT;
        float config_depth = config->depth;

        
       
        cgh.parallel_for(sycl::range<1>(points.size()), [=](sycl::item<1> item) 
        {
            int global_id = item[0];
            // load from grid info
            const int ACTUALL_CELL_SIZE         = (int)acc_grid_info_buf[0];
            const int ACTUALL_EDGE_SIZE         = (int)acc_grid_info_buf[1];
            const int ACTUALL_MAX_EDGE_SIZE     = (int)acc_grid_info_buf[2];
            const int ACTUALL_VERTEX_SIZE       = (int)acc_grid_info_buf[3];
            const int ACTUALL_ZTOP_LAYER        = (int)acc_grid_info_buf[4];
            // default
            const int MAX_VERTEX_NUM            = 20;
            const int MAX_CELL_NEIGHBOR_NUM     = 21;
            const int MAX_VERTLEVELS            = 80;
            const int NEIGHBOR_NUM              = 3;
            // const int TOTAY_ZTOP_LAYER          = 60;
            // const int VERTLEVELS                = 60;
            double fixed_depth                  = -1.0f * config_depth;

            double runTime = 0.0;
            int save_times = 0;
            bool bFirstLoop = true;
            int cell_id_vec[MAX_VERTEX_NUM];
            int base_idx = global_id * each_points_size;
            int update_points_idx = 0;

            // 1. 获取point position
            vec3 position; 
            int cell_id = -1;
            vec3 new_position;
            double pos_x, pos_y, pos_z;
            int cell_neig_vec[MAX_CELL_NEIGHBOR_NUM];

            auto CalcVelocityAt = [&](const vec3& pos, int& cid, double alpha) -> vec3 {
                // 判断是否在大陆上
                vec3 current_position = pos;
                int cell_id = cid;
                double alpha_for_interplate = alpha;
                auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
                SYCLKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx, MAX_VERTEX_NUM, ACTUALL_MAX_EDGE_SIZE, acc_verticesOnCell_buf);
                bool is_land = SYCLKernel::IsInOcean(cell_id, ACTUALL_MAX_EDGE_SIZE, current_position, acc_numberVertexOnCell_buf, acc_verticesOnCell_buf, acc_vertexCoord_buf);
                if (!is_land)
                {
                    return vec3(0.0, 0.0, 0.0);
                }

                
                // 计算当前点 的ZtopFront, ZtopBack
                double current_point_ztop_front_vec[MAX_VERTLEVELS];
                double current_point_ztop_back_vec[MAX_VERTLEVELS];

                vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
                double current_cell_vertex_weight[MAX_VERTEX_NUM];
                bool rc = SYCLKernel::GetCellVertexPos(current_cell_vertex_pos, current_cell_vertices_idx, MAX_VERTEX_NUM, current_cell_vertices_number, acc_vertexCoord_buf);
                if (!rc)
                {
                    out << "[ERROR]:: GetCellVertexPos Failed....(VLA < current_cell_vertices_number)" << sycl::endl;
                    return vec3(0.0, 0.0, 0.0);
                }
                // washpress
                Interpolator::CalcPolygonWachspress(current_position, current_cell_vertex_pos, current_cell_vertex_weight, current_cell_vertices_number);

                int ztop_range_front = acc_cellVertexZTop_front_buf.get_range()[0];
                int vel_range_front = acc_cellVertexVelocity_front_buf.get_range()[0];
                int ztop_range_back = acc_cellVertexZTop_back_buf.get_range()[0];
                int vel_range_back = acc_cellVertexVelocity_back_buf.get_range()[0];

                for (auto k = 0; k < ACTUALL_ZTOP_LAYER; ++k)
                {
                    double current_point_ztop_in_layer_front = 0.0;
                    double current_point_ztop_in_layer_back = 0.0;
                    // 获取每个顶点的ztop
                    double current_cell_vertex_ztop_front[MAX_VERTEX_NUM];
                    double current_cell_vertex_ztop_back[MAX_VERTEX_NUM];
                    for (auto v_idx = 0; v_idx < current_cell_vertices_number; v_idx++)
                    {
                        auto VID = current_cell_vertices_idx[v_idx];
                        if (VID < 0 || VID >= acc_vertexCoord_buf.get_range()[0]) {
                            out << "[ERROR] Invalid VID: " << VID << sycl::endl;
                            return vec3(0.0, 0.0, 0.0);
                        }

                        double ztop_front = acc_cellVertexZTop_front_buf[VID * ACTUALL_ZTOP_LAYER + k];
                        double ztop_back = acc_cellVertexZTop_back_buf[VID * ACTUALL_ZTOP_LAYER + k];
                        current_cell_vertex_ztop_front[v_idx] = ztop_front;
                        current_cell_vertex_ztop_back[v_idx] = ztop_back;
                    }
                    // 计算当前点的ZTOP
                    for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
                    {
                        current_point_ztop_in_layer_front += current_cell_vertex_weight[v_idx] * current_cell_vertex_ztop_front[v_idx];
                        current_point_ztop_in_layer_back += current_cell_vertex_weight[v_idx] * current_cell_vertex_ztop_back[v_idx];
                    }
                    current_point_ztop_front_vec[k] = current_point_ztop_in_layer_front;
                    current_point_ztop_back_vec[k] = current_point_ztop_in_layer_back;

                }

                const double EPSILON = 1e-6;
                int local_layer_front = 0;
                int local_layer_back = 0;
                for (auto k = 1; k < ACTUALL_ZTOP_LAYER; ++k)
                {
                    if (fixed_depth <= current_point_ztop_front_vec[k - 1] - EPSILON && fixed_depth >= current_point_ztop_front_vec[k] + EPSILON)
                    {
                        local_layer_front = k;
                        break;
                    }
                }
                for (auto k = 1; k < ACTUALL_ZTOP_LAYER; ++k)
                {
                    if (fixed_depth <= current_point_ztop_back_vec[k - 1] - EPSILON && fixed_depth >= current_point_ztop_back_vec[k] + EPSILON)
                    {
                        local_layer_back = k;
                        break;
                    }
                }
            
               
                if (local_layer_front == 0 || local_layer_back == 0)
                {
                    acc_sample_points_buf[global_id] = current_position;
                    return vec3(0.0, 0.0, 0.0);
                }
                else
                {
                    // 计算当前点的速度
                    double ztop_layer1, ztop_layer2, ztop_layer3, ztop_layer4;
                    ztop_layer1 = current_point_ztop_front_vec[local_layer_front];      // lower
                    ztop_layer2 = current_point_ztop_front_vec[local_layer_front - 1];  // upper

                    ztop_layer3 = current_point_ztop_back_vec[local_layer_back];        // lower
                    ztop_layer4 = current_point_ztop_back_vec[local_layer_back - 1];    // upper
                    double t_front = (sycl::fabs(fixed_depth) - sycl::fabs(ztop_layer1)) / (sycl::fabs(ztop_layer2) - sycl::fabs(ztop_layer1));
                    double t_back = (sycl::fabs(fixed_depth) - sycl::fabs(ztop_layer3)) / (sycl::fabs(ztop_layer4) - sycl::fabs(ztop_layer3));
                    
                    
                    vec3 final_vel_front;
                    vec3 final_vel_back;
                   
                    // lower_front
                    vec3 current_point_vel1_front = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_front, acc_cellVertexVelocity_front_buf);
                    // upper_front
                    vec3 current_point_vel2_front = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_front - 1, acc_cellVertexVelocity_front_buf);
                    
                    final_vel_front.x() = t_front * current_point_vel2_front.x() + (1 - t_front) * current_point_vel1_front.x();
                    final_vel_front.y() = t_front * current_point_vel2_front.y() + (1 - t_front) * current_point_vel1_front.y();
                    final_vel_front.z() = t_front * current_point_vel2_front.z() + (1 - t_front) * current_point_vel1_front.z();

                    vec3 current_point_vel1_back = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_back, acc_cellVertexVelocity_back_buf);
                
                    vec3 current_point_vel2_back = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_back - 1, acc_cellVertexVelocity_back_buf);
                    
                    final_vel_back.x() = t_back * current_point_vel2_back.x() + (1 - t_back) * current_point_vel1_back.x();
                    final_vel_back.y() = t_back * current_point_vel2_back.y() + (1 - t_back) * current_point_vel1_back.y();
                    final_vel_back.z() = t_back * current_point_vel2_back.z() + (1 - t_back) * current_point_vel1_back.z();


                    vec3 current_velocity = alpha_for_interplate * final_vel_back + (1 - alpha_for_interplate) * final_vel_front;
                    return current_velocity;
                }   

            };


            if (bFirstLoop == false) {
                if (cell_id < 0 || cell_id >= acc_numberVertexOnCell_buf.get_range()[0]) {
                    out << "[Error] cell_id out of range: " << cell_id << sycl::endl;
                    return;
                }
            }


            for (auto i_step = 0; i_step < n_steps; i_step++)
            {
                double alpha_for_interplate = (double)i_step / (double)n_steps;
                runTime += deltaT;
                position = acc_sample_points_buf[global_id];
                int firtst_cell_id = acc_def_cell_id_buf[global_id];
                // 第一次循环
                if (bFirstLoop)
                {
                    bFirstLoop = false;
                    cell_id = firtst_cell_id;
                    
                    // 找到这个CELL有多少个点
                    auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                    SYCLKernel::GetCellNeighborsIdx(cell_id, current_cell_vertices_number, cell_neig_vec, MAX_VERTEX_NUM, ACTUALL_MAX_EDGE_SIZE, acc_cells_onCell_buf);
                    
                    int acc_wirte_pints_idx = base_idx + 0;
                    acc_wirte_points_buf[acc_wirte_pints_idx] = position;
                }
                else
                {
                    // 找到这个CELL有多少个点
                    auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                    double max_len = std::numeric_limits<double>::max();
                    for (auto idx = 0; idx < current_cell_vertices_number + 1; idx++)
                    {
                        // 判断当前点 POS 在 哪个CELL 中  -> get new_cell_id
                        auto CID = cell_neig_vec[idx];
                        if (CID < 0 || CID >= acc_cellCoord_buf.get_range()[0]) continue;
                        vec3 pos = acc_cellCoord_buf[CID];
                        double len = YOSEF_LENGTH(pos - position);
                        if (len < max_len)
                        {
                            max_len = len;
                            cell_id = CID;
                        }
                    }
                    SYCLKernel::GetCellNeighborsIdx(cell_id, current_cell_vertices_number, cell_neig_vec, MAX_VERTEX_NUM, ACTUALL_MAX_EDGE_SIZE, acc_cells_onCell_buf);
                   
                }

                // TEST
                vec3 current_position = position;
/*
                // 判断是否在大陆上
                vec3 current_position = position;
                auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
                SYCLKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx, MAX_VERTEX_NUM, ACTUALL_MAX_EDGE_SIZE, acc_verticesOnCell_buf);
                bool is_land = SYCLKernel::IsInOcean(cell_id, ACTUALL_MAX_EDGE_SIZE, current_position, acc_numberVertexOnCell_buf, acc_verticesOnCell_buf, acc_vertexCoord_buf);
                if (is_land)
                {
                    continue;
                }

                
                // 计算当前点 的ZtopFront, ZtopBack
                double current_point_ztop_front_vec[MAX_VERTLEVELS];
                double current_point_ztop_back_vec[MAX_VERTLEVELS];

                vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
                double current_cell_vertex_weight[MAX_VERTEX_NUM];
                bool rc = SYCLKernel::GetCellVertexPos(current_cell_vertex_pos, current_cell_vertices_idx, MAX_VERTEX_NUM, current_cell_vertices_number, acc_vertexCoord_buf);
                if (!rc)
                {
                    out << "[ERROR]:: GetCellVertexPos Failed....(VLA < current_cell_vertices_number)" << sycl::endl;
                    return;
                }
                // washpress
                Interpolator::CalcPolygonWachspress(current_position, current_cell_vertex_pos, current_cell_vertex_weight, current_cell_vertices_number);

                int ztop_range_front = acc_cellVertexZTop_front_buf.get_range()[0];
                int vel_range_front = acc_cellVertexVelocity_front_buf.get_range()[0];
                int ztop_range_back = acc_cellVertexZTop_back_buf.get_range()[0];
                int vel_range_back = acc_cellVertexVelocity_back_buf.get_range()[0];

                for (auto k = 0; k < ACTUALL_ZTOP_LAYER; ++k)
                {
                    double current_point_ztop_in_layer_front = 0.0;
                    double current_point_ztop_in_layer_back = 0.0;
                    // 获取每个顶点的ztop
                    double current_cell_vertex_ztop_front[MAX_VERTEX_NUM];
                    double current_cell_vertex_ztop_back[MAX_VERTEX_NUM];
                    for (auto v_idx = 0; v_idx < current_cell_vertices_number; v_idx++)
                    {
                        auto VID = current_cell_vertices_idx[v_idx];
                        if (VID < 0 || VID >= acc_vertexCoord_buf.get_range()[0]) {
                            out << "[ERROR] Invalid VID: " << VID << sycl::endl;
                            return;
                        }

                        double ztop_front = acc_cellVertexZTop_front_buf[VID * ACTUALL_ZTOP_LAYER + k];
                        double ztop_back = acc_cellVertexZTop_back_buf[VID * ACTUALL_ZTOP_LAYER + k];
                        current_cell_vertex_ztop_front[v_idx] = ztop_front;
                        current_cell_vertex_ztop_back[v_idx] = ztop_back;
                    }
                    // 计算当前点的ZTOP
                    for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
                    {
                        current_point_ztop_in_layer_front += current_cell_vertex_weight[v_idx] * current_cell_vertex_ztop_front[v_idx];
                        current_point_ztop_in_layer_back += current_cell_vertex_weight[v_idx] * current_cell_vertex_ztop_back[v_idx];
                    }
                    current_point_ztop_front_vec[k] = current_point_ztop_in_layer_front;
                    current_point_ztop_back_vec[k] = current_point_ztop_in_layer_back;

                }

                const double EPSILON = 1e-6;
                int local_layer_front = 0;
                int local_layer_back = 0;
                for (auto k = 1; k < ACTUALL_ZTOP_LAYER; ++k)
                {
                    if (fixed_depth <= current_point_ztop_front_vec[k - 1] - EPSILON && fixed_depth >= current_point_ztop_front_vec[k] + EPSILON)
                    {
                        local_layer_front = k;
                        break;
                    }
                }
                for (auto k = 1; k < ACTUALL_ZTOP_LAYER; ++k)
                {
                    if (fixed_depth <= current_point_ztop_back_vec[k - 1] - EPSILON && fixed_depth >= current_point_ztop_back_vec[k] + EPSILON)
                    {
                        local_layer_back = k;
                        break;
                    }
                }
            
               
                if (local_layer_front == 0 || local_layer_back == 0)
                {
                    acc_sample_points_buf[global_id] = current_position;
                    continue;
                }
                else
                {
                    // 计算当前点的速度
                    double ztop_layer1, ztop_layer2, ztop_layer3, ztop_layer4;
                    ztop_layer1 = current_point_ztop_front_vec[local_layer_front];      // lower
                    ztop_layer2 = current_point_ztop_front_vec[local_layer_front - 1];  // upper

                    ztop_layer3 = current_point_ztop_back_vec[local_layer_back];        // lower
                    ztop_layer4 = current_point_ztop_back_vec[local_layer_back - 1];    // upper
                    double t_front = (sycl::fabs(fixed_depth) - sycl::fabs(ztop_layer1)) / (sycl::fabs(ztop_layer2) - sycl::fabs(ztop_layer1));
                    double t_back = (sycl::fabs(fixed_depth) - sycl::fabs(ztop_layer3)) / (sycl::fabs(ztop_layer4) - sycl::fabs(ztop_layer3));
                    
                    
                    vec3 final_vel_front;
                    vec3 final_vel_back;
                   
                    // lower_front
                    vec3 current_point_vel1_front = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_front, acc_cellVertexVelocity_front_buf);
                    // upper_front
                    vec3 current_point_vel2_front = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_front - 1, acc_cellVertexVelocity_front_buf);
                    
                    final_vel_front.x() = t_front * current_point_vel2_front.x() + (1 - t_front) * current_point_vel1_front.x();
                    final_vel_front.y() = t_front * current_point_vel2_front.y() + (1 - t_front) * current_point_vel1_front.y();
                    final_vel_front.z() = t_front * current_point_vel2_front.z() + (1 - t_front) * current_point_vel1_front.z();

                    vec3 current_point_vel1_back = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_back, acc_cellVertexVelocity_back_buf);
                
                    vec3 current_point_vel2_back = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_back - 1, acc_cellVertexVelocity_back_buf);
                    
                    final_vel_back.x() = t_back * current_point_vel2_back.x() + (1 - t_back) * current_point_vel1_back.x();
                    final_vel_back.y() = t_back * current_point_vel2_back.y() + (1 - t_back) * current_point_vel1_back.y();
                    final_vel_back.z() = t_back * current_point_vel2_back.z() + (1 - t_back) * current_point_vel1_back.z();
            }

                    vec3 current_velocity = alpha_for_interplate * final_vel_back + (1 - alpha_for_interplate) * final_vel_front;
*/

                // Runge-Kutta 4th order method
                double r = YOSEF_LENGTH(current_position);
                double dt = static_cast<double>(deltaT);
                vec3 k1 = CalcVelocityAt(current_position, cell_id, alpha_for_interplate);
                // p + 0.5 * k1 Δt   → 归一化回球面
                vec3 p2   = sycl::normalize(position + (dt * 0.5) * k1) * r;
                int  cid2 = cell_id;
                vec3 k2   = CalcVelocityAt(current_position, cell_id, alpha_for_interplate);
                // p + 0.5 * k2 Δt
                vec3 p3   = sycl::normalize(position + (dt * 0.5) * k2) * r;
                int  cid3 = cell_id;
                vec3 k3   = CalcVelocityAt(current_position, cell_id, alpha_for_interplate);
                // p + 1.0 * k3 Δt
                vec3 p4   = sycl::normalize(position + dt * k3) * r;
                int  cid4 = cell_id;
                vec3 k4   = CalcVelocityAt(current_position, cell_id, alpha_for_interplate);
                // 组合平均速度
                vec3 current_velocity = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;


                vec3 rotationAxis = SYCLKernel::CalcRotationAxis(current_position, current_velocity);
                double speed = YOSEF_LENGTH(current_velocity);
                double theta_rad = (speed * deltaT) / r;
                new_position = SYCLKernel::CalcPositionAfterRotation(current_position, rotationAxis, theta_rad);
                   

                acc_sample_points_buf[global_id] = new_position;
                    
                if ((i_step + 1) % (recordT / deltaT) == 0)
                {
                    // //save .//TODO
                    int acc_wirte_pints_idx = base_idx + update_points_idx;
                    acc_wirte_points_buf[acc_wirte_pints_idx] = new_position;
                    update_points_idx = update_points_idx + 1;
                }
                    
            // }
            }
        });
        
      
    });
    try {
        sycl_Q.wait();
    } catch (sycl::exception const& e) {
        std::cerr << "Caught SYCL exception: " << e.what() << std::endl;
        std::exit(1);
    }

    Debug("[VisualizeTrajectory]::Finished...");
    // auto after_write_p = wirte_points_buf.get_access<sycl::access::mode::read>();
    auto after_write_p = wirte_points_buf.get_host_access(sycl::read_only); // XYZ
    std::vector<CartesianCoord> last_points;
    

    // update trajectory_lines
    size_t line_idx = 0;
    size_t each_point_size = config->simulationDuration / config->recordT;
    size_t total_points = update_points.size();
    size_t total_lines = trajectory_lines.size();

    tbb::parallel_for(size_t(0), total_lines, [&](size_t line_idx) {
        size_t start_idx = line_idx * each_point_size;
        size_t end_idx = std::min(start_idx + each_point_size, total_points);

        for (size_t i = start_idx; i < end_idx; ++i) {
            vec3 p = after_write_p[i];
            trajectory_lines[line_idx].points.push_back(p);

            if (i == end_idx - 1 || i == total_points - 1) {
                trajectory_lines[line_idx].lastPoint = p;
            }
        }
    });

    return trajectory_lines;
}




