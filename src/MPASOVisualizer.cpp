#include "MPASOVisualizer.h"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include "VTKFileManager.hpp"


void MPASOVisualizer::VisualizeFixedLayer(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, sycl::queue& sycl_Q)
{
    int width = config->imageSize.x();
    int height = config->imageSize.y();
    auto minLat = config->LatRange.x();
    auto maxLat = config->LatRange.y();
    auto minLon = config->LonRange.x();
    auto maxLon = config->LonRange.y();
    auto fixed_layer = config->FixedLayer;


#pragma region KDTreeFunc
    std::vector<int> cell_id_vec; 
    cell_id_vec.resize(width * height); 
    for (auto i = 0; i < height; i++)
    {
        for (auto j = 0; j < width; j++)
        {
            vec2 pixel = vec2(i, j);
            vec2 latlon_r;
            GeoConverter::convertPixelToLatLonToRadians(width, height, minLat, maxLat, minLon, maxLon, pixel, latlon_r);
            vec3 current_position;
            GeoConverter::convertRadianLatLonToXYZ(latlon_r, current_position);
            int cell_id_value = -1;
            mpasoF->mGrid->searchKDT(current_position, cell_id_value);
            int global_id = i * width + j;
            cell_id_vec[global_id] = cell_id_value;
        }
    }

    Debug("[MPASOVisualizer]::Finished KD Tree Search....");
    
#pragma endregion KDTreeFunc
    
#pragma region sycl_buffer_image
    sycl::buffer<int, 1> width_buf(&width, 1);
    sycl::buffer<int, 1> height_buf(&height, 1);
    sycl::buffer<double, 1> minLat_buf(&minLat, 1);
    sycl::buffer<double, 1> maxLat_buf(&maxLat, 1);
    sycl::buffer<double, 1> minLon_buf(&minLon, 1);
    sycl::buffer<double, 1> maxLon_buf(&maxLon, 1);
    sycl::buffer<double, 1> img_buf(img->mPixels.data(), sycl::range<1>(img->mPixels.size()));
#pragma endregion sycl_buffer_image

#pragma region sycl_buffer_grid
    sycl::buffer<int, 1> cellID_buf(cell_id_vec.data(), sycl::range<1>(cell_id_vec.size()));
    sycl::buffer<vec3, 1> vertexCoord_buf(mpasoF->mGrid->vertexCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->vertexCoord_vec.size())); // CELL 顶点坐标
    sycl::buffer<vec3, 1> cellCoord_buf(mpasoF->mGrid->cellCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->cellCoord_vec.size()));       // CELL 中心坐标
    sycl::buffer<size_t, 1> numberVertexOnCell_buf(mpasoF->mGrid->numberVertexOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->numberVertexOnCell_vec.size())); // CELL 有几个顶点
    sycl::buffer<size_t, 1> verticesOnCell_buf(mpasoF->mGrid->verticesOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->verticesOnCell_vec.size()));             // 
    sycl::buffer<size_t, 1> cellsOnVertex_buf(mpasoF->mGrid->cellsOnVertex_vec.data(), sycl::range<1>(mpasoF->mGrid->cellsOnVertex_vec.size()));
#pragma endregion   sycl_buffer_grid


#pragma region sycl_buffer_velocity
    sycl::buffer<vec3, 1> cellCenterVelocity_buf(mpasoF->mSol_Front->cellCenterVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellCenterVelocity_vec.size()));
    sycl::buffer<vec3, 1> cellVertexVelocity_buf(mpasoF->mSol_Front->cellVertexVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexVelocity_vec.size()));
    sycl::buffer<double, 1> cellVertexZTop_buf(mpasoF->mSol_Front->cellVertexZTop_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexZTop_vec.size()));
    sycl::buffer<double, 1> cellCenterZonalVelocity_buf(mpasoF->mSol_Front->cellZonalVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellZonalVelocity_vec.size()));
     sycl::buffer<double, 1> cellCenterMeridionalVelocity_buf(mpasoF->mSol_Front->cellMeridionalVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellMeridionalVelocity_vec.size()));
#pragma endregion sycl_buffer_velocity   

    sycl_Q.submit([&](sycl::handler& cgh) 
    {

#pragma region sycl_acc_image
        auto width_acc = width_buf.get_access<sycl::access::mode::read>(cgh);
        auto height_acc = height_buf.get_access<sycl::access::mode::read>(cgh);
        auto minLat_acc = minLat_buf.get_access<sycl::access::mode::read>(cgh);
        auto maxLat_acc = maxLat_buf.get_access<sycl::access::mode::read>(cgh);
        auto minLon_acc = minLon_buf.get_access<sycl::access::mode::read>(cgh);
        auto maxLon_acc = maxLon_buf.get_access<sycl::access::mode::read>(cgh);
        auto img_acc = img_buf.get_access<sycl::access::mode::read_write>(cgh);
#pragma endregion sycl_acc_image

#pragma region sycl_acc_grid
        auto acc_cellID_buf = cellID_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_vertexCoord_buf = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnCell_buf = verticesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnVertex_buf = cellsOnVertex_buf.get_access<sycl::access::mode::read>(cgh);
        int grid_cell_size = mpasoF->mGrid->mCellsSize;
        int grid_max_edge = mpasoF->mGrid->mMaxEdgesSize;
#pragma endregion sycl_acc_grid

#pragma region sycl_acc_velocity
        auto acc_cellCenterVelocity_buf = cellCenterVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexVelocity_buf = cellVertexVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexZTop_buf = cellVertexZTop_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCenterZonalVelocity_buf = cellCenterZonalVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCenterMeridionalVelocity_buf = cellCenterMeridionalVelocity_buf.get_access<sycl::access::mode::read>(cgh);
#pragma endregion cellVertexZTop_buf

        // sycl::range<2> global_range(height, width);
        sycl::range<2> global_range((height + 7) / 8 * 8, (width + 7) / 8 * 8);
        sycl::range<2> local_range(8, 8);  

        cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), [=](sycl::nd_item<2> idx) 
        {

            int height_index = idx.get_global_id(0);
            int width_index = idx.get_global_id(1);
            int global_id = height_index * width_acc[0] + width_index;
                
            if(height_index < height && width_index < width)
            {
                const int CELL_SIZE = 4062533;
                const int VERTEX_NUM = 8;
                const int NEIGHBOR_NUM = 3;
                const int TOTAY_ZTOP_LAYER = 80;
                const int VERTLEVELS = 80;
                auto double_nan = std::numeric_limits<double>::quiet_NaN();
                vec3 vec3_nan = { double_nan, double_nan, double_nan };

#pragma region CalcPosition&CellID
                vec2 current_pixel = { height_index, width_index };
                CartesianCoord current_position;
                SphericalCoord current_latlon_r;
                GeoConverter::convertPixelToLatLonToRadians(width_acc[0], height_acc[0], minLat_acc[0], maxLat_acc[0], minLon_acc[0], maxLon_acc[0], current_pixel, current_latlon_r);
                GeoConverter::convertRadianLatLonToXYZ(current_latlon_r, current_position);
                int cell_id = acc_cellID_buf[global_id];
#pragma endregion CalcPosition&CellID

#pragma region IsOnOcean
                // 判断是否在大陆上
                bool is_land = false;
                // 1.1 计算这个CELL 有多少个顶点
                auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                // 1.2 找出所有候选顶点
                size_t current_cell_vertices_idx[VERTEX_NUM];
                for (size_t k = 0; k < VERTEX_NUM; ++k)
                {
                    current_cell_vertices_idx[k] = acc_verticesOnCell_buf[cell_id * VERTEX_NUM + k] - 1; // Assuming 7 is the max number of vertices per cell
                }
                // 1.3 不存在的顶点设置为nan
                auto nan = std::numeric_limits<size_t>::max();
                for (size_t k = current_cell_vertices_number; k < VERTEX_NUM; ++k)
                {
                    current_cell_vertices_idx[k] = nan;
                }
                // =============================== 找到VERTEX_NUM个顶点
                double normalsConsistency[VERTEX_NUM];
                for (auto k = 0; k < current_cell_vertices_number; k++)
                {
                    auto A_idx = current_cell_vertices_idx[k];
                    auto B_idx = current_cell_vertices_idx[(k + 1) % current_cell_vertices_number];
                    auto A = acc_vertexCoord_buf[A_idx];
                    auto B = acc_vertexCoord_buf[B_idx];
                    vec3 O(0.0, 0.0, 0.0);
                    auto AO = O - A;
                    auto BO = O - B;
                    auto A_point = current_position - A;
                    vec3 surface_normal = YOSEF_CROSS(AO, BO);
                    double direction = YOSEF_DOT(surface_normal, A_point);
                    normalsConsistency[k] = direction;
                }
                int sign = (normalsConsistency[0] > 0) ? 1 : -1;
                for (auto k = 1; k < current_cell_vertices_number; k++)
                {
                    int currentSign = (normalsConsistency[k] > 0) ? 1 : -1;
                    if (currentSign != sign)
                    {
                        // 这个点在大陆上
                        is_land = true;
                        break;
                    }
                }
                if (is_land)
                {
                    SetPixel(img_acc, width_acc[0], height_acc[0], height_index, width_index, vec3_nan);
                    return;
                }   
                    
#pragma endregion IsOnOcean            

                 vec3 imgValue = vec3_nan;

                if (true)//config->PositionType == CalcPositionType::kPoint //TODO
                {
                    //  计算每个Cell 顶点的速度 和 坐标
                    vec3 current_cell_vertices_velocity[VERTEX_NUM];
                    double current_cell_vertices_ztop[VERTEX_NUM];
                    vec3 current_verteices_positions[VERTEX_NUM];
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
                    for (auto v_idx = current_cell_vertices_number; v_idx < VERTEX_NUM; ++v_idx)
                    {
                        current_cell_vertices_velocity[v_idx] = vec3_nan;
                        current_verteices_positions[v_idx] = vec3_nan;
                    }
                    //  计算出当前点的速度 用 Wachspress coordinates 的参数
                    double current_cell_weight[VERTEX_NUM];
                    Interpolator::CalcPolygonWachspress(current_position, current_verteices_positions, current_cell_weight, current_cell_vertices_number);
                    for (auto v_idx = current_cell_vertices_number; v_idx < VERTEX_NUM; ++v_idx)
                    {
                        current_cell_weight[v_idx] = double_nan;
                    }
                    //TODO
                    // 要算什么 kZional
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

                    // // TEST 直接获取mer zonal
                    // double tmp_zonal = acc_cellCenterZonalVelocity_buf[cell_id * VERTLEVELS + fixed_layer];
                    // double tmp_mer = acc_cellCenterMeridionalVelocity_buf[cell_id * VERTLEVELS + fixed_layer];
                    // imgValue = { tmp_zonal, tmp_mer, 0.0 };
                    
                }
                    
                SetPixel(img_acc, width, height, height_index, width_index, imgValue);
            } 
        });
    });
    sycl_Q.wait();
   
}

template <typename T>
bool is_within_epsilon(T a, T b, T epsilon) {
    return (a >= b - epsilon) && (a <= b + epsilon);
}



void MPASOVisualizer::VisualizeFixedDepth(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, sycl::queue& sycl_Q)
{
    int width = config->imageSize.x();
    int height = config->imageSize.y();
    auto minLat = config->LatRange.x();
    auto maxLat = config->LatRange.y();
    auto minLon = config->LonRange.x();
    auto maxLon = config->LonRange.y();
    auto fixed_depth = -config->FixedDepth;

#pragma region KDTreeFunc
    std::vector<int> cell_id_vec; 
    cell_id_vec.resize(width * height); 
    for (auto i = 0; i < height; i++)
    {
        for (auto j = 0; j < width; j++)
        {
            vec2 pixel = vec2(i, j);
            vec2 latlon_r;
            GeoConverter::convertPixelToLatLonToRadians(width, height, minLat, maxLat, minLon, maxLon, pixel, latlon_r);
            vec3 current_position;
            GeoConverter::convertRadianLatLonToXYZ(latlon_r, current_position);
            int cell_id_value = -1;
            mpasoF->mGrid->searchKDT(current_position, cell_id_value);
            int global_id = i * width + j;
            cell_id_vec[global_id] = cell_id_value;
        }
    }

    Debug("[MPASOVisualizer]::Finished KD Tree Search....");
#pragma endregion KDTreeFunc

#pragma region sycl_buffer_image
        sycl::buffer<int, 1> width_buf(&width, 1);
        sycl::buffer<int, 1> height_buf(&height, 1);
        sycl::buffer<double, 1> minLat_buf(&minLat, 1);
        sycl::buffer<double, 1> maxLat_buf(&maxLat, 1);
        sycl::buffer<double, 1> minLon_buf(&minLon, 1);
        sycl::buffer<double, 1> maxLon_buf(&maxLon, 1);
        sycl::buffer<double, 1> img_buf(img->mPixels.data(), sycl::range<1>(img->mPixels.size()));
        sycl::buffer<double, 1> depth_buf(&fixed_depth, 1);
#pragma endregion sycl_buffer_image

#pragma region sycl_buffer_grid
        sycl::buffer<int, 1> cellID_buf(cell_id_vec.data(), sycl::range<1>(cell_id_vec.size()));
        sycl::buffer<vec3, 1> vertexCoord_buf(mpasoF->mGrid->vertexCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->vertexCoord_vec.size())); // CELL 顶点坐标
        sycl::buffer<vec3, 1> cellCoord_buf(mpasoF->mGrid->cellCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->cellCoord_vec.size()));       // CELL 中心坐标
        sycl::buffer<size_t, 1> numberVertexOnCell_buf(mpasoF->mGrid->numberVertexOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->numberVertexOnCell_vec.size())); // CELL 有几个顶点
        sycl::buffer<size_t, 1> verticesOnCell_buf(mpasoF->mGrid->verticesOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->verticesOnCell_vec.size()));             // 
        sycl::buffer<size_t, 1> cellsOnVertex_buf(mpasoF->mGrid->cellsOnVertex_vec.data(), sycl::range<1>(mpasoF->mGrid->cellsOnVertex_vec.size()));
#pragma endregion   sycl_buffer_grid


#pragma region sycl_buffer_velocity
        sycl::buffer<vec3, 1> cellVertexVelocity_buf(mpasoF->mSol_Front->cellVertexVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexVelocity_vec.size()));
        sycl::buffer<double, 1> cellVertexZTop_buf(mpasoF->mSol_Front->cellVertexZTop_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexZTop_vec.size()));
#pragma endregion sycl_buffer_velocity   

    sycl_Q.submit([&](sycl::handler& cgh) 
    {
#pragma region sycl_acc_image
        auto width_acc = width_buf.get_access<sycl::access::mode::read>(cgh);
        auto height_acc = height_buf.get_access<sycl::access::mode::read>(cgh);
        auto minLat_acc = minLat_buf.get_access<sycl::access::mode::read>(cgh);
        auto maxLat_acc = maxLat_buf.get_access<sycl::access::mode::read>(cgh);
        auto minLon_acc = minLon_buf.get_access<sycl::access::mode::read>(cgh);
        auto maxLon_acc = maxLon_buf.get_access<sycl::access::mode::read>(cgh);
        auto depth_acc = depth_buf.get_access<sycl::access::mode::read>(cgh);
        auto img_acc = img_buf.get_access<sycl::access::mode::read_write>(cgh);
#pragma endregion sycl_acc_image

#pragma region sycl_acc_grid
        auto acc_cellID_buf = cellID_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_vertexCoord_buf = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnCell_buf = verticesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnVertex_buf = cellsOnVertex_buf.get_access<sycl::access::mode::read>(cgh);
        int grid_cell_size = mpasoF->mGrid->mCellsSize;
#pragma endregion sycl_acc_grid

#pragma region sycl_acc_velocity
        auto acc_cellVertexVelocity_buf = cellVertexVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexZTop_buf = cellVertexZTop_buf.get_access<sycl::access::mode::read>(cgh);
#pragma endregion sycl_acc_velocity

           
        // sycl::range<2> global_range(height, width);
        sycl::range<2> global_range((height + 7) / 8 * 8, (width + 7) / 8 * 8);
        sycl::range<2> local_range(8, 8);
                        
        cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), [=](sycl::nd_item<2> idx) 
        {

            int height_index = idx.get_global_id(0);
            int width_index = idx.get_global_id(1);
            int global_id = height_index * width_acc[0] + width_index;
                        
            if(height_index < height && width_index < width)
            {
                const int CELL_SIZE = 4062533;
                const int VERTEX_NUM = 8;
                const int NEIGHBOR_NUM = 3;
                const int TOTAY_ZTOP_LAYER = 80;
                const int VERTLEVELS = 80;
                const double DEPTH = depth_acc[0];
                auto double_nan = std::numeric_limits<double>::quiet_NaN();
                vec3 vec3_nan = { double_nan, double_nan, double_nan };

#pragma region CalcPosition&CellID
                vec2 current_pixel = { height_index, width_index };
                CartesianCoord current_position;
                SphericalCoord current_latlon_r;
                GeoConverter::convertPixelToLatLonToRadians(width_acc[0], height_acc[0], minLat_acc[0], maxLat_acc[0], minLon_acc[0], maxLon_acc[0], current_pixel, current_latlon_r);
                GeoConverter::convertRadianLatLonToXYZ(current_latlon_r, current_position);
                int cell_id = acc_cellID_buf[global_id];
#pragma endregion CalcPosition&CellID

#pragma region IsOnOcean
                // 判断是否在大陆上
                bool is_land = false;
                // 1.1 计算这个CELL 有多少个顶点
                auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                // 1.2 找出所有候选顶点
                size_t current_cell_vertices_idx[VERTEX_NUM];
                for (size_t k = 0; k < VERTEX_NUM; ++k)
                {
                    current_cell_vertices_idx[k] = acc_verticesOnCell_buf[cell_id * VERTEX_NUM + k] - 1; // Assuming 7 is the max number of vertices per cell
                }
                // 1.3 不存在的顶点设置为nan
                auto nan = std::numeric_limits<size_t>::max();
                for (size_t k = current_cell_vertices_number; k < VERTEX_NUM; ++k)
                {
                    current_cell_vertices_idx[k] = nan;
                }
                // =============================== 找到7个顶点
                double normalsConsistency[VERTEX_NUM];
                for (auto k = 0; k < current_cell_vertices_number; k++)
                {
                    auto A_idx = current_cell_vertices_idx[k];
                    auto B_idx = current_cell_vertices_idx[(k + 1) % current_cell_vertices_number];
                    auto A = acc_vertexCoord_buf[A_idx];
                    auto B = acc_vertexCoord_buf[B_idx];
                    vec3 O(0.0, 0.0, 0.0);
                    auto AO = O - A;
                    auto BO = O - B;
                    auto A_point = current_position - A;
                    vec3 surface_normal = YOSEF_CROSS(AO, BO);
                    double direction = YOSEF_DOT(surface_normal, A_point);
                    normalsConsistency[k] = direction;
                }
                int sign = (normalsConsistency[0] > 0) ? 1 : -1;
                for (auto k = 1; k < current_cell_vertices_number; k++)
                {
                    int currentSign = (normalsConsistency[k] > 0) ? 1 : -1;
                    if (currentSign != sign)
                    {
                        // 这个点在大陆上
                        is_land = true;
                        break;
                    }
                }
                if (is_land)
                {
                    SetPixel(img_acc, width_acc[0], height_acc[0], height_index, width_index, vec3_nan);
                    return;
                }
#pragma endregion IsOnOcean            

                vec3 imgValue = vec3_nan;
                double current_point_ztop_vec[VERTLEVELS];

                vec3 current_cell_vertex_pos[VERTEX_NUM];
                double current_cell_vertex_weight[VERTEX_NUM];
                for (auto v_idx = 0; v_idx < current_cell_vertices_number; v_idx++)
                {
                    auto VID = current_cell_vertices_idx[v_idx];
                    vec3 pos = acc_vertexCoord_buf[VID];
                    current_cell_vertex_pos[v_idx] = pos;
                }
                // washpress
                Interpolator::CalcPolygonWachspress(current_position, current_cell_vertex_pos, current_cell_vertex_weight, current_cell_vertices_number);
                      
                for (auto k = 0; k < VERTLEVELS; ++k)
                {
                    double current_point_ztop_in_layer = 0.0;
                    // 获取每个顶点的ztop
                    double current_cell_vertex_ztop[VERTEX_NUM];
                    for (auto v_idx = 0; v_idx < current_cell_vertices_number; v_idx++)
                    {
                        auto VID = current_cell_vertices_idx[v_idx];
                        double ztop = acc_cellVertexZTop_buf[VID * 80 + k];
                        current_cell_vertex_ztop[v_idx] = ztop;
                    }
                    
                    // 计算当前点的ZTOP
                    for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
                    {
                        current_point_ztop_in_layer += current_cell_vertex_weight[v_idx] * current_cell_vertex_ztop[v_idx];
                    }
                    current_point_ztop_vec[k] = current_point_ztop_in_layer;
                }

                // int layer = -1;
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
                if (local_layer == 0)
                {
                    imgValue = vec3_nan;
                    SetPixel(img_acc, width, height, height_index, width_index, imgValue);
                    return;
                }
                else
                {
              
                    auto layer = local_layer;
                    double ztop_layer1, ztop_layer2;
                    ztop_layer1 = current_point_ztop_vec[layer];
                    ztop_layer2 = current_point_ztop_vec[layer - 1];
                    double t = (sycl::abs(DEPTH) - sycl::abs(ztop_layer1)) / (sycl::abs(ztop_layer2) - sycl::abs(ztop_layer1));
                    double current_point_ztop;
                    current_point_ztop = t * ztop_layer1 + (1 - t) * ztop_layer2;

                    vec3 final_vel;
                    // 算出 这两个layer的速度
                    vec3 vertex_vel1[VERTEX_NUM];
                    vec3 vertex_vel2[VERTEX_NUM];
                    vec3 current_point_vel1 = { 0.0, 0.0, 0.0 };
                    vec3 current_point_vel2 = { 0.0, 0.0, 0.0 };
                    for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
                    {
                        auto VID = current_cell_vertices_idx[v_idx];
                        vec3 vel1 = acc_cellVertexVelocity_buf[VID * 80 + layer]; 
                        vec3 vel2 = acc_cellVertexVelocity_buf[VID * 80 + layer - 1];
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
                            
                    // imgValue = final_vel;
                    double zional_velocity, merminoal_velicity;
                    GeoConverter::convertXYZVelocityToENU(current_position, final_vel, zional_velocity, merminoal_velicity);
                    current_point_velocity_enu = {zional_velocity, merminoal_velicity, 0.0};
                    
                    
                    

                }
                
                SetPixel(img_acc, width, height, height_index, width_index, current_point_velocity_enu);
                
            }

        });
    });

    sycl_Q.wait();

}






void MPASOVisualizer::VisualizeFixedLatitude(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, sycl::queue& sycl_Q)
{
    int width = config->imageSize.x();
    int height = config->imageSize.y();
    auto minDepth = config->DepthRange.x();
    auto maxDepth = config->DepthRange.y();
    auto minLon = config->LonRange.x();
    auto maxLon = config->LonRange.y();
    auto fixed_lat = config->FixedLatitude;

//#pragma region KDTreeFunc
//    std::vector<int> cell_id_vec;
//    cell_id_vec.resize(width * height);
//    
//    float i_step = (maxDepth - minDepth) / height;
//    float j_step = (maxLon - minLon) / width;
//    for (float i = minDepth; i < maxDepth; i = i + i_step)
//    {
//        for (float j = minLon; j < maxLon; j = j + j_step)
//        {
//            auto Lat = fixed_lat;
//            auto Lon = j;
//            SphericalCoord latlon_r = vec2(Lat * (M_PI / 180.0f), Lon * (M_PI / 180.0f));
//            CartesianCoord current_position;
//            GeoConverter::convertRadianLatLonToXYZ(latlon_r, current_position);
//            int cell_id_value = -1;
//            mpasoF->mGrid->searchKDT(current_position, cell_id_value);
//            int global_id = i * width + j;
//            cell_id_vec[global_id] = cell_id_value;
//        }
//    }
//
//    Debug("MPASOVisualizer::Finished KD Tree Search....");
//#pragma endregion KDTreeFunc

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
            double t = (sycl::abs(DEPTH) - sycl::abs(ztop_layer1)) / (sycl::abs(ztop_layer2) - sycl::abs(ztop_layer1));
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
}








void MPASOVisualizer::GenerateSamplePoint(std::vector<CartesianCoord>& points, SamplingSettings* config)
{
    auto minLat = config->sampleLatitudeRange.x(); auto maxLat = config->sampleLatitudeRange.y();
    auto minLon = config->sampleLongitudeRange.x(); auto maxLon = config->sampleLongitudeRange.y();

    double i_step = (maxLat - minLat) / static_cast<double>(config->sampleNumer.x() - 1);
    double j_step = (maxLon - minLon) / static_cast<double>(config->sampleNumer.y() - 1);

    for (double i = minLat ; i < maxLat; i += i_step)
    {
        for (double j = minLon; j < maxLon; j += j_step)
        {
            CartesianCoord p = { j, i, config->sampleDepth };
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

void MPASOVisualizer::GenerateGaussianSpherePoints(std::vector<CartesianCoord>& points, SamplingSettings* config, int numPoints, double meanLat, double meanLon, double stdDev)
{
    auto minLat = config->sampleLatitudeRange.x(); auto maxLat = config->sampleLatitudeRange.y();
    auto minLon = config->sampleLongitudeRange.x(); auto maxLon = config->sampleLongitudeRange.y();

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

#define EARTH_RADIUS            6371.01 // Earth's radius in kilometers
static void convertLatLonToXYZ(vec2& thetaPhi, vec3& position)
{
    /*
    *  convert lat lon to xyz based on the earth radius
    *  unit: radians (lat and lon)
    *  thetaPhi: (theta, phi) latitude and longitude
    *  Considering the latitude and longitude,
    *  it is a little different from the conventional
    *  spherical coordinate conversion.
    */

    auto theta = thetaPhi.x();
    auto phi = thetaPhi.y();
    auto r = EARTH_RADIUS * 1000.0;
    position.x() = r * sycl::cos(theta) * sycl::cos(phi);
    position.y() = r * sycl::cos(theta) * sycl::sin(phi);
    position.z() = r * sycl::sin(theta);
}
static void convertXYZToLatLon(vec3& position, vec2& thetaPhi)
{

    /*
    *  convert xyz to lat lon based on the earth radius
    *  unit: radians (lat and lon)
    *  position: (x, y, z) position
    *  thetaPhi: (theta, phi) latitude and longitude\
    *  Considering the latitude and longitude,
    *  it is a little different from the conventional
    *  spherical coordinate conversion.
    */

    double x = position.x();
    double y = position.y();
    double z = position.z();
    double r = sycl::sqrt(x * x + y * y + z * z);

    double theta = sycl::asin(z / r);
    double phi = sycl::atan2(y, x);

    thetaPhi.x() = theta;
    thetaPhi.y() = phi;
}
vec3 computeRotationAxis(const vec3& position, const vec3& velocity)
{
    vec3 axis;
    axis.x() = position.y() * velocity.z() - position.z() * velocity.y();
    axis.y() = position.z() * velocity.x() - position.x() * velocity.z();
    axis.z() = position.x() * velocity.y() - position.y() * velocity.x();
    return axis;
}

double magnitude(const vec3& v)
{
    return sycl::sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
}

vec3 normalize(const vec3& v)
{
    double length = sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
    vec3 normalized = { v.x() / length, v.y() / length, v.z() / length };
    return normalized;
}

void rotateAroundAxis(const vec3& point, const vec3& axis, double theta, double& x, double& y, double& z)
{
    // 传入的是角度
    double PI = 3.14159265358979323846;
    double thetaRad = theta * PI / 180.0;
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


std::vector<CartesianCoord>  MPASOVisualizer::VisualizeTrajectory(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, sycl::queue& sycl_Q)
{
    
    std::vector<vec3> update_points;
    if (!update_points.empty()) update_points.clear();
    update_points.resize(points.size() * (config->simulationDuration / config->recordT));

#pragma region sycl_buffer_grid
    sycl::buffer<vec3, 1> vertexCoord_buf(mpasoF->mGrid->vertexCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->vertexCoord_vec.size())); // CELL 顶点坐标
    sycl::buffer<vec3, 1> cellCoord_buf(mpasoF->mGrid->cellCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->cellCoord_vec.size()));       // CELL 中心坐标
    sycl::buffer<size_t, 1> numberVertexOnCell_buf(mpasoF->mGrid->numberVertexOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->numberVertexOnCell_vec.size())); // CELL 有几个顶点
    sycl::buffer<size_t, 1> verticesOnCell_buf(mpasoF->mGrid->verticesOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->verticesOnCell_vec.size()));             // 
    sycl::buffer<size_t, 1> cellsOnVertex_buf(mpasoF->mGrid->cellsOnVertex_vec.data(), sycl::range<1>(mpasoF->mGrid->cellsOnVertex_vec.size()));
    sycl::buffer<size_t, 1> cells_onCell_buf(mpasoF->mGrid->cellsOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->cellsOnCell_vec.size()));
#pragma endregion   sycl_buffer_grid


#pragma region sycl_buffer_velocity
    sycl::buffer<vec3, 1> cellVertexVelocity_buf(mpasoF->mSol_Front->cellVertexVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexVelocity_vec.size()));
    sycl::buffer<double, 1> cellVertexZTop_buf(mpasoF->mSol_Front->cellVertexZTop_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexZTop_vec.size()));
#pragma endregion sycl_buffer_velocity   

    sycl::buffer<int, 1> cellID_buf(default_cell_id.data(), sycl::range<1>(default_cell_id.size()));
    sycl::buffer<vec3> wirte_points_buf(update_points.data(), sycl::range<1>(update_points.size()));
    sycl::buffer<vec3> sample_points_buf(points.data(), sycl::range<1>(points.size()));
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
#pragma endregion sycl_acc_grid

#pragma region sycl_acc_velocity
        auto acc_cellVertexVelocity_buf = cellVertexVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexZTop_buf = cellVertexZTop_buf.get_access<sycl::access::mode::read>(cgh);
#pragma endregion sycl_acc_velocity

        auto acc_def_cell_id_buf = cellID_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_wirte_points_buf = wirte_points_buf.get_access<sycl::access::mode::write>(cgh);
        auto acc_sample_points_buf = sample_points_buf.get_access<sycl::access::mode::read_write>(cgh);

        sycl::stream out(1024, 256, cgh);
        int times = config->simulationDuration / config->deltaT;
        int each_points_size = config->simulationDuration / config->recordT;
        int recordT = config->recordT;
        int deltaT = config->deltaT;

        
        
        cgh.parallel_for(sycl::range<1>(points.size()), [=](sycl::item<1> item) 
        {

            int global_id = item[0];
            const int CELL_SIZE = 4062533;
            const int VERTEX_NUM = 8;
            const int VERTEX_NUM_ADD_ONE = 8;
            const int NEIGHBOR_NUM = 3;
            const int TOTAY_ZTOP_LAYER = 80;
            const int VERTLEVELS = 80;
            double fixed_depth = -800.0; //TODO

            double runTime = 0.0;
            int save_times = 0;
            bool bFirstLoop = true;
            int cell_id_vec[VERTEX_NUM];
            int base_idx = global_id * each_points_size;
            int update_points_idx = 0;

            // 1. 获取point position
            
            vec3 position; 
            int cell_id = -1;
            vec3 new_position;
            double pos_x, pos_y, pos_z;
            int cell_neig_vec[VERTEX_NUM_ADD_ONE];


            for (auto times_i = 0; times_i < times; times_i++)
            {
                runTime += deltaT;
                position = acc_sample_points_buf[global_id];
                int firtst_cell_id = acc_def_cell_id_buf[global_id];
                //out << times_i << " 1::: " << position.x() << " " << position.y() << " " << position.z() << sycl::endl;
                if (bFirstLoop)
                {

                    bFirstLoop = false;
                    cell_id = firtst_cell_id;
                    
                    // 找到有多少个点
                    auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                    cell_neig_vec[0] = cell_id;
                    for (auto k = 1; k < current_cell_vertices_number; k++)
                    {
                        int negi_cell_id = acc_cells_onCell_buf[VERTEX_NUM * cell_id + k];
                        cell_neig_vec[k] = negi_cell_id - 1;
                    }
                    for (auto k = current_cell_vertices_number; k < VERTEX_NUM; k++)
                    {
                        cell_neig_vec[k] = std::numeric_limits<int>::quiet_NaN();
                    }
                }
                else
                {
                    auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                    double max_len = std::numeric_limits<double>::max();
                    for (auto idx = 0; idx < current_cell_vertices_number + 1; idx++)
                    {
                        auto CID = cell_neig_vec[idx];
                        vec3 pos = acc_cellCoord_buf[CID];
                        double len = YOSEF_LENGTH(pos - position);
                        if (len < max_len)
                        {
                            max_len = len;
                            cell_id = CID;
                        }
                    }
                    cell_neig_vec[0] = cell_id;
                    for (auto k = 1; k < current_cell_vertices_number; k++)
                    {
                        int negi_cell_id = acc_cells_onCell_buf[VERTEX_NUM * cell_id + k];
                        cell_neig_vec[k] = negi_cell_id - 1;
                    }
                    for (auto k = current_cell_vertices_number; k < VERTEX_NUM; k++)
                    {
                        cell_neig_vec[k] = std::numeric_limits<int>::quiet_NaN();
                    }
                }

                // out << "1.5::::: tmp cell_id " << cell_id << sycl::endl;



#pragma region IsOnOcean
                // 判断是否在大陆上
                bool is_land = false;
                // 1.1 计算这个CELL 有多少个顶点
                auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                // 1.2 找出所有候选顶点
                size_t current_cell_vertices_idx[VERTEX_NUM];
                for (size_t k = 0; k < VERTEX_NUM; ++k)
                {
                    current_cell_vertices_idx[k] = acc_verticesOnCell_buf[cell_id * VERTEX_NUM + k] - 1; // Assuming 7 is the max number of vertices per cell
                }
                // 1.3 不存在的顶点设置为nan
                auto nan = std::numeric_limits<size_t>::max();
                for (size_t k = current_cell_vertices_number; k < VERTEX_NUM; ++k)
                {
                    current_cell_vertices_idx[k] = nan;
                }
                // =============================== 找到7个顶点
                double normalsConsistency[VERTEX_NUM];
                for (auto k = 0; k < current_cell_vertices_number; k++)
                {
                    auto A_idx = current_cell_vertices_idx[k];
                    auto B_idx = current_cell_vertices_idx[(k + 1) % current_cell_vertices_number];
                    auto A = acc_vertexCoord_buf[A_idx];
                    auto B = acc_vertexCoord_buf[B_idx];
                    vec3 O(0.0, 0.0, 0.0);
                    auto AO = O - A;
                    auto BO = O - B;
                    auto A_point = position - A;
                    vec3 surface_normal = YOSEF_CROSS(AO, BO);
                    double direction = YOSEF_DOT(surface_normal, A_point);
                    normalsConsistency[k] = direction;
                }
                int sign = (normalsConsistency[0] > 0) ? 1 : -1;
                for (auto k = 1; k < current_cell_vertices_number; k++)
                {
                    int currentSign = (normalsConsistency[k] > 0) ? 1 : -1;
                    if (currentSign != sign)
                    {
                        // 这个点在大陆上
                        is_land = true;
                        break;
                    }
                }
                if (is_land)
                {
                    continue;
                }
#pragma endregion IsOnOcean  
                
               
                double current_point_ztop_vec[VERTLEVELS];
                vec3 current_cell_vertex_pos[VERTEX_NUM];
                double current_cell_vertex_weight[VERTEX_NUM];
                for (auto v_idx = 0; v_idx < current_cell_vertices_number; v_idx++)
                {
                    auto VID = current_cell_vertices_idx[v_idx];
                    vec3 pos = acc_vertexCoord_buf[VID];
                    current_cell_vertex_pos[v_idx] = pos;
                }
                // washpress
                Interpolator::CalcPolygonWachspress(position, current_cell_vertex_pos, current_cell_vertex_weight, current_cell_vertices_number);

                for (auto k = 0; k < VERTLEVELS; ++k)
                {
                    double current_point_ztop_in_layer = 0.0;
                    // 获取每个顶点的ztop
                    double current_cell_vertex_ztop[VERTEX_NUM];
                    for (auto v_idx = 0; v_idx < current_cell_vertices_number; v_idx++)
                    {
                        auto VID = current_cell_vertices_idx[v_idx];
                        double ztop = acc_cellVertexZTop_buf[VID * 80 + k];
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
                const double EPSILON = 1e-6;
                for (auto k = 1; k < VERTLEVELS; ++k)
                {
                    if (fixed_depth <= current_point_ztop_vec[k - 1] - EPSILON && fixed_depth >= current_point_ztop_vec[k] + EPSILON)
                    {
                        layer = k;
                        break;
                    }
                }
            
               
                if (layer == -1)
                {
                    continue;
                }
                else
                {
                    double ztop_layer1, ztop_layer2;
                    ztop_layer1 = current_point_ztop_vec[layer];
                    ztop_layer2 = current_point_ztop_vec[layer - 1];
                    double t = (sycl::abs(fixed_depth) - sycl::abs(ztop_layer1)) / (sycl::abs(ztop_layer2) - sycl::abs(ztop_layer1));
                    double current_point_ztop;
                    current_point_ztop = t * ztop_layer1 + (1 - t) * ztop_layer2;
                    //imgValue = { current_point_ztop , current_point_ztop , current_point_ztop };
                    vec3 final_vel;
                    // 算出 这两个layer的速度
                    vec3 vertex_vel1[VERTEX_NUM];
                    vec3 vertex_vel2[VERTEX_NUM];
                    vec3 current_point_vel1 = { 0.0, 0.0, 0.0 };
                    vec3 current_point_vel2 = { 0.0, 0.0, 0.0 };
                    for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
                    {
                        auto VID = current_cell_vertices_idx[v_idx];
                        vec3 vel1 = acc_cellVertexVelocity_buf[VID * 80 + layer];
                        vec3 vel2 = acc_cellVertexVelocity_buf[VID * 80 + layer - 1];
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

                    vec3 current_velocity = final_vel;

                    double r = YOSEF_LENGTH(position);
                    vec3 rotationAxis = computeRotationAxis(position, current_velocity);

                    double speed = magnitude(current_velocity);
                    double theta = (speed * deltaT) / r;
                    double thetaInDegrees = theta * (180.0 / 3.141592653589793); // 将弧度转换为角度

                    //vec3 rotatedPosition = 
                    rotateAroundAxis(position, rotationAxis, thetaInDegrees, pos_x, pos_y, pos_z);
                    //out << " 2:::: " << position.x() << " " << position.y() << " " << position.z() << " ps " << pos_x << " " << pos_y << " " << pos_z << sycl::endl;   
                    new_position = {pos_x, pos_y, pos_z};
                    acc_sample_points_buf[global_id] = new_position;
                    
                    if ((int)runTime % recordT == 0)
                    {
                        //save .//TODO
                        int acc_wirte_pints_idx = base_idx + update_points_idx;
                        acc_wirte_points_buf[acc_wirte_pints_idx] = new_position;
                        update_points_idx = update_points_idx + 1;
                    }

                } 
            
            
            
            }
        });
       
    });
    sycl_Q.wait();

    Debug("[VisualizeTrajectory]::Finished...");
    auto after_p = sample_points_buf.get_access<sycl::access::mode::read>(); 
    auto after_write_p = wirte_points_buf.get_access<sycl::access::mode::read>();
    std::vector<CartesianCoord> last_points;
    // for (size_t i = 0; i < points.size(); ++i) 
    // {
    //     vec3 p = after_p[i];
    //     // Convert to lat/lon
    //     vec2 new_latlon;
    //     GeoConverter::convertXYZToLatLonDegree(p, new_latlon);
    //     std::cout << std::fixed << std::setprecision(4) <<"GPU times " << config->simulationDuration / config->deltaT  << " " << p.x() << " " << p.y() << " " << p.z() << std::endl;
    //     std::cout << std::fixed << std::setprecision(4) <<"GPU times " << config->simulationDuration / config->deltaT  << " " << new_latlon.x() << " " << new_latlon.y() << std::endl;
    // }

    //Save
    std::vector<vtkSmartPointer<vtkPolyData>> polyDataList;
    std::vector<std::string> file_name_vec;
    int each_point_size = config->simulationDuration / config->recordT;
    int file_times = 0;
    for (auto i = 0; i < update_points.size(); i++) 
    {
        vec3 p = after_write_p[i];
        vtkSmartPointer<vtkPoints> vtk_points = vtkSmartPointer<vtkPoints>::New();
    	// 转为经纬度
    	vec2 new_latlon;
    	GeoConverter::convertXYZToLatLonDegree(p, new_latlon);
        vtk_points->InsertNextPoint(new_latlon.y(), new_latlon.x(), 10); // Assuming the z-coordinate is 0
        // Create a PolyData object and set the points
        vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
        polydata->SetPoints(vtk_points);
        polyDataList.push_back(polydata);

        if (((i + 1) % each_point_size) == 0 || i == update_points.size() - 1) 
        {
            std::string fileName = std::to_string(file_times) + ".vtp";
            file_name_vec.push_back(fileName);
            VTKFileManager::ConnectPointsToOneLine(polyDataList, fileName);
            polyDataList.clear();
            file_times++;

            // 这个点是该采样的最后一个路径点 p
            last_points.push_back(p);
        }
    }
   
    // 合并所有生成的 .vtp 文件
    VTKFileManager::MergeVTPFiles(file_name_vec, config->fileName);
    // std::cout << "last points size " << last_points.size() << std::endl;
    return last_points;
}

// void MPASOVisualizer::VisualizeFixedLayer_TimeVarying(int width, int height, ImageBuffer<double> *img1, ImageBuffer<double> *img2, float time1, float time2, float time, sycl::queue &sycl_Q)
// {
//     float t = (time - time1) / (time2 - time1);
//     sycl::buffer<double, 1> img_buf1(img1->mPixels.data(), sycl::range<1>(img1->mPixels.size()));
//     sycl::buffer<double, 1> img_buf2(img2->mPixels.data(), sycl::range<1>(img2->mPixels.size()));
//     sycl::buffer<int, 1> width_buf(&width, 1);
//     sycl::buffer<int, 1> height_buf(&height, 1);
//     sycl::buffer<float,1 > t_buf(&t, 1);
//     sycl_Q.submit([&](sycl::handler& cgh) 
//     {

// #pragma region sycl_acc_image
//         auto img_acc1 = img_buf1.get_access<sycl::access::mode::read_write>(cgh);
//         auto img_acc2 = img_buf2.get_access<sycl::access::mode::read_write>(cgh);
//         auto width_acc = width_buf.get_access<sycl::access::mode::read>(cgh);
//         auto height_acc = height_buf.get_access<sycl::access::mode::read>(cgh);
//         auto t_acc = t_buf.get_access<sycl::access::mode::read>(cgh);
// #pragma endregion sycl_acc_image

//         sycl::range<2> global_range((height + 7) / 8 * 8, (width + 7) / 8 * 8);
//         sycl::range<2> local_range(8, 8);  

//         cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), [=](sycl::nd_item<2> idx) 
//         {

//             int height_index = idx.get_global_id(0);
//             int width_index = idx.get_global_id(1);
//             int global_id = height_index * width_acc[0] + width_index;
                
//             if(height_index < height && width_index < width)
//             {
//                 vec3 img1_rgb; vec3 img2_rgb; vec3 final_rgb;
//                 GetPixel(img_acc1, width_acc[0], height_acc[0], height_index, width_index, img1_rgb);
//                 GetPixel(img_acc2, width_acc[0], height_acc[0], height_index, width_index, img2_rgb);
//                 final_rgb.x() = (1.0 - t_acc[0]) * img1_rgb.x() + t_acc[0] * img2_rgb.x();
//                 final_rgb.y() = (1.0 - t_acc[0]) * img1_rgb.y() + t_acc[0] * img2_rgb.y();
//                 final_rgb.z() = (1.0 - t_acc[0]) * img1_rgb.z() + t_acc[0] * img2_rgb.z();
//                 //final_rgb = img2_rgb;
//                 SetPixel(img_acc1, width_acc[0], height_acc[0], height_index, width_index, final_rgb);
//             }
//         });


//     });
//     sycl_Q.wait();
// }



std::vector<CartesianCoord>  MPASOVisualizer::TEST_VisualizeTrajectory(std::vector<CartesianCoord>& points, TrajectorySettings* config, sycl::queue& sycl_Q)
{
    std::vector<vec3> update_points;
    if (!update_points.empty()) update_points.clear();
    update_points.resize(points.size() * (config->simulationDuration / config->recordT));

    std::cout << "update_points size " << update_points.size() << std::endl;

    sycl::buffer<vec3> wirte_points_buf(update_points.data(), sycl::range<1>(update_points.size()));
    sycl::buffer<vec3> sample_points_buf(points.data(), sycl::range<1>(points.size()));
    sycl_Q.submit([&](sycl::handler& cgh) 
    {


        auto acc_wirte_points_buf = wirte_points_buf.get_access<sycl::access::mode::write>(cgh);
        auto acc_sample_points_buf = sample_points_buf.get_access<sycl::access::mode::read_write>(cgh);

        sycl::stream out(4096, 1024, cgh);
        int times = config->simulationDuration / config->deltaT;
        std::cout << "times " << times << std::endl;
        int each_points_size = config->simulationDuration / config->recordT;
        int recordT = config->recordT;
        int deltaT = config->deltaT;

        
        cgh.parallel_for(sycl::range<1>(points.size()), [=](sycl::item<1> item) 
        {

            int global_id = item[0];
           

            double runTime = 0.0;
            int save_times = 0;
            bool bFirstLoop = true;
            int base_idx = global_id * each_points_size;
            int update_points_idx = 0;

            // 1. 获取point position
            vec3 position; 
            vec3 new_position;
            double pos_x, pos_y, pos_z;


            for (auto times_i = 0; times_i < times; times_i++)
            {
                runTime += deltaT;  
                position = acc_sample_points_buf[global_id];
                
                // out << times_i << ":::/" << times << " " << position.x() << " " << position.y() << " " << position.z() << sycl::endl;
              

                double v_zonal = 2.0; double v_meridional = 0.0;
                vec3 current_velocity;
                GeoConverter::convertENUVelocityToXYZ(position, v_zonal, v_meridional, 0.0, current_velocity);
                

                double r = YOSEF_LENGTH(position);
                vec3 rotationAxis = computeRotationAxis(position, current_velocity);

                double speed = magnitude(current_velocity);
                double theta = (speed * deltaT) / r;
                double thetaInDegrees = theta * (180.0 / 3.141592653589793); // 将弧度转换为角度
                rotateAroundAxis(position, rotationAxis, thetaInDegrees, pos_x, pos_y, pos_z);
                
                new_position = {pos_x, pos_y, pos_z};
                acc_sample_points_buf[global_id] = new_position;
                    
                if ((int)runTime % recordT == 0)
                {
                    //save .//TODO
                    out << "save times " << save_times << " " << new_position.x() << " " << new_position.y() << " " << new_position.z() << sycl::endl;
                    save_times++;
                    int acc_wirte_pints_idx = base_idx + update_points_idx;
                    acc_wirte_points_buf[acc_wirte_pints_idx] = new_position;
                    update_points_idx = update_points_idx + 1;
                    
                }




            
               
                
                    

               
            
            
            
            }
        });
       
    });
    sycl_Q.wait();

    Debug("[VisualizeTrajectory]::Finished...");
    auto after_p = sample_points_buf.get_access<sycl::access::mode::read>(); 
    auto after_write_p = wirte_points_buf.get_access<sycl::access::mode::read>();
    std::vector<CartesianCoord> last_points;
   

    //Save
    vec3 p = after_write_p[update_points.size() - 1];
    vec2 new_latlon;
    GeoConverter::convertXYZToLatLonDegree(p, new_latlon);
    std::cout << "last point [lat lon] " << new_latlon.x() << " " << new_latlon.y() << std::endl;
    std::cout << "last point (x y z) " << p.x() << " " << p.y() << " " << p.z() << std::endl;
    
    
    
    return last_points;
}
