#include "GPU/SYCL/MPASOVisualizerSYCL.h"

#include "Common/TrajectoryCommon.h"
#include "GPU/SYCL/Kernel/SYCLKernel.h"
#include <tbb/parallel_for.h>

#define CHECK_VECTOR(vec, name) \
    if (vec.data() == nullptr || vec.size() == 0) { \
        std::cerr << "[Warning] Vector '" << name << "' is empty or nullptr!" << std::endl; \
    }

namespace MOPS::GPU::SYCLBackend {

void VisualizeFixedLayer(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, sycl::queue& sycl_Q)
{
    int width = config->imageSize.x();
    int height = config->imageSize.y();
    auto minLat = config->LatRange.x();
    auto maxLat = config->LatRange.y();
    auto minLon = config->LonRange.x();
    auto maxLon = config->LonRange.y();
    auto fixed_layer = config->FixedLayer;

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(mpasoF->mGrid.get());


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
    sycl::buffer<vec3, 1> vertexCoord_buf(mpasoF->mGrid->vertexCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->vertexCoord_vec.size()));
    sycl::buffer<vec3, 1> cellCoord_buf(mpasoF->mGrid->cellCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->cellCoord_vec.size()));
    sycl::buffer<size_t, 1> numberVertexOnCell_buf(mpasoF->mGrid->numberVertexOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->numberVertexOnCell_vec.size()));
    sycl::buffer<size_t, 1> verticesOnCell_buf(mpasoF->mGrid->verticesOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->verticesOnCell_vec.size()));
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


                // Determine whether it is on the mainland Determine whether it is on the mainland
                // 1.1 Calculate how many vertices are in this cell.
                auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                auto nan = std::numeric_limits<size_t>::max();
                //1.2 Find all candidate vertices
                size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
                SYCLKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx, MAX_VERTEX_NUM, max_edge, acc_verticesOnCell_buf);
                bool is_land = SYCLKernel::IsInMesh(cell_id, max_edge, current_position, acc_numberVertexOnCell_buf, acc_verticesOnCell_buf, acc_vertexCoord_buf);
                if (!is_land)
                {
                    SetPixel(img_acc, width_acc[0], height_acc[0], height_index, width_index, vec3_nan);
                    return;
                }

                vec3 imgValue = vec3_nan;

                if (true)
                {
                    //  Calculate the (velocity , coordinates and zTOP) of each Cell vertex.
                    vec3 current_cell_vertices_velocity[MAX_VERTEX_NUM];
                    vec3 current_verteices_positions[MAX_VERTEX_NUM];
                    double current_cell_vertices_ztop[MAX_VERTEX_NUM];
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

void VisualizeFixedDepth(MPASOField* mpasoF, VisualizationSettings* config, std::vector<ImageBuffer<double>>& img_vec, sycl::queue& sycl_Q)
{
    int width = config->imageSize.x();
    int height = config->imageSize.y();
    auto minLat = config->LatRange.x();
    auto maxLat = config->LatRange.y();
    auto minLon = config->LonRange.x();
    auto maxLon = config->LonRange.y();
    auto fixed_depth = -config->FixedDepth;

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(mpasoF->mGrid.get());

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
    sycl::buffer<vec3, 1> vertexCoord_buf(mpasoF->mGrid->vertexCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->vertexCoord_vec.size())); //
    sycl::buffer<vec3, 1> cellCoord_buf(mpasoF->mGrid->cellCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->cellCoord_vec.size()));
    sycl::buffer<size_t, 1> numberVertexOnCell_buf(mpasoF->mGrid->numberVertexOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->numberVertexOnCell_vec.size()));
    sycl::buffer<size_t, 1> verticesOnCell_buf(mpasoF->mGrid->verticesOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->verticesOnCell_vec.size()));             //
    sycl::buffer<size_t, 1> cellsOnVertex_buf(mpasoF->mGrid->cellsOnVertex_vec.data(), sycl::range<1>(mpasoF->mGrid->cellsOnVertex_vec.size()));
    sycl::buffer<size_t, 1> grid_info_buf(grid_info_vec.data(), sycl::range<1>(grid_info_vec.size()));

    sycl::buffer<vec3, 1> cellVertexVelocity_buf(mpasoF->mSol_Front->cellVertexVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexVelocity_vec.size()));
    sycl::buffer<double, 1> cellVertexZTop_buf(mpasoF->mSol_Front->cellVertexZTop_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexZTop_vec.size()));


    bool bDoubleAttributes = false;
    std::vector<std::string> attr_names;
    std::vector<sycl::buffer<double, 1>> attr_bufs;
    if (mpasoF->mSol_Front->mDoubleAttributes.size() > 1)
    {
        bDoubleAttributes = true;


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

        int attr_count = 0;
        std::array<sycl::accessor<double, 1, sycl::access::mode::read>, MAX_ATTRS> acc_attr_bufs;
        if (bDoubleAttributes)
        {

            attr_count = attr_bufs.size();

            for (int i = 0; i < attr_count; ++i) {
                acc_attr_bufs[i] = attr_bufs[i].get_access<sycl::access::mode::read>(cgh);
            }

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
            const int ACTUALL_VERTEX_SIZE       = (int)acc_grid_info_buf[3];
            const int TOTAY_ZTOP_LAYER = (int)acc_grid_info_buf[4];
            const int MAX_VERTLEVELS = 100;
            const int VERTLEVELS = (int)acc_grid_info_buf[4];
            const double DEPTH = depth_acc[0];
            auto nan = std::numeric_limits<size_t>::max();
            auto double_nan = std::numeric_limits<double>::quiet_NaN();
            vec3 vec3_nan = { double_nan, double_nan, double_nan };

            vec2 current_pixel = { height_index, width_index };
            CartesianCoord current_position;
            SphericalCoord current_latlon_r;
            GeoConverter::convertPixelToLatLonToRadians(width_acc[0], height_acc[0], minLat_acc[0], maxLat_acc[0], minLon_acc[0], maxLon_acc[0], current_pixel, current_latlon_r);
            GeoConverter::convertRadianLatLonToXYZ(current_latlon_r, current_position);
            int cell_id = acc_cellID_buf[global_id];

            auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
            size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
            if (cell_id < 0 || cell_id >= grid_cell_size)
            {
                out << "index " << cell_id << " is out of range\n" << sycl::endl;
            }
            SYCLKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx, MAX_VERTEX_NUM, max_edge, acc_verticesOnCell_buf);
            bool is_inMesh = SYCLKernel::IsInMesh(cell_id, max_edge, current_position, acc_numberVertexOnCell_buf, acc_verticesOnCell_buf, acc_vertexCoord_buf);
            if (!is_inMesh)
            {
                for (int i = 0; i < img_count; ++i)
                {
                    SetPixel(img_accs[i], width_acc[0], height_acc[0], height_index, width_index, vec3_nan);
                }
                return;
            }


            vec3 imgValue = vec3_nan;
            double current_point_ztop_vec[MAX_VERTLEVELS];

            vec3   vpos[MAX_VERTEX_NUM];
            double w[MAX_VERTEX_NUM];

            if (!SYCLKernel::GetCellVertexPos(vpos, current_cell_vertices_idx, MAX_VERTEX_NUM,
                                            current_cell_vertices_number, acc_vertexCoord_buf)) {
                imgValue = vec3_nan;
                for (int i=0;i<img_count;++i) SetPixel(img_accs[i], width, height, height_index, width_index, imgValue);
                return;
            }

            Interpolator::CalcPolygonWachspress(current_position, vpos, w, current_cell_vertices_number);

            const int ztop_levels = (int)(acc_cellVertexZTop_buf.get_range()[0] / ACTUALL_VERTEX_SIZE);

            for (int k = 0; k < ztop_levels; ++k) {
                double acc = 0.0;
                for (int v = 0; v < current_cell_vertices_number; ++v) {
                    int VID = current_cell_vertices_idx[v];
                    double ztop = acc_cellVertexZTop_buf[VID * ztop_levels + k];
                    acc += w[v] * ztop;
                }
                current_point_ztop_vec[k] = acc;
            }


            for (int k = 1; k < ztop_levels; ++k) {
                if (current_point_ztop_vec[k] > current_point_ztop_vec[k-1]) {
                    current_point_ztop_vec[k] = current_point_ztop_vec[k-1] - 1e-9;
                }
            }


            double z_surf = current_point_ztop_vec[0];
            double z_bot  = current_point_ztop_vec[ztop_levels - 1];
            if (z_surf < z_bot) { double t = z_surf; z_surf = z_bot; z_bot = t; }

            double epsd = sycl::fmax(1e-6, 1e-8 * sycl::fabs(z_surf - z_bot));

            if (!(DEPTH <= z_surf + epsd && DEPTH >= z_bot - epsd)) {

                imgValue = vec3_nan;
                for (int i=0;i<img_count;++i) SetPixel(img_accs[i], width, height, height_index, width_index, imgValue);
                return;
            }


            int local_layer = -1;
            for (int k = 1; k < ztop_levels; ++k) {
                double topI = current_point_ztop_vec[k-1];
                double botI = current_point_ztop_vec[k];
                if (topI < botI) { double t = topI; topI = botI; botI = t; }
                if (DEPTH <= topI + 1e-8 && DEPTH >= botI - 1e-8) { local_layer = k; break; }
            }
            if (DEPTH <= current_point_ztop_vec[0])
            {
                local_layer = 0;
            }
            if (local_layer < 0)
            {

                imgValue = vec3_nan;
                for (int i=0;i<img_count;++i) SetPixel(img_accs[i], width, height, height_index, width_index, imgValue);
                return;
            }

            double topI = current_point_ztop_vec[local_layer-1];
            double botI = current_point_ztop_vec[local_layer];
            if (topI < botI) { double tmp = topI; topI = botI; botI = tmp; }

            double denom  = topI - botI;
            double tparam = (denom > 1e-12) ? (DEPTH - botI) / denom : 0.5;

            const int vel_levels = (int)(acc_cellVertexVelocity_buf.get_range()[0] / ACTUALL_VERTEX_SIZE);
            int j     = sycl::clamp(local_layer - 1, 0, vel_levels - 1);
            int j_bot = sycl::min(j + 1, vel_levels - 1);
            int j_top = j;

            vec3 v_top = SYCLKernel::CalcVelocity(current_cell_vertices_idx, w,
                        MAX_VERTEX_NUM, current_cell_vertices_number, vel_levels, j_top, acc_cellVertexVelocity_buf);
            vec3 v_bot = SYCLKernel::CalcVelocity(current_cell_vertices_idx, w,
                        MAX_VERTEX_NUM, current_cell_vertices_number, vel_levels, j_bot, acc_cellVertexVelocity_buf);


            double mtop = MOPS_LENGTH(v_top), mbot = MOPS_LENGTH(v_bot);
            vec3 final_vel;
            if (mtop < 1e-12 && mbot < 1e-12)      final_vel = vec3{0.0,0.0,0.0};
            else if (mtop < 1e-12)                 final_vel = v_bot;
            else if (mbot < 1e-12)                 final_vel = v_top;
            else                                    final_vel = (1.0 - tparam) * v_bot + tparam * v_top;

            double u_east, v_north;
            GeoConverter::convertXYZVelocityToENU(current_position, final_vel, u_east, v_north);
            double spd = sycl::sqrt(u_east*u_east + v_north*v_north);
            vec3 current_point_velocity_enu = {u_east, v_north, spd};


            vec3 current_point_attr_value = {0.0, 0.0, 0.0};
            if (bDoubleAttributes)
            {

                if (attr_count >= 1) {
                    const int attr_levels = (int)(acc_attr_bufs[0].get_range()[0] / ACTUALL_VERTEX_SIZE);
                    int aj     = sycl::clamp(local_layer - 1, 0, attr_levels - 1);
                    int aj_bot = sycl::min(aj + 1, attr_levels - 1);

                    double a0 = SYCLKernel::CalcAttribute(current_cell_vertices_idx, w,
                                MAX_VERTEX_NUM, current_cell_vertices_number, attr_levels, aj, acc_attr_bufs[0]);
                    current_point_attr_value.x() = a0;
                }
                if (attr_count >= 2) {
                    const int attr_levels1 = (int)(acc_attr_bufs[1].get_range()[0] / ACTUALL_VERTEX_SIZE);
                    int aj1 = sycl::clamp(local_layer - 1, 0, attr_levels1 - 1);
                    double a1 = SYCLKernel::CalcAttribute(current_cell_vertices_idx, w,
                                MAX_VERTEX_NUM, current_cell_vertices_number, attr_levels1, aj1, acc_attr_bufs[1]);
                    current_point_attr_value.y() = a1;
                }
            }


            SetPixel(img_accs[0], width, height, height_index, width_index, current_point_velocity_enu);
            if (bDoubleAttributes)
                SetPixel(img_accs[1], width, height, height_index, width_index, current_point_attr_value);


        }

        });
    });

    sycl_Q.wait();
}

void VisualizeFixedLatitude(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, sycl::queue& sycl_Q)
{

    int width = config->imageSize.x();
    int height = config->imageSize.y();
    auto minDepth = config->DepthRange.x();
    auto maxDepth = config->DepthRange.y();
    auto minLon = config->LonRange.x();
    auto maxLon = config->LonRange.y();
    auto fixed_lat = config->FixedLatitude;

    const auto& refBottomDepth = mpasoF->mGrid->cellRefBottomDepth_vec;

    minDepth = refBottomDepth[0];
    maxDepth = refBottomDepth[refBottomDepth.size() - 1];

    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(mpasoF->mGrid.get());

   std::vector<int> cell_id_vec;
   cell_id_vec.resize(width * height);

    double i_step = (height > 1) ? (maxDepth - minDepth) / (height - 1) : 0.0;
    double j_step = (width  > 1) ? (maxLon   - minLon)   / (width  - 1) : 0.0;
    for (int ih = 0; ih < height; ++ih)
    {
        double i = -1 * minDepth + ih * i_step;
        for (int jw = 0; jw < width; ++jw)
        {
            double j = minLon + jw * j_step;
            auto Lat = fixed_lat;
            auto Lon = j;

            SphericalCoord latlon_r = vec2(Lat * (M_PI / 180.0f), Lon * (M_PI / 180.0f));
            CartesianCoord current_position;
            GeoConverter::convertRadianLatLonToXYZ(latlon_r, current_position);

            int cell_id_value = -1;
            mpasoF->mGrid->searchKDT(current_position, cell_id_value);
            int global_id = ih * width + jw;
            cell_id_vec[global_id] = cell_id_value;
        }
    }

    Debug("MPASOVisualizer::Finished KD Tree Search....");


    auto double_nan = std::numeric_limits<double>::quiet_NaN();
    vec3 vec3_nan = { double_nan, double_nan, double_nan };

    for (int ih = 0; ih < height; ++ih)
    {
        double i = minDepth + ih * i_step;
        double DEPTH = -std::abs(i);
        for (int jw = 0; jw < width; ++jw)
        {
            double j = minLon + jw * j_step;
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

            int cell_id = cell_id_vec[ih * width + jw];
            if (cell_id < 0) { img->setPixel(ih, jw, vec3_nan); continue; }

            // 2. Determine whether it is on the mainland
            std::vector<size_t> current_cell_vertices_idx;
            bool is_land = mpasoF->isOnOcean(position, cell_id, current_cell_vertices_idx);
            if (is_land)
            {
                img->setPixel(ih, jw, vec3_nan);
                continue;
            }

            // This point is on the Cell

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

            int nVert = 60;
            current_point_ztop_vec.resize(nVert);

            for (int k = 0; k < nVert; ++k)
            {
                double current_point_ztop_in_layer = 0.0;
                for (int v_idx = 0; v_idx < current_cell_vertices_number; v_idx++)
                {
                    auto VID = current_cell_vertices_idx[v_idx];
                    double ztop = mpasoF->mSol_Front->cellVertexZTop_vec[VID * nVert + k];
                    current_point_ztop_in_layer += current_cell_vertex_weight[v_idx] * ztop;
                }
                current_point_ztop_vec[k] = current_point_ztop_in_layer;
            }


            int layer = -1;
            const double EPSILON = 1e-6;

            if (DEPTH > current_point_ztop_vec[0] + EPSILON || DEPTH < current_point_ztop_vec[nVert-1] - EPSILON)
            {
                img->setPixel(ih, jw, vec3_nan);
                continue;
            }

            for (int k = 1; k < nVert; ++k)
            {
                double z_up = current_point_ztop_vec[k-1];
                double z_dn = current_point_ztop_vec[k];
                if (DEPTH <= z_up + EPSILON && DEPTH >= z_dn - EPSILON)
                {
                    layer = k;
                    break;
                }
            }

            if (layer == -1)
            {
                img->setPixel(ih, jw, vec3_nan);
                continue;
            }

            double ztop_layer_dn = current_point_ztop_vec[layer];
            double ztop_layer_up = current_point_ztop_vec[layer - 1];

            double denom = (ztop_layer_dn - ztop_layer_up);
            if (std::abs(denom) < 1e-30) { img->setPixel(ih, jw, vec3_nan); continue; }

            double t = (DEPTH - ztop_layer_up) / denom;

            // 7) Perform horizontal interpolation to get velocities at two layers, then interpolate vertically using t
            vec3 current_point_vel_up = { 0.0, 0.0, 0.0 };
            vec3 current_point_vel_dn = { 0.0, 0.0, 0.0 };

            for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
            {
                auto VID = current_cell_vertices_idx[v_idx];

                vec3 vel_up = mpasoF->mSol_Front->cellVertexVelocity_vec[VID * nVert + (layer - 1)];
                vec3 vel_dn = mpasoF->mSol_Front->cellVertexVelocity_vec[VID * nVert + layer];

                current_point_vel_up.x() += current_cell_vertex_weight[v_idx] * vel_up.x();
                current_point_vel_up.y() += current_cell_vertex_weight[v_idx] * vel_up.y();
                current_point_vel_up.z() += current_cell_vertex_weight[v_idx] * vel_up.z();

                current_point_vel_dn.x() += current_cell_vertex_weight[v_idx] * vel_dn.x();
                current_point_vel_dn.y() += current_cell_vertex_weight[v_idx] * vel_dn.y();
                current_point_vel_dn.z() += current_cell_vertex_weight[v_idx] * vel_dn.z();
            }

            vec3 final_vel;
            final_vel.x() = (1.0 - t) * current_point_vel_up.x() + t * current_point_vel_dn.x();
            final_vel.y() = (1.0 - t) * current_point_vel_up.y() + t * current_point_vel_dn.y();
            final_vel.z() = (1.0 - t) * current_point_vel_up.z() + t * current_point_vel_dn.z();

            // 8) TO ENU
            double zional_velocity, merminoal_velicity;
            GeoConverter::convertXYZVelocityToENU(position, final_vel, zional_velocity, merminoal_velicity);
            vec3 current_point_velocity_enu = { zional_velocity, merminoal_velicity, 0.0 };

            img->setPixel(ih, jw, current_point_velocity_enu);
        }
    }
}

std::vector<TrajectoryLine> StreamLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, sycl::queue& sycl_Q)
{
    
    std::vector<vec3> stable_points = points; 

    auto host_buffers = MOPS::Common::InitTrajectoryOutputBuffers(
        stable_points.size(),
        config->simulationDuration,
        config->recordT,
        false);
    std::vector<vec3>& update_points = host_buffers.points;
    std::vector<vec3>& update_vels = host_buffers.velocities;

   
    std::vector<float> effective_depths = MOPS::Common::BuildEffectiveDepths(stable_points, config, "StreamLine");
    std::vector<TrajectoryLine> trajectory_lines = MOPS::Common::InitTrajectoryLines(stable_points, effective_depths, config);

   
    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(mpasoF->mGrid.get());

#pragma region sycl_buffer_grid
    sycl::buffer<vec3, 1>   vertexCoord_buf(mpasoF->mGrid->vertexCoord_vec.data(),                  sycl::range<1>(mpasoF->mGrid->vertexCoord_vec.size())); 
    sycl::buffer<vec3, 1>   cellCoord_buf(mpasoF->mGrid->cellCoord_vec.data(),                      sycl::range<1>(mpasoF->mGrid->cellCoord_vec.size()));       
    sycl::buffer<size_t, 1> numberVertexOnCell_buf(mpasoF->mGrid->numberVertexOnCell_vec.data(),    sycl::range<1>(mpasoF->mGrid->numberVertexOnCell_vec.size()));
    sycl::buffer<size_t, 1> verticesOnCell_buf(mpasoF->mGrid->verticesOnCell_vec.data(),            sycl::range<1>(mpasoF->mGrid->verticesOnCell_vec.size()));             
    sycl::buffer<size_t, 1> cellsOnVertex_buf(mpasoF->mGrid->cellsOnVertex_vec.data(),              sycl::range<1>(mpasoF->mGrid->cellsOnVertex_vec.size()));
    sycl::buffer<size_t, 1> cells_onCell_buf(mpasoF->mGrid->cellsOnCell_vec.data(),                 sycl::range<1>(mpasoF->mGrid->cellsOnCell_vec.size()));
    sycl::buffer<size_t, 1> grid_info_buf(grid_info_vec.data(),                                     sycl::range<1>(grid_info_vec.size()));
#pragma endregion sycl_buffer_grid


#pragma region sycl_buffer_velocity
    sycl::buffer<vec3, 1>   cellVertexVelocity_buf(mpasoF->mSol_Front->cellVertexVelocity_vec.data(),   sycl::range<1>(mpasoF->mSol_Front->cellVertexVelocity_vec.size()));
    sycl::buffer<double, 1> cellVertexZTop_buf(mpasoF->mSol_Front->cellVertexZTop_vec.data(),           sycl::range<1>(mpasoF->mSol_Front->cellVertexZTop_vec.size()));
    sycl::buffer<double, 1> cellVertexVertVelocity_buf(mpasoF->mSol_Front->cellVertexVertVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexVertVelocity_vec.size()));
#pragma endregion sycl_buffer_velocity   

#pragma region sycl_buffer_particle_depths
    // Per-particle depth buffer for GPU kernel
    sycl::buffer<float, 1> particle_depths_buf(effective_depths.data(), sycl::range<1>(effective_depths.size()));
#pragma endregion sycl_buffer_particle_depths

    sycl::buffer<int, 1>    cellID_buf(default_cell_id.data(),      sycl::range<1>(default_cell_id.size()));
    sycl::buffer<vec3, 1>   wirte_points_buf(update_points.data(),  sycl::range<1>(update_points.size()));
    sycl::buffer<vec3, 1>   write_vels_buf(update_vels.data(),      sycl::range<1>(update_vels.size()));
    sycl::buffer<vec3, 1>   sample_points_buf(stable_points.data(), sycl::range<1>(stable_points.size()));
    sycl_Q.submit([&](sycl::handler& cgh) 
    {

#pragma region sycl_acc_grid
        auto acc_cellID_buf             = cellID_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_vertexCoord_buf        = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf          = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnCell_buf     = verticesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnVertex_buf      = cellsOnVertex_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cells_onCell_buf       = cells_onCell_buf.get_access<sycl::access::mode::read>(cgh);
        int grid_cell_size              = mpasoF->mGrid->mCellsSize;
        auto acc_grid_info_buf          = grid_info_buf.get_access<sycl::access::mode::read>(cgh);
#pragma endregion sycl_acc_grid

#pragma region sycl_acc_velocity
        auto acc_cellVertexVelocity_buf = cellVertexVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexZTop_buf     = cellVertexZTop_buf.get_access<sycl::access::mode::read>(cgh);
    auto acc_cellVertexVertVelocity_buf = cellVertexVertVelocity_buf.get_access<sycl::access::mode::read>(cgh);
#pragma endregion sycl_acc_velocity

        auto acc_def_cell_id_buf    = cellID_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_wirte_points_buf   = wirte_points_buf.get_access<sycl::access::mode::write>(cgh);
        auto acc_write_vels_buf     = write_vels_buf.get_access<sycl::access::mode::write>(cgh);
        auto acc_sample_points_buf  = sample_points_buf.get_access<sycl::access::mode::read_write>(cgh);

        // Access per-particle depth buffer
        auto acc_particle_depths_buf = particle_depths_buf.get_access<sycl::access::mode::read_write>(cgh);

        sycl::stream out(4096, 256, cgh);
        int times               = config->simulationDuration / config->deltaT;
        int each_points_size    = config->simulationDuration / config->recordT;
        int recordT             = config->recordT;
        int dt_sign             = config->directionType == MOPS::CalcDirection::kForward ? 1 : -1;
        int deltaT              = dt_sign * config->deltaT;
        // float config_depth   = config->depth;  // REMOVED: now using per-particle depths
        bool bEulerMethod       = config->methodType == MOPS::CalcMethodType::kEuler ? true : false;

        
       
        cgh.parallel_for(sycl::range<1>(points.size()), [=](sycl::item<1> item) 
        {
            int global_id = item[0];
            // load from grid info
            const int ACTUALL_CELL_SIZE         = (int)acc_grid_info_buf[0];
            const int ACTUALL_EDGE_SIZE         = (int)acc_grid_info_buf[1];
            const int ACTUALL_MAX_EDGE_SIZE     = (int)acc_grid_info_buf[2];
            const int ACTUALL_VERTEX_SIZE       = (int)acc_grid_info_buf[3];
            const int ACTUALL_ZTOP_LAYER        = (int)acc_grid_info_buf[4];
            const int ACTUALL_ZTOP_LAYER_P1     = (int)acc_grid_info_buf[5];
            // default parameters
            const int MAX_VERTEX_NUM                    = 20;
            const int MAX_CELL_NEIGHBOR_NUM             = 21;
            const int MAX_VERTICAL_LEVEL_NUM            = 80;
            const int MAX_VERTEX_NEIGHBOR_NUM           = 3;


            // Use per-particle depth (negative because MPASO uses z-up coordinate)
            double current_depth    = -1.0 * (double)acc_particle_depths_buf[global_id];
            double runTime          = 0.0;
            bool bFirstLoop         = true;
            bool bFirstVel          = true;
            bool bOptimize          = true;
            
            
            int base_idx            = global_id * each_points_size;
            int update_points_idx   = 0;
            int save_times          = 0;

            vec3 sample_point_position, new_position;
            int cell_id = -1;
            int cell_neig_vec[MAX_CELL_NEIGHBOR_NUM];
            int cell_id_vec[MAX_VERTEX_NUM];

            enum : int {
                R_NONE              = 0,
                R_FIND_FAIL         = 1,
                R_NOT_IN_MESH       = 2,
                R_NO_LAYER          = 3,
                R_ZERO_DENOM        = 4,
                R_VEL1_ZERO         = 5,
                R_VEL2_ZERO         = 6,
                R_FINAL_ZERO        = 7,
                R_VLA_FAIL          = 8
            };

            auto RET0 = [&](int reason) -> vec3 {
                if (global_id==-1 /*or other cell id*/ ) {
                    out << "[ZERO] gid="<<global_id<<" step="<< (int)(runTime/deltaT)
                        <<" reason="<<reason<< sycl::endl;
                }
                return vec3(0.0);
            };

          

            struct V1_Out {
                vec3 h_vel;
                double v_vel;
            };

            auto CalcVelocityAt = [&](const vec3& pos, int& cid) -> V1_Out {
                const int cell_id = cid; 

                
                // 1. check in mesh
                vec3 current_position = pos;
                auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
                SYCLKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx, MAX_VERTEX_NUM, ACTUALL_MAX_EDGE_SIZE, acc_verticesOnCell_buf);
                
                bool is_inMesh = SYCLKernel::IsInMesh(cell_id, ACTUALL_MAX_EDGE_SIZE, current_position, acc_numberVertexOnCell_buf, acc_verticesOnCell_buf, acc_vertexCoord_buf);
                // True:    in mesh -> in ocean
                // False:   Not in mesh -> in land
                if (!is_inMesh)
                {
                    return { RET0(R_NOT_IN_MESH), 0.0 };
                }

                
                // 2. Calc ztop at current position
                double current_point_ztop_vec[MAX_VERTICAL_LEVEL_NUM];
                double current_cell_vertex_weight[MAX_VERTEX_NUM];
                vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
                
                bool rc = SYCLKernel::GetCellVertexPos(current_cell_vertex_pos, current_cell_vertices_idx, MAX_VERTEX_NUM, current_cell_vertices_number, acc_vertexCoord_buf);
                if (!rc) return { RET0(R_VLA_FAIL), 0.0 };
                // washpress
                Interpolator::CalcPolygonWachspress(current_position, current_cell_vertex_pos, current_cell_vertex_weight, current_cell_vertices_number);

                for (int k = 0; k < ACTUALL_ZTOP_LAYER; ++k) 
                {
                    double current_point_ztop_in_layer = 0.0;
                    double current_cell_vertex_ztop[MAX_VERTEX_NUM];
                    for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx) 
                    {
                        int VID = current_cell_vertices_idx[v_idx];
                        if (VID < 0 || VID >= ACTUALL_VERTEX_SIZE) return { RET0(R_VLA_FAIL), 0.0 };
                        double ztop = acc_cellVertexZTop_buf[VID * ACTUALL_ZTOP_LAYER + k];
                        current_cell_vertex_ztop[v_idx] = ztop;
                    }
                    for (int v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
                        current_point_ztop_in_layer += current_cell_vertex_weight[v_idx] * current_cell_vertex_ztop[v_idx];
                    current_point_ztop_vec[k] = current_point_ztop_in_layer;
                }

                for (int k = 1; k < ACTUALL_ZTOP_LAYER; ++k)
                {
                    if (current_point_ztop_vec[k] > current_point_ztop_vec[k-1]) 
                    {
                        current_point_ztop_vec[k] = current_point_ztop_vec[k-1] - 1e-9;
                    }    
                } 
                    
                const double eps = 1e-8;                   
                int local_layer = -1;
                /*
                    topI < fixexd_depth < botI -> return k
                    fixed_depth > current_point_ztop_vec[0] -> return 0
                    fixed_depth < current_point_ztop_vec[ACTUALL_ZTOP_LAYER-1] -> return ACTUALL_ZTOP_LAYER - 1
                */
                if (bOptimize == false)
                {
                    bool bSkip_Loop_Find = false;
                    if (current_depth > current_point_ztop_vec[0] + eps) 
                    {
                        local_layer = 0;
                        bSkip_Loop_Find = true;
                    }
                    else if (current_depth < current_point_ztop_vec[ACTUALL_ZTOP_LAYER - 1] - eps) 
                    {
                        local_layer = ACTUALL_ZTOP_LAYER - 1;
                        bSkip_Loop_Find = true;
                    }
                    if (!bSkip_Loop_Find)
                    {
                        for (int k = 1; k < ACTUALL_ZTOP_LAYER; ++k) 
                        {
                            double topI = current_point_ztop_vec[k-1];
                            double botI = current_point_ztop_vec[k];
                            if (current_depth <= topI + eps && current_depth >= botI - eps) 
                            {
                                local_layer = k; 
                                break;
                            }
                        }
                    }
                }
                else
                {
                    if (current_depth > current_point_ztop_vec[0] + eps) 
                    {
                        local_layer = 1;
                    } 
                    else if (current_depth < current_point_ztop_vec[ACTUALL_ZTOP_LAYER - 1] - eps) 
                    {
                        local_layer = ACTUALL_ZTOP_LAYER - 1;
                    } 
                    else 
                    {
                        // Binary Search: in [1, L-1] find mid s.t.z[mid-1] ≥ x ≥ z[mid]
                        int lo = 1;
                        int hi = ACTUALL_ZTOP_LAYER - 1;
                        int ans = 1;  
                        while (lo <= hi) 
                        {
                            int mid = (lo + hi) >> 1;
                            double topI = current_point_ztop_vec[mid - 1];
                            double botI = current_point_ztop_vec[mid];

                            if (current_depth <= topI + eps && current_depth >= botI - eps) {
                                ans = mid;
                                break;
                            }
                           
                            if (current_depth > topI + eps) 
                            {
                                hi = mid - 1;
                            } 
                            else
                            { 
                                lo = mid + 1;
                            }
                        }

                        if (ans < 1) ans = 1;
                        if (ans > ACTUALL_ZTOP_LAYER - 1) ans = ACTUALL_ZTOP_LAYER - 1;
                        local_layer = ans;
                    }
                }
                
                
                if (local_layer < 0)
                {
                    // out << "DEBUG: Particle " << global_id << " depth layer not found. " 
                    // << "fixed_depth=" << fixed_depth 
                    // << " ztop_surface=" << current_point_ztop_vec[0] 
                    // << " ztop_bottom=" << current_point_ztop_vec[ztop_levels-1] << sycl::endl;
                    return { RET0(R_NO_LAYER), 0.0 };
                }
                else
                {
                    // Calc velocity by interpolation
                    double ztop_dn, ztop_up;
                    ztop_dn = current_point_ztop_vec[local_layer];
                    ztop_up = current_point_ztop_vec[local_layer - 1];

                    double x = current_depth;
                    x = sycl::max(ztop_dn, sycl::min(x, ztop_up));
                    double denom = ztop_up - ztop_dn;
                    if (sycl::fabs(denom) < 1e-12) return { RET0(R_ZERO_DENOM), 0.0 };
                    double t = (x - ztop_dn) / denom;
                    // double current_point_ztop;
                    // current_point_ztop = t * ztop_layer1 + (1 - t) * ztop_layer2;
                    
                    vec3 final_vel;
                    vec3 current_point_vel_dn = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer, acc_cellVertexVelocity_buf);
                
                    vec3 current_point_vel_up = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer - 1, acc_cellVertexVelocity_buf);
                    
                    double vel_dn_mag = MOPS_LENGTH(current_point_vel_dn);
                    double vel_up_mag = MOPS_LENGTH(current_point_vel_up);

                    if (vel_dn_mag < 1e-12 ) return { RET0(R_VEL1_ZERO), 0.0 };
                    if (vel_up_mag < 1e-12 ) return { RET0(R_VEL2_ZERO), 0.0 };

                    final_vel.x() = t * current_point_vel_up.x() + (1 - t) * current_point_vel_dn.x();
                    final_vel.y() = t * current_point_vel_up.y() + (1 - t) * current_point_vel_dn.y();
                    final_vel.z() = t * current_point_vel_up.z() + (1 - t) * current_point_vel_dn.z();

                    vec3 current_velocity = final_vel;

                    double vel_mag = MOPS_LENGTH(current_velocity);
                    if (vel_mag < 1e-12)    return { RET0(R_FINAL_ZERO), 0.0 };

                    int dn_if = local_layer;
                    int up_if = (local_layer > 0) ? (local_layer - 1) : 0;

                    double w_dn = SYCLKernel::CalcAttribute(
                        current_cell_vertices_idx, current_cell_vertex_weight,
                        MAX_VERTEX_NUM, current_cell_vertices_number,
                        ACTUALL_ZTOP_LAYER_P1, dn_if,
                        acc_cellVertexVertVelocity_buf);

                    double w_up = SYCLKernel::CalcAttribute(
                        current_cell_vertices_idx, current_cell_vertex_weight,
                        MAX_VERTEX_NUM, current_cell_vertices_number,
                        ACTUALL_ZTOP_LAYER_P1, up_if,
                        acc_cellVertexVertVelocity_buf);

                    double current_vertical_velocity = t * w_up + (1.0 - t) * w_dn;
                    return { current_velocity, current_vertical_velocity };
                }
                
            };

            if (bFirstLoop == false)
            {
                if (cell_id < 0 || cell_id >= ACTUALL_CELL_SIZE) 
                {
                    out << "[Error] cell_id out of range: " << cell_id << sycl::endl;
                    return;
                }
            }

            for (auto times_i = 0; times_i < times; times_i++)
            {
                runTime += sycl::abs(deltaT);
                sample_point_position = acc_sample_points_buf[global_id];
                current_depth = -1.0 * (double)acc_particle_depths_buf[global_id];
                
                
                int firtst_cell_id = acc_def_cell_id_buf[global_id];
                
                if (bFirstLoop)
                {
                    bFirstLoop = false;
                    cell_id = firtst_cell_id;
                    // FindContainingCell(position, cell_id);
                    auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                    SYCLKernel::GetCellNeighborsIdx(cell_id, current_cell_vertices_number, cell_neig_vec, MAX_VERTEX_NUM, ACTUALL_MAX_EDGE_SIZE, acc_cells_onCell_buf);
                    int acc_wirte_pints_idx = base_idx + 0;
                    acc_wirte_points_buf[acc_wirte_pints_idx] = sample_point_position;
                }
                else
                {
                    auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                    double max_len = std::numeric_limits<double>::max();
                    for (auto idx = 0; idx < current_cell_vertices_number + 1; idx++)
                    {
                        auto CID = cell_neig_vec[idx];
                        if (CID < 0 || CID >= ACTUALL_CELL_SIZE) continue;
                        vec3 cell_center_position = acc_cellCoord_buf[CID];
                        double len = MOPS_LENGTH(cell_center_position - sample_point_position);
                        if (len < max_len)
                        {
                            max_len = len;
                            cell_id = CID;
                        }
                    }
                    SYCLKernel::GetCellNeighborsIdx(cell_id, current_cell_vertices_number, cell_neig_vec, MAX_VERTEX_NUM, ACTUALL_MAX_EDGE_SIZE, acc_cells_onCell_buf);
                    

                }

                vec3 current_position = sample_point_position;
                vec3 current_horizontal_velocity;
                double current_vertical_velocity;
                double r = MOPS_LENGTH(current_position);
                vec3 rk4_next_position = current_position;

                auto AdvectOnSphere = [&](const vec3& pos, const vec3& vel, double dt_local) -> vec3 {
                    const double rr = MOPS_LENGTH(pos);
                    const double speed_local = MOPS_LENGTH(vel);
                    if (rr < 1e-12 || speed_local < 1e-12) return pos;
                    vec3 axis = SYCLKernel::CalcRotationAxis(pos, vel);
                    double theta = (speed_local * dt_local) / rr;
                    return SYCLKernel::CalcPositionAfterRotation(pos, axis, theta);
                };

                // Euler method
                if (bEulerMethod)
                {
                    auto state = CalcVelocityAt(current_position, cell_id);
                    current_horizontal_velocity = state.h_vel;
                    current_vertical_velocity = state.v_vel;
                    // debug
                    if (bFirstVel)
                    {
                        out << "[First Vel] gid="<<global_id
                            <<" cell_id="<<cell_id
                            <<" vel=("<<current_horizontal_velocity.x()<<","<<current_horizontal_velocity.y()<<","<<current_horizontal_velocity.z()<<")"
                            << sycl::endl;
                    }
                }
                else
                {
                    // Runge-Kutta 4th order method
                    double dt = static_cast<double>(deltaT);
                    auto s1 = CalcVelocityAt(current_position, cell_id);
                    vec3 k1 = s1.h_vel;
                    double v1 = s1.v_vel;
                    // p + 0.5 * k1 Δt   
                    vec3 p2   = AdvectOnSphere(current_position, k1, dt * 0.5);
                    int  cid2 = cell_id;
                    auto s2 = CalcVelocityAt(p2, cid2);
                    vec3 k2 = s2.h_vel;
                    double v2 = s2.v_vel;
                    // p + 0.5 * k2 Δt
                    vec3 p3   = AdvectOnSphere(current_position, k2, dt * 0.5);
                    int  cid3 = cell_id;
                    auto s3 = CalcVelocityAt(p3, cid3);
                    vec3 k3 = s3.h_vel;
                    double v3 = s3.v_vel;
                    // p + 1.0 * k3 Δt
                    vec3 p4   = AdvectOnSphere(current_position, k3, dt);
                    int  cid4 = cell_id;
                    auto s4 = CalcVelocityAt(p4, cid4);
                    vec3 k4 = s4.h_vel;
                    double v4 = s4.v_vel;
                    current_horizontal_velocity = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
                    current_vertical_velocity = (v1 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0;

                    // Strict RK4 endpoint on position: x_{n+1} = x_n + dt * (k1 + 2k2 + 2k3 + k4) / 6,
                    // then project back to the current sphere radius before vertical-radius update.
                    vec3 x_trial = current_position + current_horizontal_velocity * dt;
                    double x_trial_len = MOPS_LENGTH(x_trial);
                    if (x_trial_len > 1e-12) {
                        rk4_next_position = (x_trial / x_trial_len) * r;
                    } else {
                        rk4_next_position = current_position;
                    }
                }
                
                if (bEulerMethod)
                {
                    vec3 rotationAxis = SYCLKernel::CalcRotationAxis(current_position, current_horizontal_velocity);
                    double speed = MOPS_LENGTH(current_horizontal_velocity);
                    double theta_rad = (speed * deltaT) / r;
                    new_position = SYCLKernel::CalcPositionAfterRotation(current_position, rotationAxis, theta_rad);
                }
                else
                {
                    new_position = rk4_next_position;
                }

                const double old_depth = (double)acc_particle_depths_buf[global_id];
                double new_depth = old_depth - current_vertical_velocity * (double)deltaT;
                new_depth = sycl::fmax(0.0, new_depth);
                double r_new = sycl::fmax(1.0, r + current_vertical_velocity * (double)deltaT);
                acc_particle_depths_buf[global_id] = (float)new_depth;
                new_position = sycl::normalize(new_position) * r_new;
               
                if (bFirstVel)
                {
                    bFirstVel = false;
                    int acc_wirte_pints_idx = base_idx + 0;
                    acc_write_vels_buf[acc_wirte_pints_idx] = current_horizontal_velocity;
                }

                acc_sample_points_buf[global_id] = new_position;
                    
                if ((int)runTime % recordT == 0)
                {
                    int acc_wirte_pints_idx = base_idx + update_points_idx;
                    acc_wirte_points_buf[acc_wirte_pints_idx] = new_position;
                    acc_write_vels_buf[acc_wirte_pints_idx] = current_horizontal_velocity;
                    update_points_idx = update_points_idx + 1;
                }
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

    size_t each_point_size = config->simulationDuration / config->recordT;
    size_t total_points = update_points.size();
    auto clean_traj = MOPS::Common::FinalizeTrajectoryLines(
        trajectory_lines,
        after_write_p,
        after_write_v,
        each_point_size,
        total_points);
    trajectory_lines.clear();
    return clean_traj;
    // return trajectory_lines;
}


























std::vector<TrajectoryLine> PathLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, sycl::queue& sycl_Q)
{
    
    std::vector<vec3> stable_points = points; 

    auto host_buffers = MOPS::Common::InitTrajectoryOutputBuffers(
        stable_points.size(),
        config->simulationDuration,
        config->recordT,
        true);
    std::vector<vec3>& update_points = host_buffers.points;
    std::vector<vec3>& update_vels = host_buffers.velocities;
    std::vector<vec3>& update_attrs = host_buffers.attrs;

    std::vector<float> effective_depths = MOPS::Common::BuildEffectiveDepths(stable_points, config, "PathLine");
    std::vector<TrajectoryLine> trajectory_lines = MOPS::Common::InitTrajectoryLines(stable_points, effective_depths, config);

   
    std::vector<size_t> grid_info_vec = MOPS::Common::BuildGridInfo(mpasoF->mGrid.get());

#pragma region sycl_buffer_grid
    sycl::buffer<vec3, 1>   vertexCoord_buf(mpasoF->mGrid->vertexCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->vertexCoord_vec.size())); 
    sycl::buffer<vec3, 1>   cellCoord_buf(mpasoF->mGrid->cellCoord_vec.data(), sycl::range<1>(mpasoF->mGrid->cellCoord_vec.size()));       
    sycl::buffer<size_t, 1> numberVertexOnCell_buf(mpasoF->mGrid->numberVertexOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->numberVertexOnCell_vec.size())); 
    sycl::buffer<size_t, 1> verticesOnCell_buf(mpasoF->mGrid->verticesOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->verticesOnCell_vec.size()));             
    sycl::buffer<size_t, 1> cellsOnVertex_buf(mpasoF->mGrid->cellsOnVertex_vec.data(), sycl::range<1>(mpasoF->mGrid->cellsOnVertex_vec.size()));
    sycl::buffer<size_t, 1> cells_onCell_buf(mpasoF->mGrid->cellsOnCell_vec.data(), sycl::range<1>(mpasoF->mGrid->cellsOnCell_vec.size()));
    sycl::buffer<size_t, 1> grid_info_buf(grid_info_vec.data(), sycl::range<1>(grid_info_vec.size()));
#pragma endregion   


#pragma region sycl_buffer_velocity
    sycl::buffer<vec3, 1>   cellVertexVelocity_front_buf(mpasoF->mSol_Front->cellVertexVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexVelocity_vec.size()));
    sycl::buffer<double, 1> cellVertexZTop_front_buf(mpasoF->mSol_Front->cellVertexZTop_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexZTop_vec.size()));
    sycl::buffer<double, 1> cellVertexVertVelocity_front_buf(mpasoF->mSol_Front->cellVertexVertVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Front->cellVertexVertVelocity_vec.size()));
    sycl::buffer<vec3, 1>   cellVertexVelocity_back_buf(mpasoF->mSol_Back->cellVertexVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Back->cellVertexVelocity_vec.size()));
    sycl::buffer<double, 1> cellVertexZTop_back_buf(mpasoF->mSol_Back->cellVertexZTop_vec.data(), sycl::range<1>(mpasoF->mSol_Back->cellVertexZTop_vec.size()));
    sycl::buffer<double, 1> cellVertexVertVelocity_back_buf(mpasoF->mSol_Back->cellVertexVertVelocity_vec.data(), sycl::range<1>(mpasoF->mSol_Back->cellVertexVertVelocity_vec.size()));
#pragma endregion   


#pragma region sycl_buffer_double_attributes
    bool bDoubleAttributes = false;
    std::vector<std::string> attr_names_front;
    std::vector<std::string> attr_names_back;
    std::vector<sycl::buffer<double, 1> > attr_bufs_front;
    std::vector<sycl::buffer<double, 1> > attr_bufs_back;
    if (mpasoF->mSol_Front->mDoubleAttributes.size() > 1)
    {
        bDoubleAttributes = true;
        
        for (const auto& [name, vec] : mpasoF->mSol_Front->mDoubleAttributes_CtoV)
        {
            attr_names_front.push_back(name);
            attr_bufs_front.emplace_back(vec.data(), sycl::range<1>(vec.size()));
        }

        for (const auto& [name, vec] : mpasoF->mSol_Back->mDoubleAttributes_CtoV)
        {
            attr_names_back.push_back(name);
            attr_bufs_back.emplace_back(vec.data(), sycl::range<1>(vec.size()));
        }
    }
#pragma endregion 

#pragma region sycl_buffer_particle_depths
    sycl::buffer<float, 1> particle_depths_buf(effective_depths.data(), sycl::range<1>(effective_depths.size()));
#pragma endregion 

    sycl::buffer<int, 1>    cellID_buf(default_cell_id.data(),      sycl::range<1>(default_cell_id.size()));
    sycl::buffer<vec3, 1>   wirte_points_buf(update_points.data(),  sycl::range<1>(update_points.size()));
    sycl::buffer<vec3, 1>   write_vels_buf(update_vels.data(),      sycl::range<1>(update_vels.size()));
    sycl::buffer<vec3, 1>   write_attrs_buf(update_attrs.data(),    sycl::range<1>(update_attrs.size()));
    sycl::buffer<vec3, 1>   sample_points_buf(stable_points.data(), sycl::range<1>(stable_points.size()));
    sycl_Q.submit([&](sycl::handler& cgh) 
    {

        constexpr int MAX_OUTPUTS = 8;  // max img_vec size
        constexpr int MAX_ATTRS = 8;    // max attribute size

#pragma region sycl_acc_grid
        auto acc_cellID_buf             = cellID_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_vertexCoord_buf        = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf          = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnCell_buf     = verticesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnVertex_buf      = cellsOnVertex_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cells_onCell_buf       = cells_onCell_buf.get_access<sycl::access::mode::read>(cgh);
        int grid_cell_size              = mpasoF->mGrid->mCellsSize;
        auto acc_grid_info_buf          = grid_info_buf.get_access<sycl::access::mode::read>(cgh);
#pragma endregion 

#pragma region sycl_acc_velocity
        auto acc_cellVertexVelocity_front_buf       = cellVertexVelocity_front_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexZTop_front_buf           = cellVertexZTop_front_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexVelocity_back_buf        = cellVertexVelocity_back_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexZTop_back_buf            = cellVertexZTop_back_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexVertVelocity_front_buf   = cellVertexVertVelocity_front_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexVertVelocity_back_buf    = cellVertexVertVelocity_back_buf.get_access<sycl::access::mode::read>(cgh);
#pragma endregion 

        int attr_count = 0;
        std::array<sycl::accessor<double, 1, sycl::access::mode::read>, MAX_ATTRS> acc_attr_bufs_front;
        std::array<sycl::accessor<double, 1, sycl::access::mode::read>, MAX_ATTRS> acc_attr_bufs_back;
        if (bDoubleAttributes)
        {

            attr_count = attr_bufs_front.size();

            for (int i = 0; i < attr_count; ++i) {
                acc_attr_bufs_front[i] = attr_bufs_front[i].get_access<sycl::access::mode::read>(cgh);
                acc_attr_bufs_back[i] = attr_bufs_back[i].get_access<sycl::access::mode::read>(cgh);
            }
        }

        auto acc_def_cell_id_buf    = cellID_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_wirte_points_buf   = wirte_points_buf.get_access<sycl::access::mode::write>(cgh);
        auto acc_write_vels_buf     = write_vels_buf.get_access<sycl::access::mode::write>(cgh);
        auto acc_write_attrs_buf   = write_attrs_buf.get_access<sycl::access::mode::write>(cgh);
        auto acc_sample_points_buf  = sample_points_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto acc_particle_depths_buf = particle_depths_buf.get_access<sycl::access::mode::read_write>(cgh);
        
        sycl::stream out(4096, 256, cgh);
        int n_steps             = config->simulationDuration / config->deltaT;
        int each_points_size    = config->simulationDuration / config->recordT;
        int recordT             = config->recordT;
        int dt_sign             = config->directionType == MOPS::CalcDirection::kForward ? 1 : -1;
        int deltaT              = dt_sign * config->deltaT;
        bool bEulerMethod       = config->methodType == MOPS::CalcMethodType::kEuler ? true : false;

        
       
        cgh.parallel_for(sycl::range<1>(points.size()), [=](sycl::item<1> item) 
        {
            int global_id = item[0];
            // load from grid info
            const int ACTUALL_CELL_SIZE         = (int)acc_grid_info_buf[0];
            const int ACTUALL_EDGE_SIZE         = (int)acc_grid_info_buf[1];
            const int ACTUALL_MAX_EDGE_SIZE     = (int)acc_grid_info_buf[2];
            const int ACTUALL_VERTEX_SIZE       = (int)acc_grid_info_buf[3];
            const int ACTUALL_ZTOP_LAYER        = (int)acc_grid_info_buf[4];
            const int ACTUALL_ZTOP_LAYER_P1     = (int)acc_grid_info_buf[5];
            // default
            const int MAX_VERTEX_NUM                    = 10;
            const int MAX_CELL_NEIGHBOR_NUM             = 11;
            const int MAX_VERTICAL_LEVEL_NUM            = 80;
            const int MAX_VERTEX_NEIGHBOR_NUM           = 3;

            double current_depth                        = -1.0 * (double)acc_particle_depths_buf[global_id];
            double runTime                              = 0.0;
            bool bFirstLoop                             = true;
            bool bFirstVel                              = true;
            bool bFirstAttr                             = true;
            bool bOptimize                              = true;
            
            int base_idx                                = global_id * each_points_size;
            int update_points_idx                       = 0;
            int save_times                              = 0;
            
            vec3 sample_point_position, new_position;
            int cell_id = -1;
            int cell_id_vec[MAX_VERTEX_NUM];
            int cell_neig_vec[MAX_CELL_NEIGHBOR_NUM];

            enum : int {
                R_NONE              = 0,
                R_FIND_FAIL         = 1,
                R_NOT_IN_MESH       = 2,
                R_NO_LAYER          = 3,
                R_ZERO_DENOM        = 4,
                R_VEL1_ZERO         = 5,
                R_VEL2_ZERO         = 6,
                R_FINAL_ZERO        = 7,
                R_VLA_FAIL          = 8
            };

            struct V2_Out {
                vec3 h_vel;     // horizontal velocity
                double v_vel;   // vertical velocity
                vec3 attr;
                
            };

            auto RET0 = [&](int reason) -> V2_Out {
                if (global_id==-1 /*or other cell id*/ ) {
                    out << "[ZERO] gid="<<global_id<<" step="<< (int)(runTime/deltaT)
                        <<" reason="<<reason<< sycl::endl;
                }
                return {vec3(0.0), 0.0, vec3(0.0)};
            };
            
            auto CalcVelocityAt = [&](const vec3& pos, int& cid, double alpha) -> V2_Out 
            {
                
                const int cell_id = cid;
                
                // 1. check in mesh
                vec3 current_position = pos;
                double alpha_for_interplate = alpha;

                auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
                SYCLKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx, MAX_VERTEX_NUM, ACTUALL_MAX_EDGE_SIZE, acc_verticesOnCell_buf);
                bool is_land = SYCLKernel::IsInMesh(cell_id, ACTUALL_MAX_EDGE_SIZE, current_position, acc_numberVertexOnCell_buf, acc_verticesOnCell_buf, acc_vertexCoord_buf);
                if (!is_land)
                {
                    RET0(R_NOT_IN_MESH);
                }

                
                // 2. Calc ztop at current position
                double current_point_ztop_front_vec[MAX_VERTICAL_LEVEL_NUM];
                double current_point_ztop_back_vec[MAX_VERTICAL_LEVEL_NUM];
                double current_cell_vertex_weight[MAX_VERTEX_NUM];
                vec3 current_cell_vertex_pos[MAX_VERTEX_NUM];
                
                bool rc = SYCLKernel::GetCellVertexPos(current_cell_vertex_pos, current_cell_vertices_idx, MAX_VERTEX_NUM, current_cell_vertices_number, acc_vertexCoord_buf);
                if (!rc) RET0(R_VLA_FAIL);
                // washpress
                Interpolator::CalcPolygonWachspress(current_position, current_cell_vertex_pos, current_cell_vertex_weight, current_cell_vertices_number);

                for (auto k = 0; k < ACTUALL_ZTOP_LAYER; ++k)
                {
                    double current_point_ztop_in_layer_front = 0.0;
                    double current_point_ztop_in_layer_back = 0.0;

                    // Get the ztop of each vertex
                    double current_cell_vertex_ztop_front[MAX_VERTEX_NUM];
                    double current_cell_vertex_ztop_back[MAX_VERTEX_NUM];
                    for (auto v_idx = 0; v_idx < current_cell_vertices_number; v_idx++)
                    {
                        auto VID = current_cell_vertices_idx[v_idx];
                        if (VID < 0 || VID >= ACTUALL_VERTEX_SIZE) return RET0(R_VLA_FAIL);

                        double ztop_front = acc_cellVertexZTop_front_buf[VID * ACTUALL_ZTOP_LAYER + k];
                        double ztop_back = acc_cellVertexZTop_back_buf[VID * ACTUALL_ZTOP_LAYER + k];
                        current_cell_vertex_ztop_front[v_idx] = ztop_front;
                        current_cell_vertex_ztop_back[v_idx] = ztop_back;
                    }
                    // Calculate the ZTOP of the current point
                    for (auto v_idx = 0; v_idx < current_cell_vertices_number; ++v_idx)
                    {
                        current_point_ztop_in_layer_front += current_cell_vertex_weight[v_idx] * current_cell_vertex_ztop_front[v_idx];
                        current_point_ztop_in_layer_back += current_cell_vertex_weight[v_idx] * current_cell_vertex_ztop_back[v_idx];
                    }
                    current_point_ztop_front_vec[k] = current_point_ztop_in_layer_front;
                    current_point_ztop_back_vec[k] = current_point_ztop_in_layer_back;
                }

                for (int k = 1; k < ACTUALL_ZTOP_LAYER; ++k)
                {
                    if (current_point_ztop_front_vec[k] > current_point_ztop_front_vec[k-1]) 
                    {
                        current_point_ztop_front_vec[k] = current_point_ztop_front_vec[k-1] - 1e-9;
                    }    
                    if (current_point_ztop_back_vec[k] > current_point_ztop_back_vec[k-1]) 
                    {
                        current_point_ztop_back_vec[k] = current_point_ztop_back_vec[k-1] - 1e-9;
                    }    
                } 

                
                const double eps            = 1e-8;           
                int local_layer_front       = -1;
                int local_layer_back        = -1;
                bool bSkip_Loop_find_front  = false;
                bool bSkip_Loop_find_back   = false;
                
                if (current_depth > current_point_ztop_front_vec[0] + eps) 
                {
                    local_layer_front = 0;
                    bSkip_Loop_find_front = true;
                }
                else if (current_depth < current_point_ztop_front_vec[ACTUALL_ZTOP_LAYER - 1] - eps) 
                {
                    local_layer_front = ACTUALL_ZTOP_LAYER - 1;
                    bSkip_Loop_find_front = true;
                }
                if (current_depth > current_point_ztop_back_vec[0] + eps) 
                {
                    local_layer_back = 0;
                    bSkip_Loop_find_back = true;
                }
                else if (current_depth < current_point_ztop_back_vec[ACTUALL_ZTOP_LAYER - 1] - eps) 
                {
                    local_layer_back = ACTUALL_ZTOP_LAYER - 1;
                    bSkip_Loop_find_back = true;
                }

                if (!bSkip_Loop_find_front)
                {
                    for (auto k = 1; k < ACTUALL_ZTOP_LAYER; ++k)
                    {
                        double topI = current_point_ztop_front_vec[k-1];
                        double botI = current_point_ztop_front_vec[k];
                        if (current_depth <= topI + eps && current_depth >= botI - eps)
                        {
                            local_layer_front = k;
                            break;
                        }
                    }
                }
                if (!bSkip_Loop_find_back)
                {
                    for (int k = 1; k < ACTUALL_ZTOP_LAYER; ++k)
                    {
                        double topI = current_point_ztop_back_vec[k-1];
                        double botI = current_point_ztop_back_vec[k];
                        if (current_depth <= topI + eps && current_depth >= botI - eps)
                        {
                            local_layer_back = k;
                            break;
                        }
                    }
                }
                
                
                if (local_layer_back < 0 || local_layer_front < 0)
                {
                    // out << "DEBUG: Particle " << global_id << " depth layer not found. " 
                    // << "current_depth=" << current_depth 
                    // << " ztop_surface_front=" << current_point_ztop_front_vec[0] 
                    // << " ztop_bottom_front=" << current_point_ztop_front_vec[ACTUALL_ZTOP_LAYER-1]
                    // << " ztop_surface_back=" << current_point_ztop_back_vec[0] 
                    // << " ztop_bottom_back=" << current_point_ztop_back_vec[ACTUALL_ZTOP_LAYER-1] 
                    // << sycl::endl;
                    return RET0(R_NO_LAYER);
                }
                else
                {
                    // Calc velocity by interpolation
                    double ztop_layer_front_dn, ztop_layer_front_up; 
                    double ztop_layer_back_dn, ztop_layer_back_up;
                    ztop_layer_front_dn = current_point_ztop_front_vec[local_layer_front];      
                    ztop_layer_front_up = current_point_ztop_front_vec[local_layer_front - 1]; 

                    ztop_layer_back_dn = current_point_ztop_back_vec[local_layer_back];        
                    ztop_layer_back_up = current_point_ztop_back_vec[local_layer_back - 1];    

                    double x_font = current_depth;
                    double x_back = current_depth;
                    
                    x_font = sycl::max(ztop_layer_front_dn, sycl::min(x_font, ztop_layer_front_up));
                    double denom = ztop_layer_front_up - ztop_layer_front_dn;
                    if (sycl::fabs(denom) < 1e-12) return RET0(R_ZERO_DENOM);
                    double t_front = (x_font - ztop_layer_front_dn) / denom;
                    x_back = sycl::max(ztop_layer_back_dn, sycl::min(x_back, ztop_layer_back_up));
                    denom = ztop_layer_back_up - ztop_layer_back_dn;
                    if (sycl::fabs(denom) < 1e-12) return RET0(R_ZERO_DENOM);
                    double t_back = (x_back - ztop_layer_back_dn) / denom;
                
                    vec3 final_vel_front;
                    vec3 final_vel_back;
                   
                    // horizontal velocity interpolation
                    // lower_front
                    vec3 current_point_vel_dn_front = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_front, acc_cellVertexVelocity_front_buf);
                    // upper_front
                    vec3 current_point_vel_up_front = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_front - 1, acc_cellVertexVelocity_front_buf);
                    
                    final_vel_front.x() = t_front * current_point_vel_up_front.x() + (1 - t_front) * current_point_vel_dn_front.x();
                    final_vel_front.y() = t_front * current_point_vel_up_front.y() + (1 - t_front) * current_point_vel_dn_front.y();
                    final_vel_front.z() = t_front * current_point_vel_up_front.z() + (1 - t_front) * current_point_vel_dn_front.z();

                    vec3 current_point_vel_dn_back = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_back, acc_cellVertexVelocity_back_buf);
                
                    vec3 current_point_vel_up_back = SYCLKernel::CalcVelocity(current_cell_vertices_idx, current_cell_vertex_weight, 
                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_back - 1, acc_cellVertexVelocity_back_buf);
                    
                    final_vel_back.x() = t_back * current_point_vel_up_back.x() + (1 - t_back) * current_point_vel_dn_back.x();
                    final_vel_back.y() = t_back * current_point_vel_up_back.y() + (1 - t_back) * current_point_vel_dn_back.y();
                    final_vel_back.z() = t_back * current_point_vel_up_back.z() + (1 - t_back) * current_point_vel_dn_back.z();

                    vec3 current_horizontal_velocity; 
                    current_horizontal_velocity.x() = alpha_for_interplate * final_vel_back.x() + (1 - alpha_for_interplate) * final_vel_front.x();
                    current_horizontal_velocity.y() = alpha_for_interplate * final_vel_back.y() + (1 - alpha_for_interplate) * final_vel_front.y();
                    current_horizontal_velocity.z() = alpha_for_interplate * final_vel_back.z() + (1 - alpha_for_interplate) * final_vel_front.z();
                    

                    // vertical velocity interpolation
                    int dn_if_front = local_layer_front;
                    int up_if_front = (local_layer_front > 0) ? (local_layer_front - 1) : 0;

                    // lower_front
                    double w_dn_front = SYCLKernel::CalcAttribute(
                        current_cell_vertices_idx, current_cell_vertex_weight,
                        MAX_VERTEX_NUM, current_cell_vertices_number,
                        ACTUALL_ZTOP_LAYER_P1, dn_if_front,
                        acc_cellVertexVertVelocity_front_buf);
                    
                    // upper_front
                    double w_up_front = SYCLKernel::CalcAttribute(
                        current_cell_vertices_idx, current_cell_vertex_weight,
                        MAX_VERTEX_NUM, current_cell_vertices_number,
                        ACTUALL_ZTOP_LAYER_P1, up_if_front,
                        acc_cellVertexVertVelocity_front_buf);

                    double w_front = t_front * w_up_front + (1.0 - t_front) * w_dn_front;

                    int dn_if_back = local_layer_back;
                    int up_if_back = (local_layer_back > 0) ? (local_layer_back - 1) : 0;

                    double w_dn_back = SYCLKernel::CalcAttribute(
                        current_cell_vertices_idx, current_cell_vertex_weight,
                        MAX_VERTEX_NUM, current_cell_vertices_number,
                        ACTUALL_ZTOP_LAYER_P1, dn_if_back,
                        acc_cellVertexVertVelocity_back_buf);

                    double w_up_back = SYCLKernel::CalcAttribute(
                        current_cell_vertices_idx, current_cell_vertex_weight,
                        MAX_VERTEX_NUM, current_cell_vertices_number,
                        ACTUALL_ZTOP_LAYER_P1, up_if_back,
                        acc_cellVertexVertVelocity_back_buf);

                    double w_back = t_back * w_up_back + (1.0 - t_back) * w_dn_back;

                    double current_vertical_velocity = alpha_for_interplate * w_back + (1 - alpha_for_interplate) * w_front;
                    
                    // calc attributes
                    vec3 current_point_attr_value = {0.0, 0.0, 0.0};
                    double current_point_attr1_value_front = 0.0;
                    double current_point_attr1_value_back = 0.0;
                    double current_point_attr2_value_front = 0.0;
                    double current_point_attr2_value_back = 0.0; 
                    
                    if (bDoubleAttributes)
                    {
                        if (attr_count >= 1) 
                        {
                            // lower_front
                            double attr_dn_front = SYCLKernel::CalcAttribute(current_cell_vertices_idx, current_cell_vertex_weight,
                                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_front, acc_attr_bufs_front[0]);
                            // upper_front   
                            double attr_up_front = SYCLKernel::CalcAttribute(current_cell_vertices_idx, current_cell_vertex_weight,
                                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_front - 1, acc_attr_bufs_front[0]);
                            current_point_attr1_value_front = t_front * attr_up_front + (1 - t_front) * attr_dn_front;


                            // lower_back
                            double attr_dn_back = SYCLKernel::CalcAttribute(current_cell_vertices_idx, current_cell_vertex_weight,
                                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_back, acc_attr_bufs_back[0]);
                            // upper_back
                            double attr_up_back = SYCLKernel::CalcAttribute(current_cell_vertices_idx, current_cell_vertex_weight,
                                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_back - 1, acc_attr_bufs_back[0]);
                            current_point_attr1_value_back = t_back * attr_up_back + (1 - t_back) * attr_dn_back;

                            current_point_attr_value.x() = alpha_for_interplate * current_point_attr1_value_back + (1 - alpha_for_interplate) * current_point_attr1_value_front;
                        }
                        if (attr_count >= 2) 
                        {
                            // lower_front
                            double attr_dn_front = SYCLKernel::CalcAttribute(current_cell_vertices_idx, current_cell_vertex_weight,
                                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_front, acc_attr_bufs_front[1]);
                            // upper_front   
                            double attr_up_front = SYCLKernel::CalcAttribute(current_cell_vertices_idx, current_cell_vertex_weight,
                                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_front - 1, acc_attr_bufs_front[1]);
                            current_point_attr2_value_front = t_front * attr_up_front + (1 - t_front) * attr_dn_front;


                            // lower_back
                            double attr_dn_back = SYCLKernel::CalcAttribute(current_cell_vertices_idx, current_cell_vertex_weight,
                                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_back, acc_attr_bufs_back[1]);
                            // upper_back
                            double attr_up_back = SYCLKernel::CalcAttribute(current_cell_vertices_idx, current_cell_vertex_weight,
                                        MAX_VERTEX_NUM, current_cell_vertices_number, ACTUALL_ZTOP_LAYER, local_layer_back - 1, acc_attr_bufs_back[1]);
                            current_point_attr2_value_back = t_back * attr_up_back + (1 - t_back) * attr_dn_back;

                            current_point_attr_value.y() = alpha_for_interplate * current_point_attr2_value_back + (1 - alpha_for_interplate) * current_point_attr2_value_front;
                        }
                    }


                    return {current_horizontal_velocity, current_vertical_velocity, current_point_attr_value};
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
                sample_point_position = acc_sample_points_buf[global_id];
                
                int firtst_cell_id = acc_def_cell_id_buf[global_id];

                current_depth = -1.0 * (double)acc_particle_depths_buf[global_id];
               
                if (bFirstLoop)
                {
                    bFirstLoop = false;
                    cell_id = firtst_cell_id;
                    
                    auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                    SYCLKernel::GetCellNeighborsIdx(cell_id, current_cell_vertices_number, cell_neig_vec, MAX_VERTEX_NUM, ACTUALL_MAX_EDGE_SIZE, acc_cells_onCell_buf);
                    int acc_wirte_pints_idx = base_idx + 0;
                    acc_wirte_points_buf[acc_wirte_pints_idx] = sample_point_position;
                }
                else
                {
                    auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
                    double max_len = std::numeric_limits<double>::max();
                    for (auto idx = 0; idx < current_cell_vertices_number + 1; idx++)
                    {
                        auto CID = cell_neig_vec[idx];
                        if (CID < 0 || CID >= ACTUALL_CELL_SIZE) continue;
                        vec3 pos = acc_cellCoord_buf[CID];
                        double len = MOPS_LENGTH(pos - sample_point_position);
                        if (len < max_len)
                        {
                            max_len = len;
                            cell_id = CID;
                        }
                    }
                    SYCLKernel::GetCellNeighborsIdx(cell_id, current_cell_vertices_number, cell_neig_vec, MAX_VERTEX_NUM, ACTUALL_MAX_EDGE_SIZE, acc_cells_onCell_buf);
                   
                }


                vec3 current_position = sample_point_position;
                double r = MOPS_LENGTH(current_position);
                vec3 rk4_next_position = current_position;
                
                
                
                // vec3 current_velocity;
                vec3 current_horizontal_velocity;
                double current_vertical_velocity;
                vec3 current_attrs;

                auto AdvectOnSphere = [&](const vec3& pos, const vec3& vel, double dt_local) -> vec3 {
                    const double rr = MOPS_LENGTH(pos);
                    const double speed_local = MOPS_LENGTH(vel);
                    if (rr < 1e-12 || speed_local < 1e-12) return pos;
                    vec3 axis = SYCLKernel::CalcRotationAxis(pos, vel);
                    double theta = (speed_local * dt_local) / rr;
                    return SYCLKernel::CalcPositionAfterRotation(pos, axis, theta);
                };

                if (bEulerMethod)
                {
                    // Euler method
                    auto current_velocity_attrs = CalcVelocityAt(current_position, cell_id, alpha_for_interplate);
                    current_horizontal_velocity            = current_velocity_attrs.h_vel;
                    current_vertical_velocity              = current_velocity_attrs.v_vel;
                    current_attrs               = current_velocity_attrs.attr;
                }
                else
                {
                    // Runge-Kutta 4th order method
                    const double dt     = static_cast<double>(deltaT);
                    const double dalpha = dt / (double)config->simulationDuration;
                    
                    // k1 @ (x, t)
                    double a1   = alpha_for_interplate;
                    int cid1    = cell_id;
                    auto s1     = CalcVelocityAt(current_position, cell_id, a1);
                    vec3 k1     = s1.h_vel;
                    vec3 A1     = s1.attr;
                    double v1     = s1.v_vel;

                    // k2 @ (x + 0.5*dt*k1, t + 0.5*dt)
                    vec3 p2     = AdvectOnSphere(current_position, k1, dt * 0.5);
                    double a2   = a1 + 0.5 * dalpha; if (a2 > 1.0) a2 = 1.0; if (a2 < 0.0) a2 = 0.0;
                    int  cid2   = cid1;
                    auto s2     = CalcVelocityAt(p2, cid2, a2);
                    vec3 k2     = s2.h_vel;
                    vec3 A2     = s2.attr;
                    double v2     = s2.v_vel;

                    // k3 @ (x + 0.5*dt*k2, t + 0.5*dt)
                    vec3 p3     = AdvectOnSphere(current_position, k2, dt * 0.5);
                    double a3   = a1 + 0.5 * dalpha; if (a3 > 1.0) a3 = 1.0; if (a3 < 0.0) a3 = 0.0;
                    int  cid3   = cid1;
                    auto s3     = CalcVelocityAt(p3, cid3, a3);
                    vec3 k3     = s3.h_vel;
                    vec3 A3     = s3.attr;
                    double v3     = s3.v_vel;

                    // k4 @ (x + dt*k3, t + dt)
                    vec3 p4     = AdvectOnSphere(current_position, k3, dt);
                    double a4   = a1 + dalpha; if (a4 > 1.0) a4 = 1.0; if (a4 < 0.0) a4 = 0.0;
                    int  cid4   = cid1;
                    auto s4     = CalcVelocityAt(p4, cid4, a4);
                    vec3 k4     = s4.h_vel;
                    vec3 A4     = s4.attr;
                    double v4     = s4.v_vel;

                    current_horizontal_velocity    = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
                    current_attrs       = (A1 + 2.0 * A2 + 2.0 * A3 + A4) / 6.0;
                    current_vertical_velocity = (v1 + 2.0 * v2 + 2.0 * v3 + v4) / 6.0;

                    // Strict RK4 endpoint on position: x_{n+1} = x_n + dt * (k1 + 2k2 + 2k3 + k4) / 6,
                    // then project back to the current sphere radius before vertical-radius update.
                    vec3 x_trial = current_position + current_horizontal_velocity * dt;
                    double x_trial_len = MOPS_LENGTH(x_trial);
                    if (x_trial_len > 1e-12) {
                        rk4_next_position = (x_trial / x_trial_len) * r;
                    } else {
                        rk4_next_position = current_position;
                    }
                    cell_id = cid4;
                }
                
                if (bEulerMethod)
                {
                    vec3 rotationAxis = SYCLKernel::CalcRotationAxis(current_position, current_horizontal_velocity);
                    double speed = MOPS_LENGTH(current_horizontal_velocity);
                    double theta_rad = (speed * deltaT) / r;
                    new_position = SYCLKernel::CalcPositionAfterRotation(current_position, rotationAxis, theta_rad);
                }
                else
                {
                    new_position = rk4_next_position;
                }
                
                if (bFirstVel)
                {
                    bFirstVel = false;
                    int acc_wirte_pints_idx = base_idx + 0;
                    acc_write_vels_buf[acc_wirte_pints_idx] = current_horizontal_velocity;
                }

                if (bFirstAttr && bDoubleAttributes)
                {
                    bFirstAttr = false;
                    int acc_wirte_pints_idx = base_idx + 0;
                    acc_write_attrs_buf[acc_wirte_pints_idx] = current_attrs;
                }

                // Update depth using signed vertical velocity.
                // Convention: particle depth is positive downward; MPAS vertical velocity is positive upward.
                const double old_depth = (double)acc_particle_depths_buf[global_id];
                double new_depth = old_depth - current_vertical_velocity * (double)deltaT;
                new_depth = sycl::fmax(0.0, new_depth);

                // Keep particle position on sphere with updated radius.
                double r_new = sycl::fmax(1.0, r + current_vertical_velocity * (double)deltaT);
                acc_particle_depths_buf[global_id] = (float)new_depth;
                new_position = sycl::normalize(new_position) * r_new;
                acc_sample_points_buf[global_id] = new_position;
                    
                if ((i_step + 1) % (recordT / deltaT) == 0)
                {
                    int acc_wirte_pints_idx = base_idx + update_points_idx;
                    acc_wirte_points_buf[acc_wirte_pints_idx] = new_position;
                    acc_write_vels_buf[acc_wirte_pints_idx] = current_horizontal_velocity;
                    

                    if (bDoubleAttributes)
                    {
                        acc_write_attrs_buf[acc_wirte_pints_idx] = current_attrs;
                    }
                    update_points_idx = update_points_idx + 1;
                }
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
    auto after_write_a = write_attrs_buf.get_host_access(sycl::read_only); // RGB

    size_t each_point_size = config->simulationDuration / config->recordT;
    size_t total_points = update_points.size();
    auto clean_traj = MOPS::Common::FinalizeTrajectoryLinesWithAttrs(
        trajectory_lines,
        after_write_p,
        after_write_v,
        after_write_a,
        each_point_size,
        total_points);
    trajectory_lines.clear();
    return clean_traj;
}




} // namespace MOPS::GPU::SYCLBackend
