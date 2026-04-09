#include "GPU/SYCL/MPASOSolutionSYCL.h"

#include "GPU/SYCL/Kernel/SYCLKernel.h"
#include "Utils/GeoConverter.hpp"
#include "Utils/Interpolation.hpp"

namespace MOPS::GPU::SYCLBackend {

void CalcCellVertexZtop(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_center_ztop,
    std::vector<double>& cell_vertex_ztop,
    const std::vector<size_t>& grid_info,
    sycl::queue& q)
{
#if USE_SYCL
    sycl::buffer<vec3, 1> vertexCoord_buf(grid->vertexCoord_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size()));
    sycl::buffer<vec3, 1> cellCoord_buf(grid->cellCoord_vec.data(), sycl::range<1>(grid->cellCoord_vec.size()));

    sycl::buffer<size_t, 1> numberVertexOnCell_buf(grid->numberVertexOnCell_vec.data(), sycl::range<1>(grid->numberVertexOnCell_vec.size()));
    sycl::buffer<size_t, 1> verticesOnCell_buf(grid->verticesOnCell_vec.data(), sycl::range<1>(grid->verticesOnCell_vec.size()));
    sycl::buffer<size_t, 1> cellsOnVertex_buf(grid->cellsOnVertex_vec.data(), sycl::range<1>(grid->cellsOnVertex_vec.size()));

    sycl::buffer<double, 1> cellCenterZTop_buf(cell_center_ztop.data(), sycl::range<1>(cell_center_ztop.size()));
    sycl::buffer<double, 1> cellVertexZTop_buf(cell_vertex_ztop.data(), sycl::range<1>(grid->vertexCoord_vec.size() * total_ztop_layer));

    sycl::buffer<size_t, 1> grid_info_buf(grid_info.data(), sycl::range<1>(grid_info.size()));

    q.submit([&](sycl::handler& cgh) {
        auto acc_vertexCoord_buf = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnCell_buf = verticesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnVertex_buf = cellsOnVertex_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCenterZTop_buf = cellCenterZTop_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexZTop_buf = cellVertexZTop_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto acc_grid_info_buf = grid_info_buf.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::range<2>(cells_size, total_ztop_layer), [=](sycl::id<2> idx) {
            size_t j = idx[0];
            size_t i = idx[1];

            auto cell_id = j;
            auto current_layer = i;

            const int CELL_SIZE = acc_grid_info_buf[0];
            const int max_edge = acc_grid_info_buf[2];
            const int MAX_VERTEX_NUM = 20;
            const int NEIGHBOR_NUM = 3;
            const int TOTAY_ZTOP_LAYER = acc_grid_info_buf[4];
            const int VERTLEVELS = acc_grid_info_buf[4];

            auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
            auto nan = std::numeric_limits<size_t>::max();
            size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
            SYCLKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx, MAX_VERTEX_NUM, max_edge, acc_verticesOnCell_buf);

            double current_cell_vertices_value[MAX_VERTEX_NUM];

            for (auto k = 0; k < MAX_VERTEX_NUM; ++k)
            {
                bool bBoundary = false;
                auto vertex_idx = current_cell_vertices_idx[k];
                if (vertex_idx == nan) {
                    current_cell_vertices_value[k] = std::numeric_limits<double>::quiet_NaN();
                    continue;
                }

                auto current_vertex = acc_vertexCoord_buf[vertex_idx];
                size_t tmp_cell_id[3];
                tmp_cell_id[0] = acc_cellsOnVertex_buf[3 * vertex_idx + 0] - 1;
                tmp_cell_id[1] = acc_cellsOnVertex_buf[3 * vertex_idx + 1] - 1;
                tmp_cell_id[2] = acc_cellsOnVertex_buf[3 * vertex_idx + 2] - 1;

                double tmp_cell_center_ztop[3];
                for (auto tmp_cell = 0; tmp_cell < NEIGHBOR_NUM; tmp_cell++)
                {
                    double value;
                    if (tmp_cell_id[tmp_cell] > CELL_SIZE + 1)
                    {
                        value = 0.0;
                        tmp_cell_center_ztop[tmp_cell] = value;
                        bBoundary = true;
                    }
                    else
                    {
                        auto ztop_idx = VERTLEVELS * tmp_cell_id[tmp_cell] + current_layer;
                        double ztop = acc_cellCenterZTop_buf[ztop_idx];
                        tmp_cell_center_ztop[tmp_cell] = ztop;
                    }
                }

                if (bBoundary)
                {
                    current_cell_vertices_value[k] = 0.0 * tmp_cell_center_ztop[0] + 0.0 * tmp_cell_center_ztop[1] + 0.0 * tmp_cell_center_ztop[2];
                }
                else
                {
                    double u, v, w;
                    vec3 p1 = acc_cellCoord_buf[tmp_cell_id[0]];
                    vec3 p2 = acc_cellCoord_buf[tmp_cell_id[1]];
                    vec3 p3 = acc_cellCoord_buf[tmp_cell_id[2]];
                    Interpolator::TRIANGLE tri(p1, p2, p3);
                    Interpolator::calcTriangleBarycentric(current_vertex, &tri, u, v, w);
                    current_cell_vertices_value[k] = u * tmp_cell_center_ztop[0] + v * tmp_cell_center_ztop[1] + w * tmp_cell_center_ztop[2];
                }

                acc_cellVertexZTop_buf[vertex_idx * TOTAY_ZTOP_LAYER + current_layer] = current_cell_vertices_value[k];
            }
        });
    });

    q.wait_and_throw();

    // Keep host access to preserve buffer synchronization behavior.
    auto host_accessor = cellVertexZTop_buf.get_host_access(sycl::read_only);
    (void)host_accessor;
#endif
}

void CalcCellCenterToVertex(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_center_attr,
    std::vector<double>& cell_vertex_attr,
    const std::vector<size_t>& grid_info,
    sycl::queue& q)
{
#if USE_SYCL
    sycl::buffer<vec3, 1> vertexCoord_buf(grid->vertexCoord_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size()));
    sycl::buffer<vec3, 1> cellCoord_buf(grid->cellCoord_vec.data(), sycl::range<1>(grid->cellCoord_vec.size()));

    sycl::buffer<size_t, 1> numberVertexOnCell_buf(grid->numberVertexOnCell_vec.data(), sycl::range<1>(grid->numberVertexOnCell_vec.size()));
    sycl::buffer<size_t, 1> verticesOnCell_buf(grid->verticesOnCell_vec.data(), sycl::range<1>(grid->verticesOnCell_vec.size()));
    sycl::buffer<size_t, 1> cellsOnVertex_buf(grid->cellsOnVertex_vec.data(), sycl::range<1>(grid->cellsOnVertex_vec.size()));

    sycl::buffer<double, 1> cellCenterAttr_buf(cell_center_attr.data(), sycl::range<1>(cell_center_attr.size()));
    sycl::buffer<double, 1> cellVertexAttr_buf(cell_vertex_attr.data(), sycl::range<1>(grid->vertexCoord_vec.size() * total_ztop_layer));

    sycl::buffer<size_t, 1> grid_info_buf(grid_info.data(), sycl::range<1>(grid_info.size()));

    q.submit([&](sycl::handler& cgh) {

        auto acc_vertexCoord_buf        = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf          = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnCell_buf     = verticesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnVertex_buf      = cellsOnVertex_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCenterAttr_buf     = cellCenterAttr_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexAttr_buf     = cellVertexAttr_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto acc_grid_info_buf          = grid_info_buf.get_access<sycl::access::mode::read>(cgh);


        cgh.parallel_for(sycl::range<2>(cells_size, total_ztop_layer), [=](sycl::id<2> idx) {
            size_t j = idx[0];
            size_t i = idx[1];

            auto cell_id = j;
            auto current_layer = i;

            const int CELL_SIZE = acc_grid_info_buf[0];
            const int max_edge = acc_grid_info_buf[2];
            const int MAX_VERTEX_NUM = 20;
            const int NEIGHBOR_NUM = 3;
            const int TOTAY_ZTOP_LAYER = acc_grid_info_buf[4];
            const int VERTLEVELS = acc_grid_info_buf[4];

            auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
            auto nan = std::numeric_limits<size_t>::max();
            size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
            SYCLKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx, MAX_VERTEX_NUM, max_edge, acc_verticesOnCell_buf);
            double current_cell_vertices_value[MAX_VERTEX_NUM];

            for (auto k = 0; k < MAX_VERTEX_NUM; ++k)
            {
                bool bBoundary = false;
                auto vertex_idx = current_cell_vertices_idx[k];
                if (vertex_idx == nan) { current_cell_vertices_value[k] = std::numeric_limits<double>::quiet_NaN(); continue; }
                auto current_vertex = acc_vertexCoord_buf[vertex_idx];
                size_t tmp_cell_id[3];
                tmp_cell_id[0] = acc_cellsOnVertex_buf[3 * vertex_idx + 0] - 1;
                tmp_cell_id[1] = acc_cellsOnVertex_buf[3 * vertex_idx + 1] - 1;
                tmp_cell_id[2] = acc_cellsOnVertex_buf[3 * vertex_idx + 2] - 1;

                double tmp_cell_center_attr[3];
                for (auto tmp_cell = 0; tmp_cell < NEIGHBOR_NUM; tmp_cell++)
                {
                    if (tmp_cell_id[tmp_cell] > CELL_SIZE + 1)
                    {
                        tmp_cell_center_attr[tmp_cell] = 0.0;
                        bBoundary = true;
                    }
                    else
                    {
                        auto attr_idx = VERTLEVELS * tmp_cell_id[tmp_cell] + current_layer;
                        tmp_cell_center_attr[tmp_cell] = acc_cellCenterAttr_buf[attr_idx];
                    }
                }
                if (bBoundary)
                {
                    current_cell_vertices_value[k] = 0.0 * tmp_cell_center_attr[0] + 0.0 * tmp_cell_center_attr[1] + 0.0 * tmp_cell_center_attr[2];
                }
                else
                {
                    double u, v, w;
                    vec3 p1 = acc_cellCoord_buf[tmp_cell_id[0]];
                    vec3 p2 = acc_cellCoord_buf[tmp_cell_id[1]];
                    vec3 p3 = acc_cellCoord_buf[tmp_cell_id[2]];
                    Interpolator::TRIANGLE tri(p1, p2, p3);
                    Interpolator::calcTriangleBarycentric(current_vertex, &tri, u, v, w);
                    current_cell_vertices_value[k] = u * tmp_cell_center_attr[0] + v * tmp_cell_center_attr[1] + w * tmp_cell_center_attr[2];
                    if (current_cell_vertices_value[k] < 0) current_cell_vertices_value[k] = 0.0;
                }

                acc_cellVertexAttr_buf[vertex_idx * TOTAY_ZTOP_LAYER + current_layer] = current_cell_vertices_value[k];
            }
        });
    });
    q.wait_and_throw();
    auto host_accessor = cellVertexAttr_buf.get_host_access(sycl::read_only);
    (void)host_accessor;
#endif
}

void CalcCellCenterVelocityByZM(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_zonal_velocity,
    const std::vector<double>& cell_meridional_velocity,
    std::vector<vec3>& cell_center_velocity,
    const std::vector<size_t>& grid_info,
    sycl::queue& q)
{
#if USE_SYCL
    sycl::buffer<vec3, 1> vertexCoord_buf(grid->vertexCoord_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size()));
    sycl::buffer<vec3, 1> cellCoord_buf(grid->cellCoord_vec.data(), sycl::range<1>(grid->cellCoord_vec.size()));
    sycl::buffer<vec3, 1> edgeCoord_buf(grid->edgeCoord_vec.data(), sycl::range<1>(grid->edgeCoord_vec.size()));

    sycl::buffer<size_t, 1> numberVertexOnCell_buf(grid->numberVertexOnCell_vec.data(), sycl::range<1>(grid->numberVertexOnCell_vec.size()));
    sycl::buffer<size_t, 1> edgesOnCell_buf(grid->edgesOnCell_vec.data(), sycl::range<1>(grid->edgesOnCell_vec.size()));
    sycl::buffer<size_t, 1> cellsOnEdge_buf(grid->cellsOnEdge_vec.data(), sycl::range<1>(grid->cellsOnEdge_vec.size()));
    sycl::buffer<size_t, 1> verticesOnEdge_buf(grid->verticesOnEdge_vec.data(), sycl::range<1>(grid->verticesOnEdge_vec.size()));

    sycl::buffer<double, 1> cellZonalVelocity_buf(cell_zonal_velocity.data(), sycl::range<1>(cell_zonal_velocity.size()));
    sycl::buffer<double, 1> cellMeridionalVelocity_buf(cell_meridional_velocity.data(), sycl::range<1>(cell_meridional_velocity.size()));
    sycl::buffer<vec3, 1> cellCenterVelocity_buf(cell_center_velocity.data(), sycl::range<1>(cell_center_velocity.size()));
    sycl::buffer<size_t, 1> grid_info_buf(grid_info.data(), sycl::range<1>(grid_info.size()));

    q.submit([&](sycl::handler& cgh) {
        auto acc_vertexCoord_buf = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_edgeCoord_buf = edgeCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_edgesOnCell_buf = edgesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnEdge_buf = cellsOnEdge_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnEdge_buf = verticesOnEdge_buf.get_access<sycl::access::mode::read>(cgh);

        auto acc_cellZonalVelocity_buf = cellZonalVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellMeridionalVelocity_buf = cellMeridionalVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCenterVelocity_buf = cellCenterVelocity_buf.get_access<sycl::access::mode::write>(cgh);
        auto acc_grid_info_buf          = grid_info_buf.get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for(sycl::range<2>(cells_size, total_ztop_layer), [=](sycl::id<2> idx) {

            size_t j = idx[0];
            size_t i = idx[1];

            auto cell_id = j;
            auto current_layer = i;

            const int TOTAY_ZTOP_LAYER = acc_grid_info_buf[4];
            vec3 cell_center_velocity = { 0.0, 0.0, 0.0 };
            vec3 cell_position = acc_cellCoord_buf[cell_id];
            double tmp_zonal = acc_cellZonalVelocity_buf[cell_id * TOTAY_ZTOP_LAYER + current_layer];
            double tmp_mer = acc_cellMeridionalVelocity_buf[cell_id * TOTAY_ZTOP_LAYER + current_layer];
            GeoConverter::convertENUVelocityToXYZ(cell_position, tmp_zonal, tmp_mer, 0.0, cell_center_velocity);
            acc_cellCenterVelocity_buf[cell_id * TOTAY_ZTOP_LAYER + current_layer] = cell_center_velocity;
        });
    });
    q.wait_and_throw();

    auto host_accessor = cellCenterVelocity_buf.get_host_access(sycl::read_only);
    (void)host_accessor;
#endif
}

void CalcCellCenterVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<double>& cell_normal_velocity,
    std::vector<vec3>& cell_center_velocity,
    const std::vector<size_t>& grid_info,
    sycl::queue& q)
{
#if USE_SYCL
    sycl::buffer<vec3, 1> vertexCoord_buf(grid->vertexCoord_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size()));
    sycl::buffer<vec3, 1> cellCoord_buf(grid->cellCoord_vec.data(), sycl::range<1>(grid->cellCoord_vec.size()));
    sycl::buffer<vec3, 1> edgeCoord_buf(grid->edgeCoord_vec.data(), sycl::range<1>(grid->edgeCoord_vec.size()));

    sycl::buffer<size_t, 1> numberVertexOnCell_buf(grid->numberVertexOnCell_vec.data(), sycl::range<1>(grid->numberVertexOnCell_vec.size()));
    sycl::buffer<size_t, 1> edgesOnCell_buf(grid->edgesOnCell_vec.data(), sycl::range<1>(grid->edgesOnCell_vec.size()));
    sycl::buffer<size_t, 1> cellsOnEdge_buf(grid->cellsOnEdge_vec.data(), sycl::range<1>(grid->cellsOnEdge_vec.size()));
    sycl::buffer<size_t, 1> verticesOnEdge_buf(grid->verticesOnEdge_vec.data(), sycl::range<1>(grid->verticesOnEdge_vec.size()));

    sycl::buffer<double, 1> cellNormalVelocity_buf(cell_normal_velocity.data(), sycl::range<1>(cell_normal_velocity.size()));
    sycl::buffer<vec3, 1> cellCenterVelocity_buf(cell_center_velocity.data(), sycl::range<1>(cell_center_velocity.size()));

    sycl::buffer<size_t, 1> grid_info_buf(grid_info.data(), sycl::range<1>(grid_info.size()));

    q.submit([&](sycl::handler& cgh) {
        auto acc_vertexCoord_buf = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_edgeCoord_buf = edgeCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_edgesOnCell_buf = edgesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnEdge_buf = cellsOnEdge_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnEdge_buf = verticesOnEdge_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellNormalVelocity_buf = cellNormalVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCenterVelocity_buf = cellCenterVelocity_buf.get_access<sycl::access::mode::write>(cgh);
        auto acc_grid_info_buf          = grid_info_buf.get_access<sycl::access::mode::read>(cgh);
        sycl::stream out(8192, 256, cgh);
        cgh.parallel_for(sycl::range<2>(cells_size, total_ztop_layer), [=](sycl::id<2> idx) {

            size_t j = idx[0];
            size_t i = idx[1];

            auto cell_id = j;
            auto current_layer = i;

            const int CELL_SIZE = acc_grid_info_buf[0];
            const int MAX_VERTEX_NUM = 7;
            const int TOTAY_ZTOP_LAYER = acc_grid_info_buf[4];
            vec3 cell_center_velocity = { 0.0, 0.0, 0.0 };
            vec3 cell_position = acc_cellCoord_buf[cell_id];

            size_t current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
            size_t current_cell_edges_id[MAX_VERTEX_NUM];
            for (auto k = 0; k < current_cell_vertices_number; ++k)
            {
                current_cell_edges_id[k] = acc_edgesOnCell_buf[cell_id * MAX_VERTEX_NUM + k] -1;
            }
            auto nan = std::numeric_limits<size_t>::max();
            for (auto k = current_cell_vertices_number; k < MAX_VERTEX_NUM; ++k)
            {
                current_cell_edges_id[k] = nan;
            }

            vec3 up = cell_position / MOPS_LENGTH(cell_position);
            vec3 k = vec3(0.0, 0.0, 1.0);
            vec3 east = MOPS_CROSS(k, up);
            if (MOPS_LENGTH(east) < 1e-6) {
                east = MOPS_CROSS(vec3(0.0, 1.0, 0.0), up);
            }
            east = east / MOPS_LENGTH(east);
            vec3 north = MOPS_CROSS(up, east);

            double planeBasisVector[2][3] = { {east.x(), east.y(), east.z()}, {north.x(), north.y(), north.z()} };
            double cellCenter[3] = { cell_position.x(), cell_position.y(), cell_position.z() };
            int pointCount = MAX_VERTEX_NUM;
            double edge_center[MAX_VERTEX_NUM][3];
            double unit_vector[MAX_VERTEX_NUM][3];
            double noraml_vel[MAX_VERTEX_NUM][1];
            double coeffs[MAX_VERTEX_NUM][3] = {{0.0}};

            for (auto kidx = 0; kidx < MAX_VERTEX_NUM; ++kidx)
            {
                auto edge_id = current_cell_edges_id[kidx];
                if (edge_id == nan) { continue; }
                vec3 edge_position = acc_edgeCoord_buf[edge_id];
                edge_center[kidx][0] = edge_position.x(); edge_center[kidx][1] = edge_position.y(); edge_center[kidx][2] = edge_position.z();
                size_t tmp_cell_id[2];
                tmp_cell_id[0] = acc_cellsOnEdge_buf[edge_id * 2.0f + 0] - 1;
                tmp_cell_id[1] = acc_cellsOnEdge_buf[edge_id * 2.0f + 1] - 1;
                auto min_cell_id = tmp_cell_id[0] < tmp_cell_id[1] ? tmp_cell_id[0] : tmp_cell_id[1];
                auto max_cell_id = tmp_cell_id[0] > tmp_cell_id[1] ? tmp_cell_id[0] : tmp_cell_id[1];
                vec3 normal_vector;
                double length;
                if (max_cell_id > CELL_SIZE)
                {
                    vec3 min_cell_position = acc_cellCoord_buf[min_cell_id];
                    normal_vector = edge_position - min_cell_position;
                    length = MOPS_LENGTH(normal_vector);
                    if (length == 0.0) { continue; }
                    normal_vector /= length;
                }
                else
                {
                    vec3 min_cell_position = acc_cellCoord_buf[min_cell_id];
                    vec3 max_cell_position = acc_cellCoord_buf[max_cell_id];
                    normal_vector = max_cell_position - min_cell_position;
                    length = MOPS_LENGTH(normal_vector);
                    if (length == 0.0) { continue; }
                    normal_vector /= length;
                }
                auto normal_velocity = acc_cellNormalVelocity_buf[edge_id * TOTAY_ZTOP_LAYER + current_layer];
                noraml_vel[kidx][0] = normal_velocity;
                unit_vector[kidx][0] = normal_vector.x();
                unit_vector[kidx][1] = normal_vector.y();
                unit_vector[kidx][2] = normal_vector.z();
            }

            double alpha = Interpolator::compute_alpha(edge_center, pointCount, cellCenter);
            alpha = 1.0;
            Interpolator::mpas_rbf_interp_func_3D_plane_vec_const_dir_comp_coeffs(pointCount, edge_center, unit_vector, cellCenter, alpha, planeBasisVector, coeffs);
            double xVel = 0.0;
            double yVel = 0.0;
            double zVel = 0.0;

            for (auto kidx = 0; kidx < MAX_VERTEX_NUM; ++kidx)
            {
                xVel += coeffs[kidx][0] * noraml_vel[kidx][0];
                yVel += coeffs[kidx][1] * noraml_vel[kidx][0];
                zVel += coeffs[kidx][2] * noraml_vel[kidx][0];
                out << "coeffs[" << kidx << "][0] = " << coeffs[kidx][0] << " coeffs[" << kidx << "][1] = " << coeffs[kidx][1] << " coeffs[" << kidx << "][2] = " << coeffs[kidx][2] << sycl::endl;
            }

            cell_center_velocity = vec3(xVel, yVel, zVel);
            acc_cellCenterVelocity_buf[cell_id * TOTAY_ZTOP_LAYER + current_layer] = cell_center_velocity;

        });
    });
    q.wait_and_throw();

    auto host_accessor = cellCenterVelocity_buf.get_host_access(sycl::read_only);
    (void)host_accessor;
#endif
}

void CalcCellVertexVelocityByZM(
    MPASOGrid* grid,
    int vertex_size,
    int total_ztop_layer,
    const std::vector<double>& cell_vertex_zonal_velocity,
    const std::vector<double>& cell_vertex_meridional_velocity,
    std::vector<vec3>& cell_vertex_velocity,
    const std::vector<size_t>& grid_info,
    sycl::queue& q)
{
#if USE_SYCL
    sycl::buffer<vec3, 1> vertexCoord_buf(grid->vertexCoord_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size()));
    sycl::buffer<vec3, 1> cellCoord_buf(grid->cellCoord_vec.data(), sycl::range<1>(grid->cellCoord_vec.size()));

    sycl::buffer<size_t, 1> numberVertexOnCell_buf(grid->numberVertexOnCell_vec.data(), sycl::range<1>(grid->numberVertexOnCell_vec.size()));

    sycl::buffer<double, 1> cellZonalVelocity_buf(cell_vertex_zonal_velocity.data(), sycl::range<1>(cell_vertex_zonal_velocity.size()));
    sycl::buffer<double, 1> cellMeridionalVelocity_buf(cell_vertex_meridional_velocity.data(), sycl::range<1>(cell_vertex_meridional_velocity.size()));
    sycl::buffer<vec3, 1> cellVertexVelocity_buf(cell_vertex_velocity.data(), sycl::range<1>(cell_vertex_velocity.size()));

    sycl::buffer<size_t, 1> grid_info_buf(grid_info.data(), sycl::range<1>(grid_info.size()));

    q.submit([&](sycl::handler& cgh) {
        auto acc_vertexCoord_buf = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellZonalVelocity_buf = cellZonalVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellMeridionalVelocity_buf = cellMeridionalVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexVelocity_buf = cellVertexVelocity_buf.get_access<sycl::access::mode::write>(cgh);
        auto acc_grid_info_buf          = grid_info_buf.get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for(sycl::range<2>(vertex_size, total_ztop_layer), [=](sycl::id<2> idx) {

            size_t j = idx[0];
            size_t i = idx[1];

            auto vertex_id = j;
            auto current_layer = i;

            const int TOTAY_ZTOP_LAYER = acc_grid_info_buf[4];
            vec3 vertex_center_velocity = { 0.0, 0.0, 0.0 };
            vec3 vertex_position = acc_vertexCoord_buf[vertex_id];
            double tmp_zonal = acc_cellZonalVelocity_buf[vertex_id * TOTAY_ZTOP_LAYER + current_layer];
            double tmp_mer = acc_cellMeridionalVelocity_buf[vertex_id * TOTAY_ZTOP_LAYER + current_layer];
            GeoConverter::convertENUVelocityToXYZ(vertex_position, tmp_zonal, tmp_mer, 0.0, vertex_center_velocity);
            acc_cellVertexVelocity_buf[vertex_id * TOTAY_ZTOP_LAYER + current_layer] = vertex_center_velocity;
        });
    });
    q.wait_and_throw();

    auto host_accessor = cellVertexVelocity_buf.get_host_access(sycl::read_only);
    (void)host_accessor;
#endif
}

void CalcCellVertexVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer,
    const std::vector<vec3>& cell_center_velocity,
    std::vector<vec3>& cell_vertex_velocity,
    const std::vector<size_t>& grid_info,
    sycl::queue& q)
{
#if USE_SYCL
    sycl::buffer<vec3, 1> vertexCoord_buf(grid->vertexCoord_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size()));
    sycl::buffer<vec3, 1> cellCoord_buf(grid->cellCoord_vec.data(), sycl::range<1>(grid->cellCoord_vec.size()));

    sycl::buffer<size_t, 1> numberVertexOnCell_buf(grid->numberVertexOnCell_vec.data(), sycl::range<1>(grid->numberVertexOnCell_vec.size()));
    sycl::buffer<size_t, 1> verticesOnCell_buf(grid->verticesOnCell_vec.data(), sycl::range<1>(grid->verticesOnCell_vec.size()));
    sycl::buffer<size_t, 1> cellsOnVertex_buf(grid->cellsOnVertex_vec.data(), sycl::range<1>(grid->cellsOnVertex_vec.size()));

    sycl::buffer<vec3, 1> cellCenterVelocity_buf(cell_center_velocity.data(), sycl::range<1>(cell_center_velocity.size()));
    sycl::buffer<vec3, 1> cellVertexVelocity_buf(cell_vertex_velocity.data(), sycl::range<1>(grid->vertexCoord_vec.size() * total_ztop_layer));

    sycl::buffer<size_t, 1> grid_info_buf(grid_info.data(), sycl::range<1>(grid_info.size()));

    q.submit([&](sycl::handler& cgh) {

        auto acc_vertexCoord_buf = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnCell_buf = verticesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnVertex_buf = cellsOnVertex_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCenterVelocity_buf = cellCenterVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexVelocity_buf = cellVertexVelocity_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto acc_grid_info_buf          = grid_info_buf.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::range<2>(cells_size, total_ztop_layer), [=](sycl::id<2> idx) {
            size_t j = idx[0];
            size_t i = idx[1];

            auto cell_id = j;
            auto current_layer = i;

            const int CELL_SIZE = acc_grid_info_buf[0];
            const int max_edge = acc_grid_info_buf[2];
            const int MAX_VERTEX_NUM = 20;
            const int NEIGHBOR_NUM = 3;
            const int TOTAY_ZTOP_LAYER = acc_grid_info_buf[4];
            const int VERTLEVELS = acc_grid_info_buf[4];
            auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
            auto nan = std::numeric_limits<size_t>::max();
            size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
            SYCLKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx, MAX_VERTEX_NUM, max_edge, acc_verticesOnCell_buf);
            vec3 current_cell_vertices_value[MAX_VERTEX_NUM];

            for (auto k = 0; k < MAX_VERTEX_NUM; ++k)
            {
                bool bBoundary = false;
                auto vertex_idx = current_cell_vertices_idx[k];
                if (vertex_idx == nan)
                {
                    auto double_nan = std::numeric_limits<double>::quiet_NaN();
                    current_cell_vertices_value[k] = { double_nan , double_nan , double_nan };
                    continue;
                }
                auto current_vertex = acc_vertexCoord_buf[vertex_idx];
                size_t tmp_cell_id[3];
                tmp_cell_id[0] = acc_cellsOnVertex_buf[3 * vertex_idx + 0] - 1;
                tmp_cell_id[1] = acc_cellsOnVertex_buf[3 * vertex_idx + 1] - 1;
                tmp_cell_id[2] = acc_cellsOnVertex_buf[3 * vertex_idx + 2] - 1;
                vec3 tmp_cell_center_vels[3];
                for (auto tmp_cell = 0; tmp_cell < NEIGHBOR_NUM; tmp_cell++)
                {
                    if (tmp_cell_id[tmp_cell] > CELL_SIZE + 1)
                    {
                        tmp_cell_center_vels[tmp_cell] = { 0.0, 0.0, 0.0 };
                        bBoundary = true;
                    }
                    else
                    {
                        auto vel_idx = VERTLEVELS * tmp_cell_id[tmp_cell] + current_layer;
                        tmp_cell_center_vels[tmp_cell] = acc_cellCenterVelocity_buf[vel_idx];
                    }
                }

                if (bBoundary)
                {
                    current_cell_vertices_value[k] = 0.0 * tmp_cell_center_vels[0] + 0.0 * tmp_cell_center_vels[1] + 0.0 * tmp_cell_center_vels[2];
                }
                else
                {
                    double u, v, w;
                    vec3 p1 = acc_cellCoord_buf[tmp_cell_id[0]];
                    vec3 p2 = acc_cellCoord_buf[tmp_cell_id[1]];
                    vec3 p3 = acc_cellCoord_buf[tmp_cell_id[2]];
                    Interpolator::TRIANGLE tri(p1, p2, p3);
                    Interpolator::calcTriangleBarycentric(current_vertex, &tri, u, v, w);
                    current_cell_vertices_value[k] = u * tmp_cell_center_vels[0] + v * tmp_cell_center_vels[1] + w * tmp_cell_center_vels[2];
                }

                acc_cellVertexVelocity_buf[vertex_idx * TOTAY_ZTOP_LAYER + current_layer] = current_cell_vertices_value[k];
            }
            });
        });
    q.wait_and_throw();
    auto host_accessor = cellVertexVelocity_buf.get_host_access(sycl::read_only);
    (void)host_accessor;
#endif
}

void CalcCellVertexVertVelocity(
    MPASOGrid* grid,
    int cells_size,
    int total_ztop_layer_p1,
    const std::vector<double>& cell_center_vert_velocity,
    std::vector<double>& cell_vertex_vert_velocity,
    const std::vector<size_t>& grid_info,
    sycl::queue& q)
{
#if USE_SYCL
    sycl::buffer<vec3, 1> vertexCoord_buf(grid->vertexCoord_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size()));
    sycl::buffer<vec3, 1> cellCoord_buf(grid->cellCoord_vec.data(), sycl::range<1>(grid->cellCoord_vec.size()));

    sycl::buffer<size_t, 1> numberVertexOnCell_buf(grid->numberVertexOnCell_vec.data(), sycl::range<1>(grid->numberVertexOnCell_vec.size()));
    sycl::buffer<size_t, 1> verticesOnCell_buf(grid->verticesOnCell_vec.data(), sycl::range<1>(grid->verticesOnCell_vec.size()));
    sycl::buffer<size_t, 1> cellsOnVertex_buf(grid->cellsOnVertex_vec.data(), sycl::range<1>(grid->cellsOnVertex_vec.size()));

    sycl::buffer<double, 1> cellCenterVertVelocity_buf(cell_center_vert_velocity.data(), sycl::range<1>(cell_center_vert_velocity.size()));
    sycl::buffer<double, 1> cellVertexVertVelocity_buf(cell_vertex_vert_velocity.data(), sycl::range<1>(grid->vertexCoord_vec.size() * total_ztop_layer_p1));

    sycl::buffer<size_t, 1> grid_info_buf(grid_info.data(), sycl::range<1>(grid_info.size()));

    q.submit([&](sycl::handler& cgh) {

        auto acc_vertexCoord_buf = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnCell_buf = verticesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnVertex_buf = cellsOnVertex_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCenterVertVelocity_buf = cellCenterVertVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexVertVelocity_buf = cellVertexVertVelocity_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto acc_grid_info_buf          = grid_info_buf.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::range<2>(cells_size, total_ztop_layer_p1), [=](sycl::id<2> idx) {
            size_t j = idx[0];
            size_t i = idx[1];

            auto cell_id = j;
            auto current_layer = i;

            const int CELL_SIZE = acc_grid_info_buf[0];
            const int max_edge = acc_grid_info_buf[2];
            const int MAX_VERTEX_NUM = 20;
            const int NEIGHBOR_NUM = 3;
            const int TOTAY_ZTOP_LAYER = acc_grid_info_buf[5];
            const int VERTLEVELS = acc_grid_info_buf[5];
            auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
            auto nan = std::numeric_limits<size_t>::max();
            size_t current_cell_vertices_idx[MAX_VERTEX_NUM];
            SYCLKernel::GetCellVerticesIdx(cell_id, current_cell_vertices_number, current_cell_vertices_idx, MAX_VERTEX_NUM, max_edge, acc_verticesOnCell_buf);
            double current_cell_vertices_value[MAX_VERTEX_NUM];
            for (auto k = 0; k < MAX_VERTEX_NUM; ++k)
            {
                bool bBoundary = false;
                auto vertex_idx = current_cell_vertices_idx[k];
                if (vertex_idx == nan)
                {
                    auto double_nan = std::numeric_limits<double>::quiet_NaN();
                    current_cell_vertices_value[k] = { double_nan };
                    continue;
                }
                auto current_vertex = acc_vertexCoord_buf[vertex_idx];
                size_t tmp_cell_id[3];
                tmp_cell_id[0] = acc_cellsOnVertex_buf[3 * vertex_idx + 0] - 1;
                tmp_cell_id[1] = acc_cellsOnVertex_buf[3 * vertex_idx + 1] - 1;
                tmp_cell_id[2] = acc_cellsOnVertex_buf[3 * vertex_idx + 2] - 1;
                double tmp_cell_center_vels[3];
                for (auto tmp_cell = 0; tmp_cell < NEIGHBOR_NUM; tmp_cell++)
                {
                    if (tmp_cell_id[tmp_cell] > CELL_SIZE + 1)
                    {
                        tmp_cell_center_vels[tmp_cell] = { 0.0};
                        bBoundary = true;
                    }
                    else
                    {
                        auto vel_idx = VERTLEVELS * tmp_cell_id[tmp_cell] + current_layer;
                        tmp_cell_center_vels[tmp_cell] = acc_cellCenterVertVelocity_buf[vel_idx];
                    }
                }

                if (bBoundary)
                {
                    current_cell_vertices_value[k] = 0.0 * tmp_cell_center_vels[0] + 0.0 * tmp_cell_center_vels[1] + 0.0 * tmp_cell_center_vels[2];
                }
                else
                {
                    double u, v, w;
                    vec3 p1 = acc_cellCoord_buf[tmp_cell_id[0]];
                    vec3 p2 = acc_cellCoord_buf[tmp_cell_id[1]];
                    vec3 p3 = acc_cellCoord_buf[tmp_cell_id[2]];
                    Interpolator::TRIANGLE tri(p1, p2, p3);
                    Interpolator::calcTriangleBarycentric(current_vertex, &tri, u, v, w);
                    current_cell_vertices_value[k] = u * tmp_cell_center_vels[0] + v * tmp_cell_center_vels[1] + w * tmp_cell_center_vels[2];
                }

                acc_cellVertexVertVelocity_buf[vertex_idx * TOTAY_ZTOP_LAYER + current_layer] = current_cell_vertices_value[k];
            }
            });
        });
    q.wait_and_throw();
    auto host_accessor = cellVertexVertVelocity_buf.get_host_access(sycl::read_only);
    (void)host_accessor;
#endif
}

} // namespace MOPS::GPU::SYCLBackend
