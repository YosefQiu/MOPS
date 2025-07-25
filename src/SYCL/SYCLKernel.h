#pragma once 
#include "ggl.h"
#include "Utils/GeoConverter.hpp"
#include "Core/MPASOField.h"
namespace MOPS
{
    class SYCLKernel
    {
    public:
        static void SearchKDTree(int* cell_id_vec, MPASOGrid* grid, int width, int height,
                                        double minLat, double maxLat,
                                        double minLon, double maxLon);
        SYCL_EXTERNAL 
        static bool IsInOcean(int cell_id, int max_edge, vec3 current_position, 
            sycl::accessor<size_t, 1, sycl::access::mode::read> acc_numberVertexOnCell_buf,
            sycl::accessor<size_t, 1, sycl::access::mode::read> acc_verticesOnCell_buf, 
            sycl::accessor<vec3, 1, sycl::access::mode::read> acc_vertexCoord_buf);
        SYCL_EXTERNAL
        static void GetCellNeighborsIdx(int cell_id, int current_cell_vertices_number, int* current_cell_neighbors_idx, const int VLA, const int max_edge,
                                        sycl::accessor<size_t, 1, sycl::access::mode::read> acc_cells_onCell_buf);
        SYCL_EXTERNAL
        static void GetCellVerticesIdx(int cell_id, int current_cell_vertices_number, size_t* current_cell_vertices_idx, const int VLA, const int max_edge,
                                        sycl::accessor<size_t, 1, sycl::access::mode::read> acc_verticesOnCell_buf);
        SYCL_EXTERNAL
        static bool GetCellVertexPos(vec3* current_cell_vertex_pos, size_t* current_cell_vertices_idx, const int VLA, int current_cell_vertices_number, sycl::accessor<vec3, 1, sycl::access::mode::read> acc_vertexCoord_buf);

        SYCL_EXTERNAL
        static vec3 CalcVelocity(size_t* current_cell_vertices_idx, double* current_cell_vertex_weight, 
                                    const int VLA, int current_cell_vertices_number, int TOTAY_ZTOP_LAYER, int layer,
                                    sycl::accessor<vec3, 1, sycl::access::mode::read> acc_cellVertexVelocity_buf);

        SYCL_EXTERNAL
        static double CalcAttribute(size_t* current_cell_vertices_idx, double* current_cell_vertex_weight, 
                                    const int VLA, int current_cell_vertices_number, int TOTAY_ZTOP_LAYER, int layer,
                                    sycl::accessor<double, 1, sycl::access::mode::read> acc_cellAttribute_buf);








        SYCL_EXTERNAL
        static vec3 CalcRotationAxis(const vec3& position, const vec3& velocity);
        SYCL_EXTERNAL
        static vec3 CalcPositionAfterRotation(const vec3& position, const vec3& axis, double theta_rad);
    };
}
