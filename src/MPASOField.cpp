#include "MPASOField.h"

void MPASOField::initField(MPASOGrid* grid, MPASOSolution* sol_1, MPASOSolution* sol_2)
{
	mGrid = std::shared_ptr<MPASOGrid>(grid);
	mSol_Front = std::shared_ptr<MPASOSolution>(sol_1);
	mSol_Back = sol_2 == nullptr ? nullptr : std::shared_ptr<MPASOSolution>(sol_2);
}

void MPASOField::initField(std::shared_ptr<MPASOGrid> grid, std::shared_ptr<MPASOSolution> sol_1, std::shared_ptr<MPASOSolution> sol_2)
{
	mGrid = grid;
	mSol_Front = sol_1;
	mSol_Back = sol_2;
}

void MPASOField::calcInWhichCells(const vec3& position, int& cell_id)
{
	mGrid->searchKDT(position, cell_id);
}

void MPASOField::calcInWhichCells(std::vector<CartesianCoord>& points_vec, std::vector<int>& cell_id_vec)
{
    if (!cell_id_vec.empty()) cell_id_vec.clear();
    cell_id_vec.resize(points_vec.size());

    for (auto i = 0; i < points_vec.size(); i++)
    {
        auto pos = points_vec[i]; int cell_id = -1;
        calcInWhichCells(pos, cell_id);
        cell_id_vec[i] = cell_id;
    }
}

bool MPASOField::isOnOcean(const vec3& position, int& cell_id, std::vector<size_t>& current_cell_vertices_idx)
{
    // 2. 判断是否在大陆上
        bool is_land = false; // 标记是否为陆地
#pragma region cell_or_not
                //      2.1 找到这个CELL的所有顶点
    auto current_cell_vertices_number = mGrid->numberVertexOnCell_vec[cell_id];
    mGrid->getVerticesOnCell(cell_id, mGrid->verticesOnCell_vec, current_cell_vertices_idx);
    //      2.2 删除不存在的顶点
    if (current_cell_vertices_idx.size() != current_cell_vertices_number)
    {
        current_cell_vertices_idx.resize(current_cell_vertices_number);
    }

    std::vector<double> normalsConsistency;
    if (!normalsConsistency.empty()) normalsConsistency.clear();
    for (auto k = 0; k < current_cell_vertices_idx.size(); ++k)
    {
        auto A_idx = current_cell_vertices_idx[k];
        auto B_idx = current_cell_vertices_idx[(k + 1) % current_cell_vertices_idx.size()];
        auto A = mGrid->vertexCoord_vec[A_idx];
        auto B = mGrid->vertexCoord_vec[B_idx];
        vec3 O(0.0, 0.0, 0.0);
        auto AO = O - A;
        auto BO = O - B;
        auto A_point = position - A;
        vec3 surface_normal = YOSEF_CROSS(AO, BO);
        double direction = YOSEF_DOT(surface_normal, A_point);
        normalsConsistency.push_back(direction);
    }

    double sign = (normalsConsistency[0] > 0) ? 1.0 : -1.0;

    for (double dir : normalsConsistency)
    {
        double currentSign = (dir > 0) ? 1.0 : -1.0;
        if (currentSign != sign)
        {
            // 这个点在大陆上
            //pixel_color = vec3(0.0, 0.0, 0.0);
            is_land = true;
            break;
        }
    }
#pragma endregion cell_or_not
    return is_land;
}
