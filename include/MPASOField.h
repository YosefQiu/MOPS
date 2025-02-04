#pragma once
#include "ggl.h"
#include "MPASOGrid.h"
#include "MPASOSolution.h"

class MPASOField
{
public:
	MPASOField() = default;
public:
	void initField(MPASOGrid* grid, MPASOSolution* sol_1, MPASOSolution* sol_2 = nullptr);
	void initField(std::shared_ptr<MPASOGrid> grid, std::shared_ptr<MPASOSolution> sol_1, std::shared_ptr<MPASOSolution> sol_2 = nullptr);
	// =============
	void calcInWhichCells(const vec3& position, int& cell_id);
	void calcInWhichCells(std::vector<CartesianCoord>& points_vec, std::vector<int>& cell_id_vec);
	bool isOnOcean(const vec3& position, int& cell_id, std::vector<size_t>& current_cell_vertices_idx);
public:
	std::shared_ptr<MPASOGrid> mGrid;
	std::shared_ptr<MPASOSolution> mSol_Front;
	std::shared_ptr<MPASOSolution> mSol_Back;
};

