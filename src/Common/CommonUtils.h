#pragma once

#include "Core/MPASOGrid.h"

namespace MOPS::Common {

inline std::vector<size_t> BuildGridInfo(const MPASOGrid* grid)
{
    std::vector<size_t> info;
    info.reserve(6);
    info.push_back(static_cast<size_t>(grid->mCellsSize));
    info.push_back(static_cast<size_t>(grid->mEdgesSize));
    info.push_back(static_cast<size_t>(grid->mMaxEdgesSize));
    info.push_back(static_cast<size_t>(grid->mVertexSize));
    info.push_back(static_cast<size_t>(grid->mVertLevels));
    info.push_back(static_cast<size_t>(grid->mVertLevelsP1));
    return info;
}

} // namespace MOPS::Common
