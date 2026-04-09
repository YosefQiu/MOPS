#pragma once

#include "Core/MPASOVisualizer.h"
#include "Common/CommonUtils.h"
#include <tbb/parallel_for.h>

namespace MOPS::Common {

struct TrajectoryOutputBuffers {
    std::vector<vec3> points;
    std::vector<vec3> velocities;
    std::vector<vec3> attrs;
};

inline TrajectoryOutputBuffers InitTrajectoryOutputBuffers(size_t particle_count, int simulation_duration, int record_t, bool with_attrs)
{
    const size_t per_particle_points = static_cast<size_t>(simulation_duration / record_t);
    const size_t total = particle_count * per_particle_points;

    TrajectoryOutputBuffers buffers;
    buffers.points.resize(total);
    buffers.velocities.resize(total);
    if (with_attrs) {
        buffers.attrs.resize(total);
    }
    return buffers;
}

inline std::vector<float> BuildEffectiveDepths(const std::vector<vec3>& stable_points, const TrajectorySettings* config, const char* stage)
{
    std::vector<float> effective_depths(stable_points.size());
    const bool per_particle = config->hasPerParticleDepths() && config->particle_depths.size() == stable_points.size();
    if (per_particle) {
        effective_depths = config->particle_depths;
        Debug("[%s] Using per-particle depths (%zu particles)", stage, stable_points.size());
    } else {
        std::fill(effective_depths.begin(), effective_depths.end(), config->depth);
        Debug("[%s] Using uniform depth: %.2f meters", stage, config->depth);
    }
    return effective_depths;
}

inline std::vector<TrajectoryLine> InitTrajectoryLines(const std::vector<vec3>& stable_points, const std::vector<float>& effective_depths, const TrajectorySettings* config)
{
    std::vector<TrajectoryLine> trajectory_lines(stable_points.size());
    tbb::parallel_for(size_t(0), stable_points.size(), [&](size_t i) {
        trajectory_lines[i].lineID = i;
        trajectory_lines[i].points.push_back(stable_points[i]);
        trajectory_lines[i].lastPoint = stable_points[i];
        trajectory_lines[i].duration = config->simulationDuration;
        trajectory_lines[i].timestamp = config->deltaT;
        trajectory_lines[i].depth = effective_depths[i];
    });
    return trajectory_lines;
}

inline std::vector<TrajectoryLine> RemoveNaNTrajectoriesAndReindex(std::vector<TrajectoryLine>& trajectory_lines)
{
    const size_t n = trajectory_lines.size();
    std::vector<size_t> cut(n, 0);

    auto is_valid = [](const CartesianCoord& p) -> bool {
        return MOPS::math::isfinite(p.x()) && MOPS::math::isfinite(p.y()) && MOPS::math::isfinite(p.z());
    };

    tbb::parallel_for(size_t(0), n, [&](size_t i) {
        const auto& line = trajectory_lines[i];
        const size_t len = line.points.size();
        size_t k = 0;
        for (; k < len; ++k) {
            if (!is_valid(line.points[k])) break;
        }
        cut[i] = k;
    });

    std::vector<TrajectoryLine> cleaned;
    cleaned.reserve(n);

    int new_id = 0;
    for (size_t i = 0; i < n; ++i) {
        auto& line = trajectory_lines[i];
        const size_t original_len = line.points.size();

        if (original_len == 0) {
            continue;
        }

        line.velocity.resize(original_len, CartesianCoord{0.0, 0.0, 0.0});
        line.temperature.resize(original_len, 0.0);
        line.salinity.resize(original_len, 0.0);

        size_t k = cut[i];

        if (k == 0) {
            CartesianCoord first_pos = line.points[0];
            CartesianCoord zero_vel = {0.0, 0.0, 0.0};
            double first_temp = line.temperature[0];
            double first_sal = line.salinity[0];

            for (size_t j = 0; j < original_len; ++j) {
                line.points[j] = first_pos;
                line.velocity[j] = zero_vel;
                line.temperature[j] = first_temp;
                line.salinity[j] = first_sal;
            }
        }
        else if (k < original_len) {
            CartesianCoord last_valid_pos = line.points[k - 1];
            CartesianCoord zero_vel = {0.0, 0.0, 0.0};
            double last_temp = line.temperature[k - 1];
            double last_sal = line.salinity[k - 1];

            line.velocity[k - 1] = zero_vel;

            for (size_t j = k; j < original_len; ++j) {
                line.points[j] = last_valid_pos;
                line.velocity[j] = zero_vel;
                line.temperature[j] = last_temp;
                line.salinity[j] = last_sal;
            }
        }

        line.lastPoint = line.points.back();
        line.lineID = new_id++;
        cleaned.push_back(line);
    }

    return cleaned;
}

template <typename PosAccessor, typename VelAccessor>
inline std::vector<TrajectoryLine> FinalizeTrajectoryLines(
    std::vector<TrajectoryLine>& trajectory_lines,
    const PosAccessor& positions,
    const VelAccessor& velocities,
    size_t each_point_size,
    size_t total_points)
{
    const size_t total_lines = trajectory_lines.size();
    tbb::parallel_for(size_t(0), total_lines, [&](size_t line_idx) {
        size_t start_idx = line_idx * each_point_size;
        size_t end_idx = std::min(start_idx + each_point_size, total_points);

        for (size_t i = start_idx; i < end_idx; ++i) {
            vec3 p = positions[i];
            vec3 v = velocities[i];
            trajectory_lines[line_idx].points.push_back(p);
            trajectory_lines[line_idx].velocity.push_back(v);

            if (i == end_idx - 1 || i == total_points - 1) {
                trajectory_lines[line_idx].lastPoint = p;
            }
        }
    });

    return RemoveNaNTrajectoriesAndReindex(trajectory_lines);
}

template <typename PosAccessor, typename VelAccessor, typename AttrAccessor>
inline std::vector<TrajectoryLine> FinalizeTrajectoryLinesWithAttrs(
    std::vector<TrajectoryLine>& trajectory_lines,
    const PosAccessor& positions,
    const VelAccessor& velocities,
    const AttrAccessor& attrs,
    size_t each_point_size,
    size_t total_points)
{
    const size_t total_lines = trajectory_lines.size();
    tbb::parallel_for(size_t(0), total_lines, [&](size_t line_idx) {
        size_t start_idx = line_idx * each_point_size;
        size_t end_idx = std::min(start_idx + each_point_size, total_points);

        for (size_t i = start_idx; i < end_idx; ++i) {
            vec3 p = positions[i];
            vec3 v = velocities[i];
            vec3 a = attrs[i];
            trajectory_lines[line_idx].points.push_back(p);
            trajectory_lines[line_idx].velocity.push_back(v);
            trajectory_lines[line_idx].temperature.push_back(v.x());
            trajectory_lines[line_idx].salinity.push_back(v.y());

            if (i == end_idx - 1 || i == total_points - 1) {
                trajectory_lines[line_idx].lastPoint = p;
            }
            (void)a;
        }
    });

    return RemoveNaNTrajectoriesAndReindex(trajectory_lines);
}

} // namespace MOPS::Common
