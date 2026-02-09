#include "catch.hpp"
#include "ggl.h"
#include "api/MOPS.h"
#include "Core/MPASOVisualizer.h"
#include <iostream>
using namespace MOPS;

static inline double qnan() {
    return std::numeric_limits<double>::quiet_NaN();
}

static TrajectoryLine make_line(std::initializer_list<CartesianCoord> pts, int id = 42)
{
    TrajectoryLine line;
    line.lineID = id;
    line.points.assign(pts.begin(), pts.end());

    line.velocity.resize(line.points.size(), CartesianCoord{1.0, 2.0, 3.0});
    line.temperature.resize(line.points.size(), 10.0);
    line.salinity.resize(line.points.size(), 20.0);

    if (!line.points.empty()) line.lastPoint = line.points.back();
    return line;
}

TEST_CASE("removeNaNTrajectoriesAndReindex - NaN handling and truncation")
{

    auto nanv = qnan();
    std::cout << "nanv=" << nanv
            << " sycl::isnan=" << sycl::isnan(nanv)
            << " sycl::isnan=" << sycl::isnan(nanv)
            << " std::isfinite=" << std::isfinite(nanv)
            << " sycl::isfinite=" << sycl::isfinite(nanv)
            << std::endl;

    SECTION("Case 1: first point is NaN => trajectory kept, all points filled with first point, velocity=0")
    {
        std::vector<TrajectoryLine> in;
        // Assume trajectory length = 4, first point is NaN
        in.push_back(make_line({
            CartesianCoord{qnan(), 0.0, 0.0},
            CartesianCoord{5.0, 6.0, 7.0},
            CartesianCoord{8.0, 9.0, 10.0},
            CartesianCoord{11.0, 12.0, 13.0},
        }));

        auto out = MPASOVisualizer::removeNaNTrajectoriesAndReindex(in);
        REQUIRE(out.size() == 1);  // Trajectory is kept, not removed
        REQUIRE(out[0].points.size() == 4);  // Original length preserved

        // All points should be filled with the first point (which is NaN)
        // Since first point has x=NaN, all points will have x=NaN
        REQUIRE(sycl::isnan(out[0].points[0].x()));
        REQUIRE(sycl::isnan(out[0].points[1].x()));
        REQUIRE(sycl::isnan(out[0].points[2].x()));
        REQUIRE(sycl::isnan(out[0].points[3].x()));

        // All velocities should be 0
        for (size_t i = 0; i < 4; ++i) {
            REQUIRE(out[0].velocity[i].x() == Approx(0.0));
            REQUIRE(out[0].velocity[i].y() == Approx(0.0));
            REQUIRE(out[0].velocity[i].z() == Approx(0.0));
        }
    }

    SECTION("Case 2: first valid, second NaN => keep original length, pad with last valid point, velocity=0")
    {
        std::vector<TrajectoryLine> in;
        // Assume trajectory length = 4, NaN at index 1
        in.push_back(make_line({
            CartesianCoord{1.0, 2.0, 3.0},
            CartesianCoord{qnan(), 0.0, 0.0},
            CartesianCoord{7.0, 8.0, 9.0},
            CartesianCoord{10.0, 11.0, 12.0}
        }));

        auto out = MPASOVisualizer::removeNaNTrajectoriesAndReindex(in);
        REQUIRE(out.size() == 1);

        REQUIRE(out[0].lineID == 0);
        REQUIRE(out[0].points.size() == 4);  // Original length preserved

        // First point unchanged
        REQUIRE(out[0].points[0].x() == Approx(1.0));
        REQUIRE(out[0].points[0].y() == Approx(2.0));
        REQUIRE(out[0].points[0].z() == Approx(3.0));

        // Points 1, 2, 3 should all be padded with last valid point (first point)
        for (size_t i = 1; i < 4; ++i) {
            REQUIRE(out[0].points[i].x() == Approx(1.0));
            REQUIRE(out[0].points[i].y() == Approx(2.0));
            REQUIRE(out[0].points[i].z() == Approx(3.0));
        }

        // Velocity at index 0 (before NaN) should be 0
        REQUIRE(out[0].velocity[0].x() == Approx(0.0));
        REQUIRE(out[0].velocity[0].y() == Approx(0.0));
        REQUIRE(out[0].velocity[0].z() == Approx(0.0));

        // All velocities from NaN point onward should be 0
        for (size_t i = 1; i < 4; ++i) {
            REQUIRE(out[0].velocity[i].x() == Approx(0.0));
            REQUIRE(out[0].velocity[i].y() == Approx(0.0));
            REQUIRE(out[0].velocity[i].z() == Approx(0.0));
        }

        // lastPoint should be the last valid point (first point)
        REQUIRE(out[0].lastPoint.x() == Approx(1.0));
        REQUIRE(out[0].lastPoint.y() == Approx(2.0));
        REQUIRE(out[0].lastPoint.z() == Approx(3.0));
    }

    SECTION("Case 3: NaN in the middle => keep original length, pad with last valid point, velocity=0")
    {
        std::vector<TrajectoryLine> in;
        // Assume trajectory length = 5, NaN at index 2
        in.push_back(make_line({
            CartesianCoord{10.0, 1.0, 1.0},
            CartesianCoord{11.0, 2.0, 2.0},
            CartesianCoord{qnan(), 0.0, 0.0},
            CartesianCoord{13.0, 4.0, 4.0},
            CartesianCoord{14.0, 5.0, 5.0}
        }));

        auto out = MPASOVisualizer::removeNaNTrajectoriesAndReindex(in);
        REQUIRE(out.size() == 1);

        REQUIRE(out[0].points.size() == 5);  // Original length preserved

        // First two points unchanged
        REQUIRE(out[0].points[0].x() == Approx(10.0));
        REQUIRE(out[0].points[0].y() == Approx(1.0));
        REQUIRE(out[0].points[1].x() == Approx(11.0));
        REQUIRE(out[0].points[1].y() == Approx(2.0));

        // Points 2, 3, 4 should be padded with last valid point (index 1, x=11)
        for (size_t i = 2; i < 5; ++i) {
            REQUIRE(out[0].points[i].x() == Approx(11.0));
            REQUIRE(out[0].points[i].y() == Approx(2.0));
            REQUIRE(out[0].points[i].z() == Approx(2.0));
        }

        // Velocity at index 1 (before NaN) should be 0
        REQUIRE(out[0].velocity[1].x() == Approx(0.0));
        REQUIRE(out[0].velocity[1].y() == Approx(0.0));
        REQUIRE(out[0].velocity[1].z() == Approx(0.0));

        // Velocities at NaN positions and after should be 0
        for (size_t i = 2; i < 5; ++i) {
            REQUIRE(out[0].velocity[i].x() == Approx(0.0));
            REQUIRE(out[0].velocity[i].y() == Approx(0.0));
            REQUIRE(out[0].velocity[i].z() == Approx(0.0));
        }

        // lastPoint should be the last valid one (x=11)
        REQUIRE(out[0].lastPoint.x() == Approx(11.0));
        REQUIRE(out[0].lastPoint.y() == Approx(2.0));
    }

    SECTION("Case 4: all points valid => no changes, original length and values preserved")
    {
        std::vector<TrajectoryLine> in;
        // Assume trajectory length = 4, all points valid
        in.push_back(make_line({
            CartesianCoord{1.0, 1.0, 1.0},
            CartesianCoord{2.0, 2.0, 2.0},
            CartesianCoord{3.0, 3.0, 3.0},
            CartesianCoord{4.0, 4.0, 4.0}
        }));

        auto out = MPASOVisualizer::removeNaNTrajectoriesAndReindex(in);
        REQUIRE(out.size() == 1);
        REQUIRE(out[0].points.size() == 4);  // Original length preserved

        // All points unchanged
        REQUIRE(out[0].points[0].x() == Approx(1.0));
        REQUIRE(out[0].points[1].x() == Approx(2.0));
        REQUIRE(out[0].points[2].x() == Approx(3.0));
        REQUIRE(out[0].points[3].x() == Approx(4.0));

        // Velocities unchanged (original value from make_line is {1.0, 2.0, 3.0})
        for (size_t i = 0; i < 4; ++i) {
            REQUIRE(out[0].velocity[i].x() == Approx(1.0));
            REQUIRE(out[0].velocity[i].y() == Approx(2.0));
            REQUIRE(out[0].velocity[i].z() == Approx(3.0));
        }

        // lastPoint should be the last point
        REQUIRE(out[0].lastPoint.x() == Approx(4.0));
        REQUIRE(out[0].lastPoint.y() == Approx(4.0));
        REQUIRE(out[0].lastPoint.z() == Approx(4.0));
    }
}