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
            << " std::isnan=" << sycl::isnan(nanv)
            << " sycl::isnan=" << sycl::isnan(nanv)
            << " std::isfinite=" << std::isfinite(nanv)
            << " sycl::isfinite=" << sycl::isfinite(nanv)
            << std::endl;

    SECTION("Case 1: first point is NaN => trajectory removed, output size = 0")
    {
        std::vector<TrajectoryLine> in;
        in.push_back(make_line({
            CartesianCoord{qnan(), 0.0, 0.0},
            CartesianCoord{5.0, 6.0, 7.0},
        }));

        auto out = MPASOVisualizer::removeNaNTrajectoriesAndReindex(in);
        REQUIRE(out.size() == 0);
    }

    SECTION("Case 2: first valid, second NaN => keep only the first point")
    {
        std::vector<TrajectoryLine> in;
        in.push_back(make_line({
            CartesianCoord{1.0, 2.0, 3.0},
            CartesianCoord{qnan(), 0.0, 0.0},
            CartesianCoord{7.0, 8.0, 9.0}
        }));

        auto out = MPASOVisualizer::removeNaNTrajectoriesAndReindex(in);
        REQUIRE(out.size() == 1);

        REQUIRE(out[0].lineID == 0);
        REQUIRE(out[0].points.size() == 1);
        REQUIRE(out[0].points[0].x() == Approx(1.0));
        REQUIRE(out[0].points[0].y() == Approx(2.0));
        REQUIRE(out[0].points[0].z() == Approx(3.0));

        // lastPoint should be the only remaining point
        REQUIRE(out[0].lastPoint.x() == Approx(1.0));
        REQUIRE(out[0].lastPoint.y() == Approx(2.0));
        REQUIRE(out[0].lastPoint.z() == Approx(3.0));
    }

    SECTION("Case 3: NaN in the middle => truncate before NaN")
    {
        std::vector<TrajectoryLine> in;
        in.push_back(make_line({
            CartesianCoord{10.0, 0.0, 0.0},
            CartesianCoord{11.0, 0.0, 0.0},
            CartesianCoord{qnan(), 0.0, 0.0},
            CartesianCoord{13.0, 0.0, 0.0}
        }));

        auto out = MPASOVisualizer::removeNaNTrajectoriesAndReindex(in);
        REQUIRE(out.size() == 1);

        REQUIRE(out[0].points.size() == 2);
        REQUIRE(out[0].points[0].x() == Approx(10.0));
        REQUIRE(out[0].points[1].x() == Approx(11.0));

        // lastPoint should be the last valid one (x=11)
        REQUIRE(out[0].lastPoint.x() == Approx(11.0));
    }

    SECTION("Case 4: feed cleaned lastPoint back as a new single-point trajectory")
    {
        std::vector<TrajectoryLine> in;
        in.push_back(make_line({
            CartesianCoord{1.0, 0.0, 0.0},
            CartesianCoord{2.0, 0.0, 0.0},
            CartesianCoord{qnan(), 0.0, 0.0},
            CartesianCoord{4.0, 0.0, 0.0}
        }));

        auto out1 = MPASOVisualizer::removeNaNTrajectoriesAndReindex(in);
        REQUIRE(out1.size() == 1);
        REQUIRE(out1[0].points.size() == 2);
        auto lp = out1[0].lastPoint;  

        std::vector<TrajectoryLine> in2;
        in2.push_back(make_line({ lp }));

        auto out2 = MPASOVisualizer::removeNaNTrajectoriesAndReindex(in2);
        REQUIRE(out2.size() == 1);
        REQUIRE(out2[0].points.size() == 1);
        REQUIRE(out2[0].points[0].x() == Approx(2.0));
        REQUIRE(out2[0].lastPoint.x() == Approx(2.0));
    }
}