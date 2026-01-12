#include "catch.hpp"
#include "ggl.h"
#include "Utils/Interpolation.hpp"

#ifndef MAX_EDGE
#define MAX_EDGE 3
#endif

TEST_CASE("gauss_elimination_fixed solves 3x3 system", "[mops][gauss]") {
    double A[MAX_EDGE][MAX_EDGE] = {
        {2.0, 3.0, -1.0},
        {4.0, 4.0, -3.0},
        {-2.0, 3.0, 2.0}
    };
    double b[MAX_EDGE] = {5.0, 3.0, 4.0};
    double x[MAX_EDGE] = {0.0};

    MOPS::Interpolator::gauss_elimination_fixed(A, b, 3, x);

    const double expected[MAX_EDGE] = {4.75, 0.5, 6.0};
    const double tol = 1e-6;

    for (int i = 0; i < MAX_EDGE; ++i) {
        INFO("i=" << i << " x=" << x[i] << " expected=" << expected[i]);
        REQUIRE(sycl::fabs(x[i] - expected[i]) <= tol);
    }
}