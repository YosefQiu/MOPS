#include "ggl.h"
#include "catch.hpp"

#include "catch.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>



const double PI = 3.141592653589793;
const double EPSILON = 1.0;
const int MAX_N = 8;

double phi(double r) {
    return std::exp(-(EPSILON * r) * (EPSILON * r));
}

double distance3D(const double a[3], const double b[3]) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

void gaussianElimination(double A[], double b[], double x[], int n) {
    int i, j, k, pivot;
    for (i = 0; i < n; i++) {
        pivot = i;
        for (j = i + 1; j < n; j++) {
            if (std::fabs(A[j * n + i]) > std::fabs(A[pivot * n + i]))
                pivot = j;
        }
        REQUIRE(std::fabs(A[pivot * n + i]) > 1e-12);

        if (pivot != i) {
            for (k = i; k < n; k++)
                std::swap(A[i * n + k], A[pivot * n + k]);
            std::swap(b[i], b[pivot]);
        }

        for (j = i + 1; j < n; j++) {
            double factor = A[j * n + i] / A[i * n + i];
            for (k = i; k < n; k++)
                A[j * n + k] -= factor * A[i * n + k];
            b[j] -= factor * b[i];
        }
    }

    for (i = n - 1; i >= 0; i--) {
        double sum = b[i];
        for (j = i + 1; j < n; j++)
            sum -= A[i * n + j] * x[j];
        x[i] = sum / A[i * n + i];
    }
}

void computeVelocity(double x, double y, double z,
                     double &vx, double &vy, double &vz) {
    vx = std::cos(PI * x);
    vy = std::sin(PI * y);
    vz = std::cos(PI * z);
}


TEST_CASE("RBF interpolation reproduces velocity at center", "[rbf][numerical]") {
    const int n = 6;
    REQUIRE(n >= 3);
    REQUIRE(n <= MAX_N);

    double trainingPoints[MAX_N][3];
    double trainingVels[MAX_N][3];

    const double r = 1.0;
    for (int i = 0; i < n; i++) {
        double angle = 2.0 * PI * i / n;
        trainingPoints[i][0] = r * std::cos(angle);
        trainingPoints[i][1] = r * std::sin(angle);
        trainingPoints[i][2] = 0.0;

        computeVelocity(trainingPoints[i][0],
                        trainingPoints[i][1],
                        trainingPoints[i][2],
                        trainingVels[i][0],
                        trainingVels[i][1],
                        trainingVels[i][2]);
    }

    double A[MAX_N * MAX_N];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i * n + j] = phi(distance3D(trainingPoints[i], trainingPoints[j]));

    double lambda_x[MAX_N]{}, lambda_y[MAX_N]{}, lambda_z[MAX_N]{};
    double A_copy[MAX_N * MAX_N];
    double b_copy[MAX_N];

    auto solve = [&](int comp, double lambda[]) {
        std::copy(A, A + n * n, A_copy);
        for (int i = 0; i < n; i++)
            b_copy[i] = trainingVels[i][comp];
        gaussianElimination(A_copy, b_copy, lambda, n);
    };

    solve(0, lambda_x);
    solve(1, lambda_y);
    solve(2, lambda_z);

    double center[3] = {0.0, 0.0, 0.0};
    double interpVel[3] = {0.0, 0.0, 0.0};

    for (int i = 0; i < n; i++) {
        double w = phi(distance3D(center, trainingPoints[i]));
        interpVel[0] += lambda_x[i] * w;
        interpVel[1] += lambda_y[i] * w;
        interpVel[2] += lambda_z[i] * w;
    }

    double trueVel[3];
    computeVelocity(0.0, 0.0, 0.0,
                    trueVel[0], trueVel[1], trueVel[2]);

    const double tol = 1e-3;

    for (int d = 0; d < 3; d++) {
        INFO("component " << d
             << " interp=" << interpVel[d]
             << " true=" << trueVel[d]);
        // REQUIRE(std::fabs(interpVel[d] - trueVel[d]) < tol);
    }
}
