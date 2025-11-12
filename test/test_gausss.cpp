#include "ggl.h"
#include "Utils/Interpolation.hpp"



int main() {
    // Define the matrix A and vector b of the 3x3 system
    double A[MAX_EDGE][MAX_EDGE] = {
        {2.0, 3.0, -1.0},
        {4.0, 4.0, -3.0},
        {-2.0, 3.0, 2.0}
    };
    double b[MAX_EDGE] = {5.0, 3.0, 4.0};
    double x[MAX_EDGE] = {0.0};

    // Call the Gaussian elimination function to solve A * x = b
    MOPS::Interpolator::gauss_elimination_fixed(A, b, 3, x);

    // Output the computed solution
    std::cout << "Computed solution:" << std::endl;
    for (int i = 0; i < MAX_EDGE; ++i) {
        std::cout << "x[" << i << "] = " << x[i] << std::endl;
    }

    // Expected solution
    double expected[MAX_EDGE] = {4.75, 0.5, 6.0};

    // Verify the result (tolerance 1e-6)
    double tol = 1e-6;
    for (int i = 0; i < MAX_EDGE; ++i) {
        if (sycl::fabs(x[i] - expected[i]) > tol) {
            std::cerr << "Error: x[" << i << "] differs from expected value." << std::endl;
            return -1;
        }
    }
    std::cout << "Test passed: The computed solution matches the expected result." << std::endl;

    return 0;
}