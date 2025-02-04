#include "ggl.h"
#include "Interpolation.hpp"



//=================================================================
// 1. Verify the Gaussian elimination function
//  
//       2x + 3y -  z = 5
//       4x + 4y - 3z = 3
//      -2x + 3y + 2z = 4

//    Theoretically：x = 4.75, y = 0.5, z = 6.0
int main() 
{
    double A[MAX_EDGE][MAX_EDGE] = {
        {2.0, 3.0, -1.0},
        {4.0, 4.0, -3.0},
        {-2.0, 3.0, 2.0}
    };
    double b[MAX_EDGE] = {5.0, 3.0, 4.0};
    double x[MAX_EDGE] = {0.0};

    Interpolator::gauss_elimination_fixed(A, b, 3, x);
    std::cout << "Computed solution:" << std::endl;
    for (int i = 0; i < MAX_EDGE; ++i) {
        std::cout << "x[" << i << "] = " << x[i] << std::endl;
    }

    double expected[MAX_EDGE] = {4.75, 0.5, 6.0};


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