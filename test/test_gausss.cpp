#include "ggl.h"
#include "Utils/Interpolation.hpp"



// static void gauss_elimination_fixed(double A[MAX_EDGE][MAX_EDGE], double b[MAX_EDGE], int n, double x[MAX_EDGE])
// {
//     int pivot[MAX_EDGE];
//     for (int i = 0; i < n; i++)
//         pivot[i] = i;

//     // Forward elimination
//     for (int j = 0; j < n - 1; ++j) 
//     {
//         // 选择主元
//         int maxRow = j;
//         for (int i = j + 1; i < n; ++i) 
//         {
//             if (sycl::fabs(A[pivot[i]][j]) > sycl::fabs(A[pivot[maxRow]][j]))
//                 maxRow = i;
//         }
//         std::swap(pivot[j], pivot[maxRow]);

//         for (int i = j + 1; i < n; ++i) 
//         {
//             double factor = A[pivot[i]][j] / A[pivot[j]][j];
//             A[pivot[i]][j] = factor; // 存储消元因子
//             for (int k = j + 1; k < n; ++k) 
//             {
//                 A[pivot[i]][k] -= factor * A[pivot[j]][k];
//             }
//             b[pivot[i]] -= factor * b[pivot[j]];
//         }
//     }

//     // Back substitution
//     x[n-1] = b[pivot[n-1]] / A[pivot[n-1]][n-1];
//     for (int i = n - 2; i >= 0; --i) 
//     {
//         double sum = 0.0;
//         for (int j = i + 1; j < n; ++j) 
//         {
//             sum += A[pivot[i]][j] * x[j];
//         }
//         x[i] = (b[pivot[i]] - sum) / A[pivot[i]][i];
//     }
// }

//=================================================================
// 3. 验证高斯消元函数
//    构造如下 3×3 线性系统：
//       2x + 3y -  z = 5
//       4x + 4y - 3z = 3
//      -2x + 3y + 2z = 4
//    理论上，解析解为：x = 4.75, y = 0.5, z = 6.0
int main() {
    // 定义 3x3 系统的矩阵 A 和向量 b
    double A[MAX_EDGE][MAX_EDGE] = {
        {2.0, 3.0, -1.0},
        {4.0, 4.0, -3.0},
        {-2.0, 3.0, 2.0}
    };
    double b[MAX_EDGE] = {5.0, 3.0, 4.0};
    double x[MAX_EDGE] = {0.0};

    // 调用高斯消元函数求解 A * x = b
    MOPS::Interpolator::gauss_elimination_fixed(A, b, 3, x);

    // 输出计算得到的解
    std::cout << "Computed solution:" << std::endl;
    for (int i = 0; i < MAX_EDGE; ++i) {
        std::cout << "x[" << i << "] = " << x[i] << std::endl;
    }

    // 预期解
    double expected[MAX_EDGE] = {4.75, 0.5, 6.0};

    // 验证结果（误差容限为 1e-6）
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