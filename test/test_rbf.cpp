#include <iostream>
#include <cmath>
#include <cstdlib>
#include <iomanip>

// 常量定义
const double PI = 3.141592653589793;
const double EPSILON = 1.0;  // 高斯核形状参数
const int MAX_N = 8;         // 最多 8 个顶点

// 高斯径向基函数：phi(r) = exp[ - (EPSILON * r)^2 ]
double phi(double r) {
    return std::exp(- (EPSILON * r) * (EPSILON * r));
}

// 计算 3D 两点间的欧几里得距离（两点均为 double[3]）
double distance3D(const double a[3], const double b[3]) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// 使用高斯消元法（带部分主元）求解线性方程组 A * x = b  
// A 为 n×n 矩阵（以一维数组存储，行主序，下标 A[i*n+j] 表示第 i 行第 j 列）  
// b 和 x 为长度为 n 的数组  
void gaussianElimination(double A[], double b[], double x[], int n) {
    int i, j, k, pivot;
    // 前向消元
    for (i = 0; i < n; i++) {
        // 选择主元所在行
        pivot = i;
        for (j = i + 1; j < n; j++) {
            if (std::fabs(A[j * n + i]) > std::fabs(A[pivot * n + i])) {
                pivot = j;
            }
        }
        if (std::fabs(A[pivot * n + i]) < 1e-12) {
            std::cerr << "矩阵奇异或接近奇异！" << std::endl;
            std::exit(1);
        }
        // 若主元行不是当前行，则交换第 i 行与 pivot 行（同时交换 b 中对应元素）
        if (pivot != i) {
            for (k = i; k < n; k++) {
                double temp = A[i * n + k];
                A[i * n + k] = A[pivot * n + k];
                A[pivot * n + k] = temp;
            }
            double temp = b[i];
            b[i] = b[pivot];
            b[pivot] = temp;
        }
        // 消去
        for (j = i + 1; j < n; j++) {
            double factor = A[j * n + i] / A[i * n + i];
            for (k = i; k < n; k++) {
                A[j * n + k] -= factor * A[i * n + k];
            }
            b[j] -= factor * b[i];
        }
    }
    // 回代求解
    for (i = n - 1; i >= 0; i--) {
        double sum = b[i];
        for (j = i + 1; j < n; j++) {
            sum -= A[i * n + j] * x[j];
        }
        x[i] = sum / A[i * n + i];
    }
}

// 示例：根据位置计算速度  
// 定义示例函数：vx = cos(pi*x), vy = sin(pi*y), vz = cos(pi*z)
void computeVelocity(double x, double y, double z, double &vx, double &vy, double &vz) {
    vx = std::cos(PI * x);
    vy = std::sin(PI * y);
    vz = std::cos(PI * z);
}

int main() {
    // --- 1. 设置正 n 边形参数 ---
    // n 表示顶点个数（可取 5、6、7、8，本例取 6，即正六边形）
    int n = 6;
    if(n < 3 || n > MAX_N) {
        std::cerr << "n 必须在 3 到 " << MAX_N << " 之间！" << std::endl;
        return 1;
    }
    
    // --- 2. 构造训练数据 ---
    // 顶点位置数组（最多 MAX_N 个顶点，每个点 3 维）
    double trainingPoints[MAX_N][3];
    // 顶点速度数组（每个点 3 维速度）
    double trainingVels[MAX_N][3];
    // 构造正 n 边形（顶点均在 x-y 平面上，圆心在原点，半径为 1）
    double r = 1.0;
    for (int i = 0; i < n; i++) {
        double angle = 2.0 * PI * i / n;
        trainingPoints[i][0] = r * std::cos(angle);
        trainingPoints[i][1] = r * std::sin(angle);
        trainingPoints[i][2] = 0.0;
        // 计算该顶点处的速度
        computeVelocity(trainingPoints[i][0], trainingPoints[i][1], trainingPoints[i][2],
                        trainingVels[i][0], trainingVels[i][1], trainingVels[i][2]);
    }
    
    // --- 3. 构造 RBF 矩阵 A (n x n) ---
    // A 按行主序存储，大小为 n*n，使用固定大小数组（最大为 MAX_N*MAX_N）
    double A[MAX_N * MAX_N];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double d = distance3D(trainingPoints[i], trainingPoints[j]);
            A[i * n + j] = phi(d);
        }
    }
    
    // --- 4. 分别求解各速度分量的 RBF 插值系数 ---
    // 定义每个分量对应的系数数组
    double lambda_x[MAX_N], lambda_y[MAX_N], lambda_z[MAX_N];
    // 定义临时数组，用于高斯消元过程的拷贝（固定大小，不使用动态内存）
    double A_copy[MAX_N * MAX_N];
    double b_copy[MAX_N];
    
    // 对 x 分量求解
    for (int i = 0; i < n * n; i++) {
        A_copy[i] = A[i];
    }
    for (int i = 0; i < n; i++) {
        b_copy[i] = trainingVels[i][0];
    }
    gaussianElimination(A_copy, b_copy, lambda_x, n);
    
    // 对 y 分量求解
    for (int i = 0; i < n * n; i++) {
        A_copy[i] = A[i];
    }
    for (int i = 0; i < n; i++) {
        b_copy[i] = trainingVels[i][1];
    }
    gaussianElimination(A_copy, b_copy, lambda_y, n);
    
    // 对 z 分量求解
    for (int i = 0; i < n * n; i++) {
        A_copy[i] = A[i];
    }
    for (int i = 0; i < n; i++) {
        b_copy[i] = trainingVels[i][2];
    }
    gaussianElimination(A_copy, b_copy, lambda_z, n);
    
    // --- 5. 在中心点处利用 RBF 插值计算速度 ---
    // 假定中心点位置为 (0,0,0)
    double center[3] = {0.0, 0.0, 0.0};
    double interpVel[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < n; i++) {
        double d = distance3D(center, trainingPoints[i]);
        double phi_val = phi(d);
        interpVel[0] += lambda_x[i] * phi_val;
        interpVel[1] += lambda_y[i] * phi_val;
        interpVel[2] += lambda_z[i] * phi_val;
    }
    
    // --- 6. 计算中心点处的真实速度（用相同的 computeVelocity 函数） ---
    double trueVel[3];
    computeVelocity(center[0], center[1], center[2], trueVel[0], trueVel[1], trueVel[2]);
    
    // --- 7. 输出结果 ---
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "中心点 (0,0,0) 处的速度插值结果：" << std::endl;
    std::cout << "插值速度 = (" << interpVel[0] << ", " 
              << interpVel[1] << ", " << interpVel[2] << ")" << std::endl;
    std::cout << "真实速度 = (" << trueVel[0] << ", " 
              << trueVel[1] << ", " << trueVel[2] << ")" << std::endl;
    std::cout << "误差 = (" 
              << std::fabs(interpVel[0] - trueVel[0]) << ", " 
              << std::fabs(interpVel[1] - trueVel[1]) << ", " 
              << std::fabs(interpVel[2] - trueVel[2]) << ")" << std::endl;
    
    return 0;
}
