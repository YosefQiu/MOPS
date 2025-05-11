#include <iostream>
#include <cmath>
#include <cstdlib>
#include <iomanip>

// 
const double PI = 3.141592653589793;
const double EPSILON = 1.0;  
const int MAX_N = 8;         

// Gaussian radial basis function: phi(r) = exp[ - (EPSILON * r)^2 ]
double phi(double r) {
    return std::exp(- (EPSILON * r) * (EPSILON * r));
}

// Calculate 3D Euclidean distance between two points (both are double[3])
double distance3D(const double a[3], const double b[3]) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Solve linear system A * x = b using Gaussian elimination (with partial pivoting)
// A is an n x n matrix (stored as 1D array, row-major, A[i*n+j] is i-th row, j-th col)
// b and x are arrays of length n
void gaussianElimination(double A[], double b[], double x[], int n) {
    int i, j, k, pivot;
    for (i = 0; i < n; i++) {

        pivot = i;
        for (j = i + 1; j < n; j++) {
            if (std::fabs(A[j * n + i]) > std::fabs(A[pivot * n + i])) {
                pivot = j;
            }
        }
        if (std::fabs(A[pivot * n + i]) < 1e-12) {
            std::cerr << "Matrix is singular or nearly singular!" << std::endl;
            std::exit(1);
        }
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
    for (i = n - 1; i >= 0; i--) {
        double sum = b[i];
        for (j = i + 1; j < n; j++) {
            sum -= A[i * n + j] * x[j];
        }
        x[i] = sum / A[i * n + i];
    }
}

// Example: compute velocity from position
// Define example function: vx = cos(pi*x), vy = sin(pi*y), vz = cos(pi*z)
void computeVelocity(double x, double y, double z, double &vx, double &vy, double &vz) {
    vx = std::cos(PI * x);
    vy = std::sin(PI * y);
    vz = std::cos(PI * z);
}

int main() {
    // --- 1. Set regular n-gon parameters ---
    // n is the number of vertices (can be 5, 6, 7, 8; here we use 6, i.e., regular hexagon)
    int n = 6;
    if(n < 3 || n > MAX_N) {
        std::cerr << "n must be between 3 and " << MAX_N << "!" << std::endl;
        return 1;
    }
    
    // --- 2. Construct training data ---
    // Vertex position array (up to MAX_N vertices, each point is 3D)
    double trainingPoints[MAX_N][3];
    // Vertex velocity array (each point has 3D velocity)
    double trainingVels[MAX_N][3];
    // Construct regular n-gon (vertices are on x-y plane, center at origin, radius 1)
    double r = 1.0;
    for (int i = 0; i < n; i++) {
        double angle = 2.0 * PI * i / n;
        trainingPoints[i][0] = r * std::cos(angle);
        trainingPoints[i][1] = r * std::sin(angle);
        trainingPoints[i][2] = 0.0;
        // Compute velocity at this vertex
        computeVelocity(trainingPoints[i][0], trainingPoints[i][1], trainingPoints[i][2],
                        trainingVels[i][0], trainingVels[i][1], trainingVels[i][2]);
    }
    
    // --- 3. Construct RBF matrix A (n x n) ---
    // A is stored row-major, size n*n, use fixed-size array (max MAX_N*MAX_N)
    double A[MAX_N * MAX_N];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double d = distance3D(trainingPoints[i], trainingPoints[j]);
            A[i * n + j] = phi(d);
        }
    }
    
    // --- 4. Solve RBF interpolation coefficients for each velocity component ---
    // Define coefficient arrays for each component
    double lambda_x[MAX_N], lambda_y[MAX_N], lambda_z[MAX_N];
    // Temporary arrays for Gaussian elimination (fixed size, no dynamic memory)
    double A_copy[MAX_N * MAX_N];
    double b_copy[MAX_N];
    
    // Solve for x component
    for (int i = 0; i < n * n; i++) {
        A_copy[i] = A[i];
    }
    for (int i = 0; i < n; i++) {
        b_copy[i] = trainingVels[i][0];
    }
    gaussianElimination(A_copy, b_copy, lambda_x, n);
    
    // Solve for y component
    for (int i = 0; i < n * n; i++) {
        A_copy[i] = A[i];
    }
    for (int i = 0; i < n; i++) {
        b_copy[i] = trainingVels[i][1];
    }
    gaussianElimination(A_copy, b_copy, lambda_y, n);
    
    // Solve for z component
    for (int i = 0; i < n * n; i++) {
        A_copy[i] = A[i];
    }
    for (int i = 0; i < n; i++) {
        b_copy[i] = trainingVels[i][2];
    }
    gaussianElimination(A_copy, b_copy, lambda_z, n);
    
    // --- 5. Use RBF interpolation to compute velocity at center point ---
    // Assume center point is at (0,0,0)
    double center[3] = {0.0, 0.0, 0.0};
    double interpVel[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < n; i++) {
        double d = distance3D(center, trainingPoints[i]);
        double phi_val = phi(d);
        interpVel[0] += lambda_x[i] * phi_val;
        interpVel[1] += lambda_y[i] * phi_val;
        interpVel[2] += lambda_z[i] * phi_val;
    }
    
    // --- 6. Compute true velocity at center point (using same computeVelocity function) ---
    double trueVel[3];
    computeVelocity(center[0], center[1], center[2], trueVel[0], trueVel[1], trueVel[2]);
    
    // --- 7. Output results ---
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Velocity interpolation result at center point (0,0,0):" << std::endl;
    std::cout << "Interpolated velocity = (" << interpVel[0] << ", " 
              << interpVel[1] << ", " << interpVel[2] << ")" << std::endl;
    std::cout << "True velocity = (" << trueVel[0] << ", " 
              << trueVel[1] << ", " << trueVel[2] << ")" << std::endl;
    std::cout << "Error = (" 
              << std::fabs(interpVel[0] - trueVel[0]) << ", " 
              << std::fabs(interpVel[1] - trueVel[1]) << ", " 
              << std::fabs(interpVel[2] - trueVel[2]) << ")" << std::endl;
    
    return 0;
}
