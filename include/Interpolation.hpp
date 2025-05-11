#pragma once
#include "ggl.h"
#include <iostream>

#define EARTH_RADIUS            6371.01 // Earth's radius in kilometers

namespace MOPS
{
    class Interpolator
    {
    public:
        struct TRIANGLE
        {
            vec3 v[3];
            TRIANGLE() = default;
            TRIANGLE(vec3 v1, vec3 v2, vec3 v3)
            {
                v[0] = v1;
                v[1] = v2;
                v[2] = v3;
            }
        };
        static double calcTriangleArea(TRIANGLE& tri)
        {
            auto A = tri.v[0];
            auto B = tri.v[1];
            auto C = tri.v[2];

            auto AB = B - A;
            auto AC = C - A;

            auto tmp_cross = YOSEF_CROSS(AB, AC);
            auto tmp_length = YOSEF_LENGTH(tmp_cross);
            return 0.5 * tmp_length;
        }
        /*
        static float calcSphereDist(const vec2f& v1, const vec2f& v2)
        {
            auto theta1 = v1.x;
            auto phi1 = v1.y;
            auto theta2 = v2.x;
            auto phi2 = v2.y;

            auto r = EARTH_RADIUS * 1000.0f;
            float d = r * acos(sin(theta1) * sin(theta2) + cos(theta1) * cos(theta2) + cos(abs(phi2 - phi1)));
            return d;
        }

        static float calcLinearInterpolate(float a, float b, float t)
        {
            return a + (b - a) * t;
        }

        static vec2f calcLinearInterpolate(const vec2f& a, const vec2f& b, float t)
        {
            return vec2f(
                Interpolator::calcLinearInterpolate(a.x, b.x, t),
                Interpolator::calcLinearInterpolate(a.y, b.y, t)
            );
        }
        
        static vec3f calcLinearInterpolate(const vec3f& a, const vec3f& b, float t)
        {
            return vec3f(
                Interpolator::calcLinearInterpolate(a.x, b.x, t),
                Interpolator::calcLinearInterpolate(a.y, b.y, t),
                Interpolator::calcLinearInterpolate(a.z, b.z, t)
            );
        }
        static float calcBiLinearInterpolate(float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2, float x, float y) {
            float x2x1 = x2 - x1, y2y1 = y2 - y1, x2x = x2 - x, y2y = y2 - y, yy1 = y - y1, xx1 = x - x1;
            return 1.0f / (x2x1 * y2y1) * (
                q11 * x2x * y2y +
                q21 * xx1 * y2y +
                q12 * x2x * yy1 +
                q22 * xx1 * yy1
                );
        }*/

        static void calcTriangleBarycentric(const vec3& p, TRIANGLE* tri, double& u, double& v, double& w)
        {
            auto v0 = tri->v[1] - tri->v[0];
            auto v1 = tri->v[2] - tri->v[0];
            auto v2 = p - tri->v[0];
            double d00 = YOSEF_DOT(v0, v0);
            double d01 = YOSEF_DOT(v0, v1);
            double d11 = YOSEF_DOT(v1, v1);
            double d20 = YOSEF_DOT(v2, v0);
            double d21 = YOSEF_DOT(v2, v1);
            double denom = d00 * d11 - d01 * d01;
            v = (d11 * d20 - d01 * d21) / denom;
            w = (d00 * d21 - d01 * d20) / denom;
            u = 1.0 - v - w;
        }

        static double triangle_area(const vec3& a, const vec3& b, const vec3& c)
        {
            // Calculate the vectors for two edges of the triangle
            vec3 edge1 = {b.x() - a.x(), b.y() - a.y(), b.z() - a.z()};
            vec3 edge2 = {c.x() - a.x(), c.y() - a.y(), c.z() - a.z()};

            // Calculate the cross product of the two edge vectors
            vec3 cross_product = {
                edge1.y() * edge2.z() - edge1.z() * edge2.y(),
                edge1.z() * edge2.x() - edge1.x() * edge2.z(),
                edge1.x() * edge2.y() - edge1.y() * edge2.x()
            };

            // The area of the triangle is half the magnitude of the cross product vector
            return sqrt(cross_product.x() * cross_product.x() + cross_product.y() * cross_product.y() + cross_product.z() * cross_product.z()) / 2.0f;
        }

        static void CalcPolygonWachspress(const vec3& p, std::vector<vec3>& poly, std::vector<double>& weights)
        {
            int N = poly.size();
            weights.clear();
            weights.resize(N, 0.0);
            double sumweights = 0.0;
            double A_i, A_iplus1, B;
            
            A_iplus1 = triangle_area(poly[N - 1], poly[0], p);
            for(int i = 0; i < N; i++) {
                A_i = A_iplus1;
                A_iplus1 = triangle_area(poly[i], poly[(i + 1) % N], p);
                
                B = triangle_area(poly[(i - 1 + N) % N], poly[i], poly[(i + 1) % N]);
                
                weights[i] = B / (A_i * A_iplus1);
                sumweights += weights[i];
            }
            
            // Normalize the weights
            double recp = 1.0 / sumweights;
            for(int i = 0; i < N; i++) {
                weights[i] *= recp;
            }
        }
        static void CalcPolygonWachspress(const vec3& p, vec3* poly, double* weights, const int vertex_number)
        {
            int N = vertex_number;
            
            for (int i = 0; i < N; i++) {
                weights[i] = 0.0;
            }

            double sumweights = 0.0;
            double A_i, A_iplus1, B;

    
            A_iplus1 = triangle_area(poly[N - 1], poly[0], p);
            for (int i = 0; i < N; i++) {
                A_i = A_iplus1;
                A_iplus1 = triangle_area(poly[i], poly[(i + 1) % N], p);

                B = triangle_area(poly[(i - 1 + N) % N], poly[i], poly[(i + 1) % N]);

                weights[i] = B / (A_i * A_iplus1);
                sumweights += weights[i];
            }

            // Normalize the weights
            double recp = 1.0 / sumweights;
            for (int i = 0; i < N; i++) {
                weights[i] *= recp;
            }
        }

    // RBF interpolation
    #define MAX_EDGE 8
        static double evaluate_rbf(double rSquared)
        {
            return 1.0 / sycl::sqrt(1.0 + rSquared);
        }

        static void gauss_elimination_fixed(double A[MAX_EDGE][MAX_EDGE], double b[MAX_EDGE], int n, double x[MAX_EDGE])
        {
            int pivot[MAX_EDGE];
            for (int i = 0; i < n; i++)
                pivot[i] = i;

            // Forward elimination
            for (int j = 0; j < n - 1; ++j) 
            {
                // 选择主元
                int maxRow = j;
                for (int i = j + 1; i < n; ++i) 
                {
                    if (sycl::fabs(A[pivot[i]][j]) > sycl::fabs(A[pivot[maxRow]][j]))
                        maxRow = i;
                }
                std::swap(pivot[j], pivot[maxRow]);

                for (int i = j + 1; i < n; ++i) 
                {
                    double factor = A[pivot[i]][j] / A[pivot[j]][j];
                    A[pivot[i]][j] = factor; // 存储消元因子
                    for (int k = j + 1; k < n; ++k) 
                    {
                        A[pivot[i]][k] -= factor * A[pivot[j]][k];
                    }
                    b[pivot[i]] -= factor * b[pivot[j]];
                }
            }

            // Back substitution
            x[n-1] = b[pivot[n-1]] / A[pivot[n-1]][n-1];
            for (int i = n - 2; i >= 0; --i) 
            {
                double sum = 0.0;
                for (int j = i + 1; j < n; ++j) 
                {
                    sum += A[pivot[i]][j] * x[j];
                }
                x[i] = (b[pivot[i]] - sum) / A[pivot[i]][i];
            }
        }
        

        static double compute_alpha(const double sourcePoints[MAX_EDGE][3], int pointCount, const double cellCenter[3])
        {
            double sum = 0.0;
            for (int i = 0; i < pointCount; ++i) 
            {
                double dx = sourcePoints[i][0] - cellCenter[0];
                double dy = sourcePoints[i][1] - cellCenter[1];
                double dz = sourcePoints[i][2] - cellCenter[2];
                double r = sycl::sqrt(dx * dx + dy * dy + dz * dz);
                sum += r;
            }
            return (pointCount > 0) ? sum / pointCount : 1.0;
        }

        static void mpas_rbf_interp_func_3D_plane_vec_const_dir_comp_coeffs(
            int pointCount,
            const double sourcePoints[MAX_EDGE][3],
            const double unitVectors[MAX_EDGE][3],
            const double destinationPoint[3],
            double alpha,
            const double planeBasisVectors[2][3],
            double coefficients[MAX_EDGE][3]
        )
        {
            // 先将 3D 源点与单位向量投影到给定平面（2D）
            double planarSourcePoints[MAX_EDGE][2] = {0};
            double planarUnitVectors[MAX_EDGE][2] = {0};
            double planarDestinationPoint[2] = {0};

            for (int i = 0; i < pointCount; ++i) {
                // 投影：点在 planeBasisVectors[0] 和 [1] 上的分量
                planarSourcePoints[i][0] =
                    sourcePoints[i][0] * planeBasisVectors[0][0] +
                    sourcePoints[i][1] * planeBasisVectors[0][1] +
                    sourcePoints[i][2] * planeBasisVectors[0][2];
                planarSourcePoints[i][1] =
                    sourcePoints[i][0] * planeBasisVectors[1][0] +
                    sourcePoints[i][1] * planeBasisVectors[1][1] +
                    sourcePoints[i][2] * planeBasisVectors[1][2];

                planarUnitVectors[i][0] =
                    unitVectors[i][0] * planeBasisVectors[0][0] +
                    unitVectors[i][1] * planeBasisVectors[0][1] +
                    unitVectors[i][2] * planeBasisVectors[0][2];
                planarUnitVectors[i][1] =
                    unitVectors[i][0] * planeBasisVectors[1][0] +
                    unitVectors[i][1] * planeBasisVectors[1][1] +
                    unitVectors[i][2] * planeBasisVectors[1][2];
            }
            for (int d = 0; d < 2; ++d) {
                planarDestinationPoint[d] =
                    destinationPoint[0] * planeBasisVectors[d][0] +
                    destinationPoint[1] * planeBasisVectors[d][1] +
                    destinationPoint[2] * planeBasisVectors[d][2];
            }

            // 构造 2D 的 RBF 系数矩阵 A 和右端项 rhs（A: [pointCount][pointCount], rhs: [pointCount][2]）
            double A[MAX_EDGE][MAX_EDGE] = {0};
            double rhs[MAX_EDGE][2] = {0};

            for (int j = 0; j < pointCount; ++j) {
                for (int i = j; i < pointCount; ++i) {
                    double rSquared = 0.0;
                    for (int d = 0; d < 2; ++d) {
                        double diff = planarSourcePoints[i][d] - planarSourcePoints[j][d];
                        rSquared += diff * diff;
                    }
                    rSquared /= (alpha * alpha);
                    double rbfValue = evaluate_rbf(rSquared);
                    double dotProduct = planarUnitVectors[i][0] * planarUnitVectors[j][0] +
                                        planarUnitVectors[i][1] * planarUnitVectors[j][1];
                    A[i][j] = rbfValue * dotProduct;
                    A[j][i] = A[i][j];  // 对称
                }
                double rSquaredDest = 0.0;
                for (int d = 0; d < 2; ++d) {
                    double diff = planarDestinationPoint[d] - planarSourcePoints[j][d];
                    rSquaredDest += diff * diff;
                }
                rSquaredDest /= (alpha * alpha);
                double rbfValueDest = evaluate_rbf(1.0);
                rhs[j][0] = rbfValueDest * planarUnitVectors[j][0];
                rhs[j][1] = rbfValueDest * planarUnitVectors[j][1];
            }

            // 分别求解两个线性系统：A * x1 = rhs(:,0) 和 A * x2 = rhs(:,1)
            double x1[MAX_EDGE] = {0};
            double x2[MAX_EDGE] = {0};
            double A_copy[MAX_EDGE][MAX_EDGE];

            // 复制 A 到 A_copy
            for (int i = 0; i < pointCount; ++i)
                for (int j = 0; j < pointCount; ++j)
                    A_copy[i][j] = A[i][j];

            {
                double b_col[MAX_EDGE];
                for (int i = 0; i < pointCount; ++i)
                    b_col[i] = rhs[i][0];
                gauss_elimination_fixed(A_copy, b_col, pointCount, x1);
            }

            // 重新复制 A 到 A_copy
            for (int i = 0; i < pointCount; ++i)
                for (int j = 0; j < pointCount; ++j)
                    A_copy[i][j] = A[i][j];

            {
                double b_col[MAX_EDGE];
                for (int i = 0; i < pointCount; ++i)
                    b_col[i] = rhs[i][1];
                gauss_elimination_fixed(A_copy, b_col, pointCount, x2);
            }

            // 将 2D 解转换为 3D 系数： coefficients[i] = planeBasisVectors[0] * x1[i] + planeBasisVectors[1] * x2[i]
            for (int i = 0; i < pointCount; ++i) {
                for (int d = 0; d < 3; ++d) {
                    coefficients[i][d] = planeBasisVectors[0][d] * x1[i] + planeBasisVectors[1][d] * x2[i];
                }
            }
        }
    };
}
