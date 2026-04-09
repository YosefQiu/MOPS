#include "Core/MPASOVisualizer.h"
#include "MPASOVisualizer.h"
#include "Common/MOPSFactory.h"
#include "Common/TrajectoryCommon.h"
#include <bits/types/locale_t.h>
#include <cstdlib>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <vector>
#if MOPS_VTK
#include "IO/VTKFileManager.hpp"
#endif


using namespace MOPS;

void MPASOVisualizer::VisualizeFixedLayer(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, const RuntimeContext& ctx)
{
    MOPS::Factory::VisualizeFixedLayer(mpasoF, config, img, ctx);
}

void MPASOVisualizer::VisualizeFixedDepth(MPASOField* mpasoF, VisualizationSettings* config, std::vector<ImageBuffer<double>>& img_vec, const RuntimeContext& ctx)
{
    MOPS::Factory::VisualizeFixedDepth(mpasoF, config, img_vec, ctx);
}

void MPASOVisualizer::VisualizeFixedDepth(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, const RuntimeContext& ctx)
{
    if (img == nullptr) {
        return;
    }

    std::vector<ImageBuffer<double>> img_vec;
    img_vec.push_back(*img);
    MOPS::Factory::VisualizeFixedDepth(mpasoF, config, img_vec, ctx);
    *img = std::move(img_vec[0]);
}

void MPASOVisualizer::VisualizeFixedLatitude(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, const RuntimeContext& ctx)
{
    MOPS::Factory::VisualizeFixedLatitude(mpasoF, config, img, ctx);
}

std::vector<TrajectoryLine> MPASOVisualizer::StreamLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, const RuntimeContext& ctx)
{
    return MOPS::Factory::StreamLine(mpasoF, points, config, default_cell_id, ctx);
}

std::vector<TrajectoryLine> MPASOVisualizer::PathLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, const RuntimeContext& ctx)
{
    return MOPS::Factory::PathLine(mpasoF, points, config, default_cell_id, ctx);
}

void MPASOVisualizer::VisualizeFixedLayer(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, const GPUContext& ctx)
{
    VisualizeFixedLayer(mpasoF, config, img, RuntimeContext::FromGPU(ctx));
}

void MPASOVisualizer::VisualizeFixedDepth(MPASOField* mpasoF, VisualizationSettings* config, std::vector<ImageBuffer<double>>& img_vec, const GPUContext& ctx)
{
    VisualizeFixedDepth(mpasoF, config, img_vec, RuntimeContext::FromGPU(ctx));
}

void MPASOVisualizer::VisualizeFixedDepth(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, const GPUContext& ctx)
{
    VisualizeFixedDepth(mpasoF, config, img, RuntimeContext::FromGPU(ctx));
}

void MPASOVisualizer::VisualizeFixedLatitude(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, const GPUContext& ctx)
{
    VisualizeFixedLatitude(mpasoF, config, img, RuntimeContext::FromGPU(ctx));
}

std::vector<TrajectoryLine> MPASOVisualizer::StreamLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, const GPUContext& ctx)
{
    return StreamLine(mpasoF, points, config, default_cell_id, RuntimeContext::FromGPU(ctx));
}

std::vector<TrajectoryLine> MPASOVisualizer::PathLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, const GPUContext& ctx)
{
    return PathLine(mpasoF, points, config, default_cell_id, RuntimeContext::FromGPU(ctx));
}

void MPASOVisualizer::VisualizeFixedLayer(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, sycl::queue& sycl_Q)
{
    MOPS::GPUContext ctx = MOPS::GPUContext::FromSYCL(sycl_Q);
    VisualizeFixedLayer(mpasoF, config, img, RuntimeContext::FromGPU(ctx));
}


void MPASOVisualizer::VisualizeFixedDepth(MPASOField* mpasoF, VisualizationSettings* config, std::vector<ImageBuffer<double>>& img_vec, sycl::queue& sycl_Q)
{
    MOPS::GPUContext ctx = MOPS::GPUContext::FromSYCL(sycl_Q);
    VisualizeFixedDepth(mpasoF, config, img_vec, RuntimeContext::FromGPU(ctx));

}

void MPASOVisualizer::VisualizeFixedDepth(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, sycl::queue& sycl_Q)
{
    if (img == nullptr) {
        return;
    }

    std::vector<ImageBuffer<double>> img_vec;
    img_vec.push_back(*img);
    VisualizeFixedDepth(mpasoF, config, img_vec, sycl_Q);
    *img = std::move(img_vec[0]);
}




void MPASOVisualizer::VisualizeFixedLatitude(MPASOField* mpasoF, VisualizationSettings* config, ImageBuffer<double>* img, sycl::queue& sycl_Q)
{
    MOPS::GPUContext ctx = MOPS::GPUContext::FromSYCL(sycl_Q);
    VisualizeFixedLatitude(mpasoF, config, img, RuntimeContext::FromGPU(ctx));
}


void MPASOVisualizer::GenerateSamplePoint(std::vector<CartesianCoord>& points, SamplingSettings* config)
{
    auto minLat = config->getLatitudeRange().x(); auto maxLat = config->getLatitudeRange().y();
    auto minLon = config->getLongitudeRange().x(); auto maxLon = config->getLongitudeRange().y();

    double i_step = (maxLat - minLat) / static_cast<double>(config->getSampleRange().x() - 1);
    double j_step = (maxLon - minLon) / static_cast<double>(config->getSampleRange().y() - 1);

    for (double i = minLat ; i < maxLat; i += i_step)
    {
        for (double j = minLon; j < maxLon; j += j_step)
        {
            CartesianCoord p = { j, i, config->getDepth() };
            points.push_back(p);
        }
    }


    Debug("Generate %d sample points in [ %f, %f ] -> [ %f, %f ]", points.size(), minLat, minLon, maxLat, maxLon);

    for (auto i = 0; i < points.size(); i++)
    {
        vec3 get_points = points[i];
        SphericalCoord latlon_d; SphericalCoord latlon_r; CartesianCoord position;
        latlon_d.x() = get_points.y(); latlon_d.y() = get_points.x();
        GeoConverter::convertDegreeToRadian(latlon_d, latlon_r);
        GeoConverter::convertRadianLatLonToXYZ(latlon_r, position);
        points[i].x() = position.x(); points[i].y() = position.y(); points[i].z() = position.z();
    }
}

void MPASOVisualizer::GenerateSamplePointAtCenter(std::vector<CartesianCoord>& points, SamplingSettings* config)
{
    if (config->isAtCellCenter() == false) return;
    Debug("Generate %d sample points At Cell Center]", points.size());

}


//TODO: Temporarily unavailable, it will be released later.
[[deprecated]]
void MPASOVisualizer::GenerateGaussianSpherePoints(std::vector<CartesianCoord>& points, SamplingSettings* config, int numPoints, double meanLat, double meanLon, double stdDev)
{
    auto minLat = config->getLatitudeRange().x(); auto maxLat = config->getLatitudeRange().y();
    auto minLon = config->getLongitudeRange().x(); auto maxLon = config->getLongitudeRange().y();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> latDist(meanLat, stdDev); // Latitude Gaussian distribution
    std::normal_distribution<double> lonDist(meanLon, stdDev); // Longitude Gaussian distribution

    for (int i = 0; i < numPoints; ++i) {
        double lat, lon;

        // Generate latitudes and longitudes within the range
        do {
            lat = latDist(gen);
        } while (lat < minLat || lat > maxLat);

        do {
            lon = lonDist(gen);
        } while (lon < minLon || lon > maxLon);

        // Convert to Cartesian coordinates
        SphericalCoord latlon_d = { lat, lon };
        SphericalCoord latlon_r;
        CartesianCoord position;
        GeoConverter::convertDegreeToRadian(latlon_d, latlon_r);
        GeoConverter::convertRadianLatLonToXYZ(latlon_r, position);

        points.push_back(position);
    }
    Debug("Generate %d sample points in [ %f, %f ] -> [ %f ]", points.size(), meanLat, meanLon, stdDev);
}


[[deprecated]]
vec3 computeRotationAxis(const vec3& position, const vec3& velocity)
{
    vec3 axis;
    axis.x() = position.y() * velocity.z() - position.z() * velocity.y();
    axis.y() = position.z() * velocity.x() - position.x() * velocity.z();
    axis.z() = position.x() * velocity.y() - position.y() * velocity.x();
    return axis;
}
[[deprecated]]
double magnitude(const vec3& v)
{
    return MOPS::math::sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
}
[[deprecated]]
vec3 normalize(const vec3& v)
{
    double length = sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
    vec3 normalized = { v.x() / length, v.y() / length, v.z() / length };
    return normalized;
}
[[deprecated]]
void rotateAroundAxis(const vec3& point, const vec3& axis, double theta_rad, double& x, double& y, double& z)
{
    // theta_rad is in radians
    double thetaRad = theta_rad;
    double cosTheta = MOPS::math::cos(thetaRad);
    double sinTheta = MOPS::math::sin(thetaRad);
    vec3 u = axis;
    MOPS_VEC3_NORMALIZE(u);

    vec3 rotated;
    rotated.x() = (cosTheta + u.x() * u.x() * (1.0 - cosTheta)) * point.x() +
        (u.x() * u.y() * (1.0 - cosTheta) - u.z() * sinTheta) * point.y() +
        (u.x() * u.z() * (1.0 - cosTheta) + u.y() * sinTheta) * point.z();

    rotated.y() = (u.y() * u.x() * (1.0 - cosTheta) + u.z() * sinTheta) * point.x() +
        (cosTheta + u.y() * u.y() * (1.0 - cosTheta)) * point.y() +
        (u.y() * u.z() * (1.0 - cosTheta) - u.x() * sinTheta) * point.z();

    rotated.z() = (u.z() * u.x() * (1.0 - cosTheta) - u.y() * sinTheta) * point.x() +
        (u.z() * u.y() * (1.0 - cosTheta) + u.x() * sinTheta) * point.y() +
        (cosTheta + u.z() * u.z() * (1.0 - cosTheta)) * point.z();

    x = rotated.x();
    y = rotated.y();
    z = rotated.z();
}


[[deprecated]]
bool isClose(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) < eps;
}
[[deprecated]]
bool isCloseVec3(const CartesianCoord& a, const CartesianCoord& b, double eps = 1e-6) {
    return std::fabs(a.x() - b.x()) < eps &&
           std::fabs(a.y() - b.y()) < eps &&
           std::fabs(a.z() - b.z()) < eps;
}
[[deprecated]]
bool compareTrajectoryLines(const std::vector<TrajectoryLine>& a, const std::vector<TrajectoryLine>& b) {
    if (a.size() != b.size()) {
        std::cout << "[Compare] Line vector size mismatch: " << a.size() << " vs " << b.size() << std::endl;
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        const auto& la = a[i];
        const auto& lb = b[i];

        if (la.lineID != lb.lineID) {
            std::cout << "[Compare] LineID mismatch at index " << i << ": " << la.lineID << " vs " << lb.lineID << std::endl;
            return false;
        }

        if (la.points.size() != lb.points.size()) {
            std::cout << "[Compare] Points size mismatch at line " << i << ": " << la.points.size() << " vs " << lb.points.size() << std::endl;
            return false;
        }

        for (size_t j = 0; j < la.points.size(); ++j) {
            if (!isCloseVec3(la.points[j], lb.points[j])) {
                std::cout << "[Compare] Point mismatch at line " << i << ", point " << j << std::endl;
                return false;
            }
        }

        if (!isCloseVec3(la.lastPoint, lb.lastPoint)) {
            std::cout << "[Compare] LastPoint mismatch at line " << i << std::endl;
            return false;
        }

        if (!isClose(la.duration, lb.duration)) {
            std::cout << "[Compare] Duration mismatch at line " << i << ": " << la.duration << " vs " << lb.duration << std::endl;
            return false;
        }

        if (!isClose(la.timestamp, lb.timestamp)) {
            std::cout << "[Compare] Timestamp mismatch at line " << i << ": " << la.timestamp << " vs " << lb.timestamp << std::endl;
            return false;
        }
    }

    std::cout << "[Compare] All trajectory lines are identical!" << std::endl;
    return true;
}

std::vector<TrajectoryLine> MPASOVisualizer::removeNaNTrajectoriesAndReindex(std::vector<TrajectoryLine>& trajectory_lines)
{
    return MOPS::Common::RemoveNaNTrajectoriesAndReindex(trajectory_lines);
}

std::vector<TrajectoryLine> MPASOVisualizer::StreamLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, sycl::queue& sycl_Q)
{
    MOPS::GPUContext ctx = MOPS::GPUContext::FromSYCL(sycl_Q);
    return StreamLine(mpasoF, points, config, default_cell_id, RuntimeContext::FromGPU(ctx));
}

std::vector<TrajectoryLine> MPASOVisualizer::PathLine(MPASOField* mpasoF, std::vector<CartesianCoord>& points, TrajectorySettings* config, std::vector<int>& default_cell_id, sycl::queue& sycl_Q)
{
    MOPS::GPUContext ctx = MOPS::GPUContext::FromSYCL(sycl_Q);
    return PathLine(mpasoF, points, config, default_cell_id, RuntimeContext::FromGPU(ctx));
}
