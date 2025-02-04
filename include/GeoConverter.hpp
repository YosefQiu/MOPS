#pragma once
#include "ggl.h"

class GeoConverter
{
public:
	static void convertPixelToLatLonToRadians(int width, int height,
		double minLat, double maxLat,
		double minLon, double maxLon,
		vec2& pixel, SphericalCoord& latlon_radians)
	{
        /*
       *  convert pixel to lat lon based on the image size and the lat lon range
       *  unit: radians (lat and lon)
       *  width: image width
       *  height: image height
       *  minLat: minimum latitude
       *  maxLat: maximum latitude
       *  minLon: minimum longitude
       *  maxLon: maximum longitude
       *  pixel: (i, j) image i row, j column
       *  latlon_radians: (lat, lon) latitude and longitude
       */


        double lat = maxLat - (static_cast<double>(pixel.x()) / static_cast<double>(height) * (maxLat - minLat));
        double lon = (static_cast<double>(pixel.y()) / static_cast<double>(width) * (maxLon - minLon)) + minLon;
        latlon_radians.x() = lat; latlon_radians.y() = lon;
        latlon_radians.x() = latlon_radians.x() * (M_PI / 180.0);
        latlon_radians.y() = latlon_radians.y() * (M_PI / 180.0);
	}

    static void convertPixelToLatLonToDegrees(int width, int height,
        double minLat, double maxLat,
        double minLon, double maxLon,
        vec2& pixel, SphericalCoord& latlon_degrees)
    {
        /*
       *  convert pixel to lat lon based on the image size and the lat lon range
       *  unit: degrees (lat and lon)
       *  width: image width
       *  height: image height
       *  minLat: minimum latitude
       *  maxLat: maximum latitude
       *  minLon: minimum longitude
       *  maxLon: maximum longitude
       *  pixel: (i, j) image i row, j column
       *  latlon_radians: (lat, lon) latitude and longitude
       */


        double lat = maxLat - (static_cast<double>(pixel.x()) / static_cast<double>(height) * (maxLat - minLat));
        double lon = (static_cast<double>(pixel.y()) / static_cast<double>(width) * (maxLon - minLon)) + minLon;

        latlon_degrees.x() = lat; latlon_degrees.y() = lon;
    }

    static void convertDegreeLatLonToPixel(int width, int height,
        double minLat, double maxLat, double minLon, double maxLon,
        const SphericalCoord& latlon_degree, vec2& pixel)
    {
        /*
       *  convert lat lon to pixel based on the image size and the lat lon range
       *  unit: degree (lat and lon)
       *  width: image width
       *  height: image height
       *  minLat: minimum latitude
       *  maxLat: maximum latitude
       *  minLon: minimum longitude
       *  maxLon: maximum longitude
       *  pixel: (i, j) image i row, j column
       *  latlon_degree: (lat, lon) latitude and longitude
       */


        pixel.x() = (maxLat - latlon_degree.x()) / (maxLat - minLat) * static_cast<double>(height);
        pixel.y() = (latlon_degree.y() - minLon) / (maxLon - minLon) * static_cast<double>(width);
    }

    static void convertRadianLatLonToPixel(int width, int height,
        double minLat, double maxLat, double minLon, double maxLon,
        const SphericalCoord& latlon_radian, vec2& pixel)
    {
        /*
        *  convert lat lon to pixel based on the image size and the lat lon range
        *  unit: radians (lat and lon)
        *  width: image width
        *  height: image height
        *  minLat: minimum latitude
        *  maxLat: maximum latitude
        *  minLon: minimum longitude
        *  maxLon: maximum longitude
        *  pixel: (i, j) image i row, j column
        *  latlon_radian: (lat, lon) latitude and longitude
        */

        double lat_degree = latlon_radian.x() * (180.0 / M_PI);
        double lon_degree = latlon_radian.y() * (180.0 / M_PI);

        pixel.x() = (maxLat - lat_degree) / (maxLat - minLat) * static_cast<double>(height);
        pixel.y() = (lon_degree - minLon) / (maxLon - minLon) * static_cast<double>(width);
    }


    static void convertRadianLatLonToXYZ(SphericalCoord& thetaPhi, CartesianCoord& position, double r = 6371010.0f)
    {
        /*
        *  convert lat lon to xyz based on the earth radius
        *  unit: radians (lat and lon)
        *  thetaPhi: (theta, phi) latitude and longitude
        *  Considering the latitude and longitude,
        *  it is a little different from the conventional
        *  spherical coordinate conversion.
        */

        double theta = thetaPhi.x();
        double phi = thetaPhi.y();
        double costheta = sycl::cos(theta); double cosphi = sycl::cos(phi);
        double sintheta = sycl::sin(theta); double sinphi = sycl::sin(phi);
        position.x() = r * costheta * cosphi;
        position.y() = r * costheta * sinphi;
        position.z() = r * sintheta;
    }
    
    static void convertXYZToLatLonRadian(CartesianCoord& position, SphericalCoord& thetaPhi_radian)
    {

        /*
        *  convert xyz to lat lon based on the earth radius
        *  unit: radians (lat and lon)
        *  position: (x, y, z) position
        *  thetaPhi: (theta, phi) latitude and longitude\
        *  Considering the latitude and longitude,
        *  it is a little different from the conventional
        *  spherical coordinate conversion.
        */

        double x = position.x();
        double y = position.y();
        double z = position.z();
        double r = sycl::sqrt(x * x + y * y + z * z);

        double theta = sycl::asin(z / r);
        double phi = sycl::atan2(y, x);

        thetaPhi_radian.x() = theta;
        thetaPhi_radian.y() = phi;
    }

    static void convertXYZToLatLonDegree(CartesianCoord& position, SphericalCoord& thetaPhi_degree)
    {
        /*
        *  convert xyz to lat lon based on the earth radius
        *  unit: degrees (lat and lon)
        *  position: (x, y, z) position
        *  thetaPhi: (theta, phi) latitude and longitude
        *  Considering the latitude and longitude,
        *  it is a little different from the conventional
        *  spherical coordinate conversion.
        */

        double x = position.x();
        double y = position.y();
        double z = position.z();
        double r = sycl::sqrt(x * x + y * y + z * z);

        double theta = sycl::asin(z / r);
        double phi = sycl::atan2(y, x);

        thetaPhi_degree.x() = theta * (180.0 / M_PI);
        thetaPhi_degree.y() = phi * (180.0 / M_PI);
    }

    static void convertDegreeToRadian(const SphericalCoord& degree, SphericalCoord& radian)
    {
        /*
        *  Convert latitude and longitude from degrees to radians
        *  degree: (lat, lon) latitude and longitude in degrees
        *  radian: (lat, lon) latitude and longitude in radians
        */

        radian.x() = degree.x() * (M_PI / 180.0);
        radian.y() = degree.y() * (M_PI / 180.0);
    }

    static void convertRadianToDegree(const SphericalCoord& radian, SphericalCoord& degree)
    {
        /*
        *  Convert latitude and longitude from radians to degrees
        *  radian: (lat, lon) latitude and longitude in radians
        *  degree: (lat, lon) latitude and longitude in degrees
        */

        degree.x() = radian.x() * (180.0 / M_PI);
        degree.y() = radian.y() * (180.0 / M_PI);
    }

    static void convertXYZVelocityToENU(const CartesianCoord& xyzPoint, const vec3& xyzVel, double& Uzon, double& Umer)
    {
        double Rxy, Rxyz, slon, clon, slat, clat;

        // Test for singularities at the poles
        if (xyzPoint[0] == 0.0 && xyzPoint[1] == 0.0) 
        {
            Uzon = 0.0;
            Umer = 0.0;
            return;
        }

        // Compute geometric coordinate transform coefficients
        Rxy = sycl::sqrt(xyzPoint[0] * xyzPoint[0] + xyzPoint[1] * xyzPoint[1]);
        Rxyz = sycl::sqrt(xyzPoint[0] * xyzPoint[0] + xyzPoint[1] * xyzPoint[1] + xyzPoint[2] * xyzPoint[2]);
        slon = xyzPoint[1] / Rxy;
        clon = xyzPoint[0] / Rxy;
        slat = xyzPoint[2] / Rxyz;
        clat = Rxy / Rxyz;

        // Compute the zonal and meridional velocity fields
        Uzon = -slon * xyzVel[0] + clon * xyzVel[1];
        Umer = -slat * (clon * xyzVel[0] + slon * xyzVel[1]) + clat * xyzVel[2];
    }

    static void convertENUVelocityToXYZ(const CartesianCoord& xyzPoint, const double& Uzon, const double& Umer, const double& Uup, vec3& xyzVel) 
    {
        double Rxy, Rxyz, slon, clon, slat, clat;

        // Test for singularities at the poles
        if (xyzPoint[0] == 0.0 && xyzPoint[1] == 0.0) 
        {
            xyzVel[0] = 0.0;
            xyzVel[1] = 0.0;
            xyzVel[2] = Uup;  // Only the vertical component remains
            return;
        }

        // Compute geometric coordinate transform coefficients
        Rxy = sycl::sqrt(xyzPoint[0] * xyzPoint[0] + xyzPoint[1] * xyzPoint[1]);
        Rxyz = sycl::sqrt(xyzPoint[0] * xyzPoint[0] + xyzPoint[1] * xyzPoint[1] + xyzPoint[2] * xyzPoint[2]);
        slon = xyzPoint[1] / Rxy;
        clon = xyzPoint[0] / Rxy;
        slat = xyzPoint[2] / Rxyz;
        clat = Rxy / Rxyz;

        // Convert ENU to XYZ
        xyzVel[0] = -slon * Uzon - slat * clon * Umer + clon * clat * Uup;
        xyzVel[1] = clon * Uzon - slat * slon * Umer + slon * clat * Uup;
        xyzVel[2] = clat * Umer + slat * Uup;
    }

    static void convertXYZPositionToENUUnitVectory(const CartesianCoord& xyzPoint, CartesianCoord& Uzon, CartesianCoord& Umer)
    {
        double Rxy, Rxyz, slon, clon, slat, clat;

        // Test for singularities at the poles
        if (xyzPoint[0] == 0.0 && xyzPoint[1] == 0.0) 
        {
            Uzon = 0.0;
            Umer = 0.0;
            return;
        }

        // Compute geometric coordinate transform coefficients
        Rxy = sycl::sqrt(xyzPoint[0] * xyzPoint[0] + xyzPoint[1] * xyzPoint[1]);
        Rxyz = sycl::sqrt(xyzPoint[0] * xyzPoint[0] + xyzPoint[1] * xyzPoint[1] + xyzPoint[2] * xyzPoint[2]);
        slon = xyzPoint[1] / Rxy;
        clon = xyzPoint[0] / Rxy;
        slat = xyzPoint[2] / Rxyz;
        clat = Rxy / Rxyz;

        // Compute the zonal and meridional velocity fields
        Uzon = { -slon, clon, 0.0 };
        Umer = {-slat * clon, -slat * slon, clat};
    }

    static vec3 computeRotationAxis(const CartesianCoord& xyzPoint, const vec3& xyzVel)
    {
        vec3 axis;
        axis.x() = xyzPoint.y() * xyzVel.z() - xyzPoint.z() * xyzVel.y();
        axis.y() = xyzPoint.z() * xyzVel.x() - xyzPoint.x() * xyzVel.z();
        axis.z() = xyzPoint.x() * xyzVel.y() - xyzPoint.y() * xyzVel.x();
        return axis;
    }

    static CartesianCoord rotateAroundAxis(const CartesianCoord& point, const vec3& axis, double theta)
    {
        double PI = 3.14159265358979323846;
        double thetaRad = theta * PI / 180.0;
        double cosTheta = sycl::cos(thetaRad);
        double sinTheta = sycl::sin(thetaRad);

        auto len = sycl::sqrt(axis.x() * axis.x() + axis.y() * axis.y() + axis.z() * axis.z());
        vec3 normalized = { axis.x() / len, axis.y() / len, axis.z() / len };

        vec3 u = normalized;

        vec3 rotated;
        rotated.x() =   (cosTheta + u.x() * u.x() * (1.0 - cosTheta)) * point.x() +
                        (u.x() * u.y() * (1.0 - cosTheta) - u.z() * sinTheta) * point.y() +
                        (u.x() * u.z() * (1.0 - cosTheta) + u.y() * sinTheta) * point.z();

        rotated.y() =   (u.y() * u.x() * (1.0 - cosTheta) + u.z() * sinTheta) * point.x() +
                        (cosTheta + u.y() * u.y() * (1 - cosTheta)) * point.y() +
                        (u.y() * u.z() * (1.0 - cosTheta) - u.x() * sinTheta) * point.z();

        rotated.z() =   (u.z() * u.x() * (1.0 - cosTheta) - u.y() * sinTheta) * point.x() +
                        (u.z() * u.y() * (1.0 - cosTheta) + u.x() * sinTheta) * point.y() +
                        (cosTheta + u.z() * u.z() * (1.0 - cosTheta)) * point.z();

        return rotated;
    }

    static void COVERTLATLONTOXYZ(vec2& thetaPhi, vec3& position)
    {
        auto theta = thetaPhi.x();
        auto phi = thetaPhi.y();
        auto r = 6371.01 * 1000.0;
        position.x() = r * sycl::cos(theta) * sycl::cos(phi);
        position.y() = r * sycl::cos(theta) * sycl::sin(phi);
        position.z() = r * sycl::sin(theta);
    }
    static void CONVERTXYZTOLATLON(vec3& position, vec2& thetaPhi)
    {
        double x = position.x(); double y = position.y(); double z = position.z();
        double r = sycl::sqrt(x * x + y * y + z * z);
        double theta = sycl::asin(z / r);
        double phi = sycl::atan2(y, x);
        thetaPhi.x() = theta;
        thetaPhi.y() = phi;
    }
};