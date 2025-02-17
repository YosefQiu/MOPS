#pragma once
#include "ggl.h"
#include "cxxopts.hpp"


struct Command 
{
    enum class VisualizeType : int { kReMapping, kTrajectory, kCount };
    enum class SampleType : int { kUniform, kGaussian, kCount };

    // Default Parameters
    std::string input_yaml_path;
    std::string data_path_prefix = "output.nc";
    int image_width = 360;
    int image_height = 180;
    double longitude_min = -180.0, longitude_max = 180.0;
    double latitude_min = -90.0, latitude_max = 90.0;
    float fixed_layer = 10.0;
    float fixed_depth = 800.0;
    double delta_t = 120.0; // 120s
    double check_t = 60.0; // 1 min
    double trajectory_t = 86400.0; // 1 day
    std::array<double, 4> sample_range = {-180.0, 180.0, -90.0, 90.0}; // lon_min, lon_max, lat_min, lat_max
    int sample_number = 100;
    SampleType sample_type = SampleType::kUniform;
    VisualizeType visualize_type = VisualizeType::kReMapping;

    // Parsing command line arguments
    static Command parse(int argc, char* argv[]) 
    {
        cxxopts::Options options(argv[0], "MPAS-Ocean Particle SYCL(MPOS) Command Line Parser");

        Command cmd;

        options.add_options()
            ("i,input", "Input YAML file (Required)", cxxopts::value<std::string>(cmd.input_yaml_path))
            ("p,prefix", "Data path prefix", cxxopts::value<std::string>(cmd.data_path_prefix))
            ("imagesize", "Image Size (width height)", cxxopts::value<std::vector<int>>())
            ("longitude", "Longitude Range (min max)", cxxopts::value<std::vector<double>>())
            ("latitude", "Latitude Range (min max)", cxxopts::value<std::vector<double>>())
            ("layer", "fixed_layer", cxxopts::value<float>(cmd.fixed_layer))
            ("depth", "fixed_depth", cxxopts::value<float>(cmd.fixed_depth))
            ("deltat", "Delta T", cxxopts::value<double>(cmd.delta_t))
            ("checkt", "Check T", cxxopts::value<double>(cmd.check_t))
            ("trajectoryt", "Trajectory T", cxxopts::value<double>(cmd.trajectory_t))
            ("samplerange", "Sample Range (longitude_min longitude_max latitude_min latitude_max)", cxxopts::value<std::vector<double>>())
            ("samplenumber", "Sample Number", cxxopts::value<int>(cmd.sample_number))
            ("sampletype", "Sample Type (uniform or gaussian)", cxxopts::value<std::string>())
            ("visualizetype", "Visualize Type (remap or trajectory)", cxxopts::value<std::string>())
            ("h,help", "Print help message");

        auto results = options.parse(argc, argv);

        // Display help information
        if (results.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        // Parse input_yaml_path (required)
        if (!results.count("input")) {
            throw std::invalid_argument("[ERROR] No input file detected! Use -i <yaml_file>");
        }

        // Parse imagesize
        if (results.count("imagesize")) {
            auto image_size = results["imagesize"].as<std::vector<int>>();
            if (image_size.size() != 2) {
                throw std::invalid_argument("Image size should have exactly two values: width and height.");
            }
            cmd.image_width = image_size[0];
            cmd.image_height = image_size[1];
        }

        // Parse longitude range
        if (results.count("longitude")) {
            auto longitudes = results["longitude"].as<std::vector<double>>();
            if (longitudes.size() != 2) {
                throw std::invalid_argument("Longitude range should have exactly two values: min and max.");
            }
            cmd.longitude_min = longitudes[0];
            cmd.longitude_max = longitudes[1];
        }

        // Parse latitude range
        if (results.count("latitude")) {
            auto latitudes = results["latitude"].as<std::vector<double>>();
            if (latitudes.size() != 2) {
                throw std::invalid_argument("Latitude range should have exactly two values: min and max.");
            }
            cmd.latitude_min = latitudes[0];
            cmd.latitude_max = latitudes[1];
        }

        // Parse sample range
        if (results.count("samplerange")) {
            auto sample = results["samplerange"].as<std::vector<double>>();
            if (sample.size() != 4) {
                throw std::invalid_argument("Sample range should have exactly four values: longitude_min longitude_max latitude_min latitude_max.");
            }
            cmd.sample_range = {sample[0], sample[1], sample[2], sample[3]};
        }

        // Parse sample type
        if (results.count("sampletype")) {
            std::string sample_type_str = results["sampletype"].as<std::string>();
            if (sample_type_str == "uniform") {
                cmd.sample_type = Command::SampleType::kUniform;
            } else if (sample_type_str == "gaussian") {
                cmd.sample_type = Command::SampleType::kGaussian;
            } else {
                throw std::invalid_argument("Invalid sample type: must be 'uniform' or 'gaussian'.");
            }
        }

        // Parse visualize type
        if (results.count("visualizetype")) {
            std::string visualize_type_str = results["visualizetype"].as<std::string>();
            if (visualize_type_str == "remap") {
                cmd.visualize_type = Command::VisualizeType::kReMapping;
            } else if (visualize_type_str == "trajectory") {
                cmd.visualize_type = Command::VisualizeType::kTrajectory;
            } else {
                throw std::invalid_argument("Invalid visualize type: must be 'remap' or 'trajectory'.");
            }
        }

        return cmd;
    }

    // print command line arguments
    void print() const 
    {
        std::cout << "==========================================\n";
        std::cout << "== Input YAML Path: " << input_yaml_path << "\n";
        std::cout << "== Data Path Prefix: " << data_path_prefix << "\n";
        std::cout << "== Image Size: " << image_width << " x " << image_height << "\n";
        std::cout << "== Longitude Range: (" << longitude_min << ", " << longitude_max << ")\n";
        std::cout << "== Latitude Range: (" << latitude_min << ", " << latitude_max << ")\n";
        std::cout << "== Fixed Layer: " << fixed_layer << " layer\n";
        std::cout << "== Fixed Depth: " << fixed_depth << " m\n";
        std::cout << "== Delta T: " << delta_t << " s\n";
        std::cout << "== Check T: " << check_t << " s\n";
        std::cout << "== Trajectory T: " << trajectory_t << " s\n";
        std::cout << "== Sample Range (Longitude): (" << sample_range[0] << ", " << sample_range[1] << ")\n";
        std::cout << "== Sample Range (Latitude): (" << sample_range[2] << ", " << sample_range[3] << ")\n";
        std::cout << "== Sample Number: " << sample_number << "\n";
        std::cout << "== Sample Type: " << (sample_type == SampleType::kUniform ? "Uniform" : "Gaussian") << "\n";
        std::cout << "== Visualize Type: " << (visualize_type == VisualizeType::kReMapping ? "ReMapping" : "Trajectory") << "\n";
        std::cout << "==========================================\n";
    }
};
