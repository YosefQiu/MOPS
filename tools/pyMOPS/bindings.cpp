#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>

#include "api/MOPS.h"
#include "IO/MPASOReader.h"
#include "Core/MPASOGrid.h"
#include "Core/MPASOSolution.h"
#include "Core/MOPSApp.h"
#include "SYCL/ImageBuffer.hpp"
#include "Utils/GeoConverter.hpp" 


namespace py = pybind11;


PYBIND11_MODULE(pyMOPS, m) {
    m.doc() = "pyMOPS";


    py::enum_<MOPS::AttributeFormat>(m, "AttributeFormat")
        .value("kDouble", MOPS::AttributeFormat::kDouble)
        .value("kFloat", MOPS::AttributeFormat::kFloat)
        .value("kChar", MOPS::AttributeFormat::kChar)
        .value("kVec3", MOPS::AttributeFormat::kVec3);

    py::enum_<MOPS::CalcPositionType>(m, "CalcPositionType")
        .value("kCenter", MOPS::CalcPositionType::kCenter)
        .value("kVertx", MOPS::CalcPositionType::kVertx)
        .value("kPoint", MOPS::CalcPositionType::kPoint);

    py::enum_<MOPS::CalcAttributeType>(m, "CalcAttributeType")
        .value("kZonalMerimoal", MOPS::CalcAttributeType::kZonalMerimoal)
        .value("kVelocity", MOPS::CalcAttributeType::kVelocity)
        .value("kZTop", MOPS::CalcAttributeType::kZTop)
        .value("kTemperature", MOPS::CalcAttributeType::kTemperature)
        .value("kSalinity", MOPS::CalcAttributeType::kSalinity)
        .value("kAll", MOPS::CalcAttributeType::kAll);

    py::enum_<MOPS::VisualizeType>(m, "VisualizeType")
        .value("kFixedLayer", MOPS::VisualizeType::kFixedLayer)
        .value("kFixedDepth", MOPS::VisualizeType::kFixedDepth);

    py::enum_<MOPS::CalcDirection>(m, "CalcDirection")
        .value("kForward", MOPS::CalcDirection::kForward)
        .value("kBackward", MOPS::CalcDirection::kBackward);

    py::enum_<MOPS::CalcMethodType>(m, "CalcMethodType")
        .value("kRK4", MOPS::CalcMethodType::kRK4)
        .value("kEuler", MOPS::CalcMethodType::kEuler);

    py::enum_<MOPS::SaveType>(m, "SaveType")
        .value("kVTI", MOPS::SaveType::kVTI)
        .value("kNone", MOPS::SaveType::kNone);
    
    py::enum_<MOPS::GridAttributeType>(m, "GridAttributeType")
        .value("kCellSize", MOPS::GridAttributeType::kCellSize)
        .value("kEdgeSize", MOPS::GridAttributeType::kEdgeSize)
        .value("kVertexSize", MOPS::GridAttributeType::kVertexSize)
        .value("kMaxEdgesSize", MOPS::GridAttributeType::kMaxEdgesSize)
        .value("kVertLevels", MOPS::GridAttributeType::kVertLevels)
        .value("kVertLevelsP1", MOPS::GridAttributeType::kVertLevelsP1)
        .value("kVertexCoord", MOPS::GridAttributeType::kVertexCoord)
        .value("kCellCoord", MOPS::GridAttributeType::kCellCoord)
        .value("kEdgeCoord", MOPS::GridAttributeType::kEdgeCoord)
        .value("kVertexLatLon", MOPS::GridAttributeType::kVertexLatLon)
        .value("kVerticesOnCell", MOPS::GridAttributeType::kVerticesOnCell)
        .value("kVerticesOnEdge", MOPS::GridAttributeType::kVerticesOnEdge)
        .value("kCellsOnVertex", MOPS::GridAttributeType::kCellsOnVertex)
        .value("kCellsOnCell", MOPS::GridAttributeType::kCellsOnCell)
        .value("kNumberVertexOnCell", MOPS::GridAttributeType::kNumberVertexOnCell)
        .value("kCellsOnEdge", MOPS::GridAttributeType::kCellsOnEdge)
        .value("kEdgesOnCell", MOPS::GridAttributeType::kEdgesOnCell)
        .value("kCellWeight", MOPS::GridAttributeType::kCellWeight);

    py::enum_<MOPS::AttributeType>(m, "AttributeType")
        .value("kZonalVelocity", MOPS::AttributeType::kZonalVelocity)
        .value("kMeridionalVelocity", MOPS::AttributeType::kMeridionalVelocity)
        .value("kVelocity", MOPS::AttributeType::kVelocity)
        .value("kNormalVelocity", MOPS::AttributeType::kNormalVelocity)
        .value("kZTop", MOPS::AttributeType::kZTop)
        .value("kLayerThickness", MOPS::AttributeType::kLayerThickness)
        .value("kBottomDepth", MOPS::AttributeType::kBottomDepth);


    py::class_<MOPS::MPASOReader, std::shared_ptr<MOPS::MPASOReader>>(m, "MPASOReader")
        .def(py::init<>())
        .def_static("readGridData", &MOPS::MPASOReader::readGridData, py::arg("yaml_path"))
        .def_static("readSolData", &MOPS::MPASOReader::readSolData, py::arg("yaml_path"), py::arg("time_str"), py::arg("time_index") = 0);

    py::class_<MOPS::MPASOGrid, std::shared_ptr<MOPS::MPASOGrid>>(m, "MPASOGrid")
        .def(py::init<>())
        .def("init_from_reader", [](MOPS::MPASOGrid& self, const std::shared_ptr<MOPS::MPASOReader>& reader) {
            self.initGrid(reader.get());
        })
        .def("init_from_yaml", &MOPS::MPASOGrid::initGrid_DemoLoading)
        .def("setGridAttribute", &MOPS::MPASOGrid::setGridAttribute)
        .def("setGridAttributesVec3", [](MOPS::MPASOGrid& self, MOPS::GridAttributeType type, py::array_t<double> arr) {
            if (arr.ndim() != 2 || arr.shape(1) != 3)
                throw std::runtime_error("Input array must have shape (N, 3)");
            std::vector<vec3> vec;
            auto buf = arr.unchecked<2>();
            for (ssize_t i = 0; i < buf.shape(0); ++i) {
                vec.emplace_back(buf(i, 0), buf(i, 1), buf(i, 2));
            }
            self.setGridAttributesVec3(type, vec);
        })
        .def("setGridAttributesVec2", [](MOPS::MPASOGrid& self, MOPS::GridAttributeType type, py::array_t<double> arr) {
            if (arr.ndim() != 2 || arr.shape(1) != 2)
                throw std::runtime_error("Input array must have shape (N, 2)");
            std::vector<vec2> vec;
            auto buf = arr.unchecked<2>();
            for (ssize_t i = 0; i < buf.shape(0); ++i) {
                vec.emplace_back(buf(i, 0), buf(i, 1));
            }
            self.setGridAttributesVec2(type, vec);
        })
        .def("setGridAttributesInt", [](MOPS::MPASOGrid& self, MOPS::GridAttributeType type, py::array_t<size_t> arr) {
            if (arr.ndim() != 1)
                throw std::runtime_error("Input array must be 1D");
            std::vector<size_t> vec;
            auto buf = arr.unchecked<1>();
            for (ssize_t i = 0; i < buf.shape(0); ++i) {
                vec.push_back(buf(i));
            }
            self.setGridAttributesInt(type, vec);
        })
        .def("setGridAttributesFloat", [](MOPS::MPASOGrid& self, MOPS::GridAttributeType type, py::array_t<float> arr) {
            if (arr.ndim() != 1)
                throw std::runtime_error("Input array must be 1D");
            std::vector<float> vec;
            auto buf = arr.unchecked<1>();
            for (ssize_t i = 0; i < buf.shape(0); ++i) {
                vec.push_back(buf(i));
            }
            self.setGridAttributesFloat(type, vec);
        });
    
    py::class_<MOPS::MPASOSolution, std::shared_ptr<MOPS::MPASOSolution>>(m, "MPASOSolution")
        .def(py::init<>())
        .def("init_from_reader", [](MOPS::MPASOSolution& self, const std::shared_ptr<MOPS::MPASOReader>& reader) {
            self.initSolution(reader.get());
        })
        .def("init_from_yaml", &MOPS::MPASOSolution::initSolution_DemoLoading)
        .def("add_attribute", &MOPS::MPASOSolution::addAttribute)
        .def("setTimestep", &MOPS::MPASOSolution::setTimestep)
        .def("setAttribute", &MOPS::MPASOSolution::setAttribute)
        .def("setAttributesVec3", [](MOPS::MPASOSolution& self, MOPS::AttributeType type, py::array_t<double> arr) {
            if (arr.ndim() != 2 || arr.shape(1) != 3)
                throw std::runtime_error("Input must be (N, 3) numpy array");
            std::vector<vec3> vec;
            auto buf = arr.unchecked<2>();
            for (ssize_t i = 0; i < buf.shape(0); ++i) {
                vec.emplace_back(buf(i, 0), buf(i, 1), buf(i, 2));
            }
            self.setAttributesVec3(type, vec);
        })
        .def("setAttributesDouble", [](MOPS::MPASOSolution& self, MOPS::AttributeType type, py::array_t<double> arr) {
            if (arr.ndim() != 1)
                throw std::runtime_error("Input must be (N,) numpy array");
            std::vector<double> vec;
            auto buf = arr.unchecked<1>();
            for (ssize_t i = 0; i < buf.shape(0); ++i) {
                vec.push_back(buf(i));
            }
            self.setAttributesDouble(type, vec);
        })
        .def("getID", (int (MOPS::MPASOSolution::*)() const)&MOPS::MPASOSolution::getID)
        .def("getTimeStamp", &MOPS::MPASOSolution::getTimeStamp);
        

    py::class_<MOPS::VisualizationSettings>(m, "VisualizationSettings")
        .def(py::init<>())
        .def_property("imageSize",
            [](const MOPS::VisualizationSettings& self) {
                return py::make_tuple(self.imageSize.x(), self.imageSize.y());
            },
            [](MOPS::VisualizationSettings& self, py::tuple t) {
                if (t.size() != 2) throw std::runtime_error("imageSize must be a tuple of size 2");
                self.imageSize = sycl::double2(t[0].cast<double>(), t[1].cast<double>());
            }
        )
        .def_property("LatRange",
            [](const MOPS::VisualizationSettings& self) {
                return py::make_tuple(self.LatRange.x(), self.LatRange.y());
            },
            [](MOPS::VisualizationSettings& self, py::tuple t) {
                if (t.size() != 2) throw std::runtime_error("LatRange must be a tuple of size 2");
                self.LatRange = sycl::double2(t[0].cast<double>(), t[1].cast<double>());
            }
        )
        .def_property("LonRange",
            [](const MOPS::VisualizationSettings& self) {
                return py::make_tuple(self.LonRange.x(), self.LonRange.y());
            },
            [](MOPS::VisualizationSettings& self, py::tuple t) {
                if (t.size() != 2) throw std::runtime_error("LonRange must be a tuple of size 2");
                self.LonRange = sycl::double2(t[0].cast<double>(), t[1].cast<double>());
            }
        )
        .def_readwrite("FixedDepth", &MOPS::VisualizationSettings::FixedDepth)
        .def_readwrite("TimeStep", &MOPS::VisualizationSettings::TimeStep)
        .def_readwrite("CalcType", &MOPS::VisualizationSettings::CalcType)
        .def_readwrite("VisType", &MOPS::VisualizationSettings::VisType)
        .def_readwrite("PositionType", &MOPS::VisualizationSettings::PositionType)
        .def_readwrite("SaveType", &MOPS::VisualizationSettings::SaveType);


    

   
   
    py::class_<MOPS::SamplingSettings>(m, "SeedsSettings")
        .def(py::init<>())
        .def("setSeedsRange", [](MOPS::SamplingSettings& self, py::tuple t) {
            if (t.size() != 2) throw std::runtime_error("sampleRange must be a tuple of size 2");
            self.setSampleRange(sycl::int2(t[0].cast<int>(), t[1].cast<int>()));
        })
        .def("setGeoBox", [](MOPS::SamplingSettings& self, py::tuple lat, py::tuple lon) {
            if (lat.size() != 2 || lon.size() != 2)
                throw std::runtime_error("setGeoBox expects two tuples of size 2");

            self.setGeoBox(
                sycl::double2(lat[0].cast<double>(), lat[1].cast<double>()),
                sycl::double2(lon[0].cast<double>(), lon[1].cast<double>())
            );
        })
        .def("setDepth", &MOPS::SamplingSettings::setDepth)
        .def("getDepth", &MOPS::SamplingSettings::getDepth);  

    py::class_<MOPS::TrajectoryLine>(m, "TrajectoryLine")
        .def_readwrite("lineID", &MOPS::TrajectoryLine::lineID)
        .def_readwrite("points", &MOPS::TrajectoryLine::points)
        .def_readwrite("velocity", &MOPS::TrajectoryLine::velocity)
        .def_readwrite("lastPoint", &MOPS::TrajectoryLine::lastPoint)
        .def_readwrite("duration", &MOPS::TrajectoryLine::duration)
        .def_readwrite("timestamp", &MOPS::TrajectoryLine::timestamp)
        .def_readwrite("depth", &MOPS::TrajectoryLine::depth)
        .def("__repr__", [](const MOPS::TrajectoryLine& t) { return "<TrajectoryLine>"; });

    py::class_<MOPS::TrajectorySettings>(m, "TrajectorySettings")
        .def(py::init<>())
        .def_readwrite("depth", &MOPS::TrajectorySettings::depth)
        .def_readwrite("particle_depths", &MOPS::TrajectorySettings::particle_depths)  // NEW: per-particle depths
        .def_readwrite("deltaT", &MOPS::TrajectorySettings::deltaT)
        .def_readwrite("simulationDuration", &MOPS::TrajectorySettings::simulationDuration)
        .def_readwrite("recordT", &MOPS::TrajectorySettings::recordT)
        .def_readwrite("directionType", &MOPS::TrajectorySettings::directionType)
        .def_readwrite("methodType", &MOPS::TrajectorySettings::methodType)
        .def_readwrite("fileName", &MOPS::TrajectorySettings::fileName)
        .def("hasPerParticleDepths", &MOPS::TrajectorySettings::hasPerParticleDepths);  // NEW: helper method
    
    py::class_<CartesianCoord>(m, "CartesianCoord")
        .def(py::init<>())
        .def(py::init<double, double, double>())
        .def("x", [](const CartesianCoord& self) { return self.x(); })
        .def("y", [](const CartesianCoord& self) { return self.y(); })
        .def("z", [](const CartesianCoord& self) { return self.z(); })
        .def("__repr__", [](const CartesianCoord& v) {
            return "<CartesianCoord>";
        });
    
    
    
    
    
    m.def("MOPS_Init", &MOPS::MOPS_Init);
    m.def("MOPS_Begin", &MOPS::MOPS_Begin);
    m.def("MOPS_End", &MOPS::MOPS_End);
    m.def("MOPS_AddGridMesh", &MOPS::MOPS_AddGridMesh);
    m.def("MOPS_AddAttribute", &MOPS::MOPS_AddAttribute);
    m.def("MOPS_ActiveAttribute", &MOPS::MOPS_ActiveAttribute);
    m.def("MOPS_RunRemapping", [](MOPS::VisualizationSettings* config) {
        auto img_vec = MOPS::app.runRemapping(config);

        std::vector<py::array_t<double>> py_images;
        for (auto& img : img_vec) {
            auto h = img.getHeight();
            auto w = img.getWidth();
            auto data = img.mPixels.data();
            // shape = [h, w, 4]
            py_images.emplace_back(py::array_t<double>({h, w, 4}, data));
        }
        return py_images;
        });

    m.def("MOPS_GenerateSeedsPoints", [](MOPS::SamplingSettings* setting) {
        std::vector<CartesianCoord> pts;
        MOPS::MOPS_GenerateSamplePoints(setting, pts);

        ssize_t n_pts = pts.size();
        std::vector<ssize_t> shape = {n_pts, 3};
        std::vector<ssize_t> strides = {sizeof(double) * 3, sizeof(double)};
        py::array_t<double> arr(shape, strides);

        auto buf = arr.mutable_unchecked<2>();
        for (size_t i = 0; i < pts.size(); ++i) {
            buf(i, 0) = pts[i].x();
            buf(i, 1) = pts[i].y();
            buf(i, 2) = pts[i].z();
        }
        return arr;
        });
    m.def("MOPS_RunStreamLine", 
        [](MOPS::TrajectorySettings* config, py::array_t<double> sample_points_np) {
            // check shape
            if (sample_points_np.ndim() != 2 || sample_points_np.shape(1) != 3) {
                throw std::runtime_error("Input sample_points must be a (N, 3) numpy array.");
            }
            // convert to vector
            std::vector<CartesianCoord> sample_points_vec;
            auto r = sample_points_np.unchecked<2>(); 
            for (ssize_t i = 0; i < r.shape(0); ++i) {
                sample_points_vec.emplace_back(r(i, 0), r(i, 1), r(i, 2));
            }

            auto traj_lines = MOPS::MOPS_RunStreamLine(config, sample_points_vec);

            py::list out;
            const double kNaN = std::numeric_limits<double>::quiet_NaN();

            for (const auto& ln : traj_lines) {
                const size_t np = ln.points.size();
                const size_t nv = ln.velocity.size();

                // (N,3) arrays
                py::array_t<double> pts({(py::ssize_t)np, (py::ssize_t)3});
                py::array_t<double> vel({(py::ssize_t)np, (py::ssize_t)3});

                auto P = pts.mutable_unchecked<2>();
                auto V = vel.mutable_unchecked<2>();

                for (size_t i = 0; i < np; ++i) {
                    const auto& p = ln.points[i];
                    P(i,0) = p.x(); P(i,1) = p.y(); P(i,2) = p.z();

                    if (i < nv) {
                        const auto& v = ln.velocity[i];
                        V(i,0) = v.x(); V(i,1) = v.y(); V(i,2) = v.z();
                    } else {
                        V(i,0) = V(i,1) = V(i,2) = kNaN;
                    }
                }

                py::dict d;
                d["lineID"]    = ln.lineID;      // keep if available; remove if not needed
                d["points"]    = std::move(pts);
                d["velocity"]  = std::move(vel);

                

                out.append(std::move(d));
            }

            return out;
        },
        "Run streamline simulation");

    m.def("MOPS_RunPathLine", 
        [](MOPS::TrajectorySettings* config, 
            py::array_t<double> sample_points_np) {
            
            // check shape
            if (sample_points_np.ndim() != 2 || sample_points_np.shape(1) != 3) {
                throw std::runtime_error("Input sample_points must be a (N, 3) numpy array.");
            }
            // convert to vector
            std::vector<CartesianCoord> sample_points_vec;
            auto r = sample_points_np.unchecked<2>(); 
            for (ssize_t i = 0; i < r.shape(0); ++i) {
                sample_points_vec.emplace_back(r(i, 0), r(i, 1), r(i, 2));
            }

            auto traj_lines = MOPS::MOPS_RunPathLine(config, sample_points_vec);

            py::list out;
            const double double_NaN = std::numeric_limits<double>::quiet_NaN();

            for (const auto& ln : traj_lines) {
                const size_t np = ln.points.size();
                const size_t nv = ln.velocity.size();
                const size_t nt = ln.temperature.size();
                const size_t ns = ln.salinity.size();

                py::array_t<double> pts({(py::ssize_t)np, (py::ssize_t)3});
                py::array_t<double> vel({(py::ssize_t)np, (py::ssize_t)3});
                py::array_t<double> tem((py::ssize_t)np);
                py::array_t<double> sal((py::ssize_t)np);

                auto P = pts.mutable_unchecked<2>();
                auto V = vel.mutable_unchecked<2>();
                auto T = tem.mutable_unchecked<1>();
                auto S = sal.mutable_unchecked<1>();

                for (size_t i = 0; i < np; ++i) {
                    // points
                    const auto& p = ln.points[i];
                    P(i,0) = p.x(); P(i,1) = p.y(); P(i,2) = p.z();

                    const auto& v = ln.velocity[i];
                    V(i,0) = v.x(); V(i,1) = v.y(); V(i,2) = v.z();

                    T(i) = ln.temperature[i];
                    S(i) = ln.salinity[i];
                }

                py::array_t<double> last_pt({(py::ssize_t)3});
                {
                    auto L = last_pt.mutable_unchecked<1>();
                    L(0) = ln.lastPoint.x();
                    L(1) = ln.lastPoint.y();
                    L(2) = ln.lastPoint.z();
                }


                py::dict py_line;
                py_line["lineID"] = ln.lineID;
                py_line["points"] = std::move(pts);
                py_line["velocity"] = std::move(vel);
                py_line["temperature"] = std::move(tem);
                py_line["salinity"] = std::move(sal);
                py_line["lastPoint"] = std::move(last_pt);
                py_line["depth"] = ln.depth;  // per-particle depth
                out.append(py_line);
            }

            return out;
        },
        py::arg("config"),
        py::arg("sample_points_np"),
        "Run pathline simulation");

    // ========================================================================
    // Timing API
    // ========================================================================
    m.def("MOPS_ResetTiming", &MOPS::MOPS_ResetTiming,
        "Reset all timing data");

    m.def("MOPS_PrintTimingSummary", &MOPS::MOPS_PrintTimingSummary,
        "Print timing summary grouped by category (IO, Preprocessing, GPU Kernel, etc.)");

    m.def("MOPS_PrintTimingDetailed", &MOPS::MOPS_PrintTimingDetailed,
        "Print detailed timing for each individual operation");

    m.def("MOPS_GetCategoryTime", &MOPS::MOPS_GetCategoryTime,
        py::arg("category"),
        "Get total time (ms) spent in a category. Valid categories: "
        "'IO_Read', 'IO_Write', 'Preprocessing', 'MemoryCopy', 'GPUKernel', 'CPUCompute', 'Other'");

    m.def("MOPS_GetTotalTime", &MOPS::MOPS_GetTotalTime,
        "Get total recorded time in milliseconds");
}
