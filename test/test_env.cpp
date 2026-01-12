#include "catch.hpp"
#include "ggl.h"
#include "api/MOPS.h"


#ifdef MOPS_VTK
  #include <vtkSmartPointer.h>
  #include <vtkVersion.h>
  #include <vtkRenderer.h>  
#endif


static bool try_init(const char* device, std::string& err) {
    try {
        MOPS::MOPS_Init(device);           
        return true;
    } catch (const sycl::exception& e) {
        err = e.what();
        return false;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    } catch (...) {
        err = "unknown exception";
        return false;
    }
}

static bool sycl_example() {
    sycl::buffer<sycl::opencl::cl_int, 1> Buffer(4);
    sycl::queue Queue;
    sycl::range<1> NumOfWorkItems{Buffer.size()};
    Queue.submit([&](sycl::handler &cgh) {
        auto Accessor = Buffer.get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for<class FillBuffer>(
            NumOfWorkItems, [=](sycl::id<1> WIid) {
            Accessor[WIid] = (sycl::opencl::cl_int)WIid.get(0);
            });
    });
    const auto HostAccessor = Buffer.get_host_access(sycl::read_only);;

    bool MismatchFound = false;
    for (size_t I = 0; I < Buffer.size(); ++I) {
        if (HostAccessor[I] != I) {
        std::cout << "The result is incorrect for element: " << I
                    << " , expected: " << I << " , got: " << HostAccessor[I]
                    << std::endl;
        MismatchFound = true;
        }
    }

    if (!MismatchFound) {
        std::cout << "The results are correct!" << std::endl;
    }

    return MismatchFound;
}

TEST_CASE("SYCL runtime/device availability via MOPSApp::init", "[env][sycl]") {

    SECTION("GPU environment initialization") {
        std::string err;
        const bool ok = try_init("gpu", err);

        if (!ok) {
            WARN("GPU init failed. Error: " << err);
            SUCCEED("GPU checks skipped.");
        } else {
            SUCCEED("GPU init succeeded.");
        }
    }

    SECTION("SYCL example execution") {
        REQUIRE(sycl_example() == false);
    }
}
#ifdef MOPS_VTK
TEST_CASE("VTK runtime sanity", "[env][vtk]") {

    INFO("VTK version: " << vtkVersion::GetVTKVersion());
    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    REQUIRE(renderer != nullptr);
}
#endif