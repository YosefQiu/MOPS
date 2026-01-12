#include "catch.hpp"
#include "ggl.h"
#include "api/MOPS.h"
#include "ndarray/ndarray_group_stream.hh"
#include "IO/MPASOReader.h"
namespace fs = std::filesystem;
// ============== helpers ==============
static std::string make_tmp_path(const char* prefix, const char* suffix) {
    std::string tpl = std::string("/tmp/") + prefix + "_XXXXXX";
    std::vector<char> buf(tpl.begin(), tpl.end());
    buf.push_back('\0');

    int fd = mkstemp(buf.data());
    REQUIRE(fd != -1);
    close(fd);

    std::string path = std::string(buf.data()) + suffix;
    std::remove(buf.data()); // 删除 mkstemp 生成的空文件
    return path;
}

static void nc_check(int code, const char* where) {
    INFO(where << " -> " << nc_strerror(code));
    REQUIRE(code == NC_NOERR);
}

static bool file_exists(const std::string &path) {
    return std::filesystem::exists(path);
}

static void ensure_file_downloaded(const std::string& url, const std::string& local_name) {
    if (fs::exists(local_name)) return;
    {
        std::string cmd = "wget -c -O \"" + local_name + "\" \"" + url + "\"";
        int ret = std::system(cmd.c_str());
        if (ret == 0 && fs::exists(local_name)) return;
    }
    {
        std::string cmd = "curl -L -o \"" + local_name + "\" \"" + url + "\"";
        int ret = std::system(cmd.c_str());
        INFO("Download command: " << cmd);
        INFO("Target file: " << local_name);
        REQUIRE(ret == 0);
        REQUIRE(fs::exists(local_name));
    }

}
static void write_mpas_yaml_for_grid_only(const std::string& yaml_path,
                                         const std::string& path_prefix,
                                         const std::string& mesh_filename) {
    std::ofstream ofs(yaml_path);
    REQUIRE(ofs.is_open());

    ofs <<
R"(stream:
  name: mpas
  path_prefix: ")" << path_prefix << R"("
  substreams:
    - name: mesh
      format: netcdf
      filenames: ")" << mesh_filename << R"("
      static: true
      vars:
        - name: xCell
        - name: yCell
        - name: zCell
        - name: xEdge
        - name: yEdge
        - name: zEdge
        - name: xVertex
        - name: yVertex
        - name: zVertex
        - name: latVertex
        - name: lonVertex
        - name: verticesOnCell
        - name: verticesOnEdge
        - name: cellsOnVertex
        - name: cellsOnCell
        - name: nEdgesOnCell
        - name: cellsOnEdge
        - name: edgesOnCell
        - name: refBottomDepth
)";
    ofs.close();
    REQUIRE(fs::exists(yaml_path));
}

// =====================================
// yaml-cpp sanity
// =====================================
TEST_CASE("yaml-cpp: parse basic YAML", "[io][yaml]") {
    const char* s = R"(
name: mops
version: 1
flags:
  - vtk
  - sycl
)";

    YAML::Node n = YAML::Load(s);

    REQUIRE(n["name"].as<std::string>() == "mops");
    REQUIRE(n["version"].as<int>() == 1);

    REQUIRE(n["flags"].IsSequence());
    REQUIRE(n["flags"].size() == 2);
    REQUIRE(n["flags"][0].as<std::string>() == "vtk");
    REQUIRE(n["flags"][1].as<std::string>() == "sycl");
}

// =====================================
// netcdf-c sanity: write + read back
// =====================================
TEST_CASE("netcdf-c: create/write/read small dataset", "[io][netcdf]") {
    const std::string path = make_tmp_path("mops_netcdf", ".nc");

    int ncid = -1;
    int dimid = -1;
    int varid = -1;

    // Create file
    nc_check(nc_create(path.c_str(), NC_CLOBBER, &ncid), "nc_create");
    // Define dimension (len=4)
    nc_check(nc_def_dim(ncid, "n", 4, &dimid), "nc_def_dim");
    // Define variable (double[n])
    int dimids[1] = {dimid};
    nc_check(nc_def_var(ncid, "x", NC_DOUBLE, 1, dimids, &varid), "nc_def_var");
    // End define mode
    nc_check(nc_enddef(ncid), "nc_enddef");
    // Write data
    double w[4] = {1.0, 2.0, 3.0, 4.0};
    nc_check(nc_put_var_double(ncid, varid, w), "nc_put_var_double");
    // Close
    nc_check(nc_close(ncid), "nc_close (write)");
    ncid = -1;
    // Re-open and read back
    nc_check(nc_open(path.c_str(), NC_NOWRITE, &ncid), "nc_open");
    int varid2 = -1;
    nc_check(nc_inq_varid(ncid, "x", &varid2), "nc_inq_varid");
    double r[4] = {0,0,0,0};
    nc_check(nc_get_var_double(ncid, varid2, r), "nc_get_var_double");
    nc_check(nc_close(ncid), "nc_close (read)");
    for (int i = 0; i < 4; ++i) {
        INFO("i=" << i << " r=" << r[i] << " expected=" << w[i]);
        REQUIRE(r[i] == Approx(w[i]));
    }

    std::remove(path.c_str());
}

// =====================================
// ndarray sanity
// =====================================

TEST_CASE("ndarray: header is usable (compile/link sanity)", "[io][ndarray]") {
    SUCCEED("ndarray headers are included successfully.");
}

// =====================================
// MPASOGrid
// =====================================

