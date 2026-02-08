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
