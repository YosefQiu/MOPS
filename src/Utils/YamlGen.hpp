#pragma once
#include "ggl.h"


namespace MOPS_IO
{
    static const char* kYAMLTemplate = R"(
stream:
  name: mpas
  path_prefix: "__PATH_PREFIX__"
  substreams:
    - name: mesh
      format: netcdf
      filenames: "__MESH_FILE__"
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
        - name: indexToCellID
        - name: indexToEdgeID
        - name: indexToVertexID
        - name: nEdgesOnCell
        - name: nEdgesOnEdge
        - name: cellsOnCell
        - name: cellsOnEdge
        - name: cellsOnVertex
        - name: edgesOnVertex
        - name: edgesOnCell
        - name: edgesOnEdge
        - name: verticesOnCell
        - name: verticesOnEdge
    - name: data
      format: netcdf
      filenames: "__DATA_FILE__"
      vars:
        - name: xtime # the name will be the variable name in netCDF if possible_names is not given
          possible_names: 
            - xtime 
            - xtime_startMonthly
            - xtime_startDaily
            - xtime_endMonthly
          dimensions: auto # by default auto
          optional: false # by default false
          multicomponents: false # if the inputs are not multicomponents, the resulting array will be made multicomponents
        - name: normalVelocity
          possible_names:
            - normalVelocity
            - timeMonthly_avg_normalVelocity
            - timeDaily_avg_normalVelocity
          dimensions: auto
          multicomponents: true
        - name: velocityMeridional
          possible_names:
            - velocityMeridional
            - timeMonthly_avg_velocityMeridional
            - timeDaily_avg_velocityMeridional
          multicomponents: true
        - name: velocityZonal
          possible_names:
            - velocityZonal
            - timeMonthly_avg_velocityZonal
            - timeDaily_avg_velocityZonal
          multicomponents: true
        - name: vertVelocityTop
          possible_names:
            - vertVelocityTop
            - timeMonthly_avg_vertVelocityTop
          multicomponents: true
        - name: salinity
          possible_names:
            - salinity
            - activeTracers_salinity
            - timeDaily_avg_activeTracers_salinity
          optional: true
          multicomponents: true
        - name: temperature
          possible_names:
            - temperature
            - timeDaily_avg_activeTracers_temperature
            - activeTracers_temperature
          optional: true
          multicomponents: true
        - name: zTop
          possible_names:
            - zTop
            - timeMonthly_avg_zTop
          optional: true
          multicomponents: true
        - name: zMid
          possible_names:
            - zMid
            - timeMonthly_avg_zMid
          optional: true
          multicomponents: true
        - name: layerThickness
          possible_names: 
            - layerThickness
            - timeMonthly_avg_layerThickness
            - timeDaily_avg_layerThickness
          optional: true
          multicomponents: true
        - name: bottomDepth
          possible_names:
            - bottomDepth
        - name: seaSurfaceHeight
          possible_names:
            - seaSurfaceHeight
            - timeMonthly_avg_ssh
            - timeDaily_avg_seaSurfaceHeight
          optional: true
          multicomponents: true

    )";

    struct NameTemplates
    {
        std::string mesh_tpl;
        std::string data_tpl;
    };

    struct YMD
    {
        int year;
        int month;
        int day;

        YMD(int y, int m = 1, int d = 1) : year(y), month(m), day(d) {}
    };

    inline bool operator<(const YMD& a, const YMD& b) {
        return (a.year < b.year) || (a.year == b.year && a.month < b.month) || (a.year == b.year && a.month == b.month && a.day < b.day);
    }
    inline bool operator==(const YMD& a, const YMD& b) { return a.year==b.year && a.month==b.month && a.day==b.day; }
    inline bool operator<=(const YMD& a, const YMD& b) { return (a < b) || (a == b); }
    inline bool is_leap(int y) {
        return (y%4==0 && y%100!=0) || (y%400==0);
    }
    inline int days_in_month(int y, int m) {
        static const int mdays[12] = {31,28,31,30,31,30,31,31,30,31,30,31};
        if (m == 2) return is_leap(y) ? 29 : 28;
        return mdays[m-1];
    }
    inline YMD operator+(const YMD& lhs, const YMD& delta) 
    {
        YMD result = lhs;
        result.year  += delta.year;
        result.month += delta.month;
        result.day   += delta.day;

        while (result.month > 12) { result.month -= 12; result.year++; }
        while (result.month < 1)  { result.month += 12; result.year--; }

        int dim = days_in_month(result.year, result.month);
        while (result.day > dim) {
            result.day -= dim;
            result.month++;
            if (result.month > 12) { result.month = 1; result.year++; }
            dim = days_in_month(result.year, result.month);
        }
        while (result.day < 1) {
            result.month--;
            if (result.month < 1) { result.month = 12; result.year--; }
            result.day += days_in_month(result.year, result.month);
        }

        return result;
    }

    inline YMD fromStringYMD(const std::string& str)
    {
        if (str.size() != 6 && str.size() != 8)
            throw std::invalid_argument("Bad date string format, expect YYMMDD or YYYYMMDD");

        int y, m, d;
        if (str.size() == 6) {
            y = std::stoi(str.substr(0, 2));
            m = std::stoi(str.substr(2, 2));
            d = std::stoi(str.substr(4, 2));
        } else {
            y = std::stoi(str.substr(0, 4));
            m = std::stoi(str.substr(4, 2));
            d = std::stoi(str.substr(6, 2));
        }
        return MOPS_IO::YMD{y, m, d};
    }

    inline YMD next_month(YMD a)
    {
        int y = a.year;
        int m = a.month + 1;
        if (m > 12) { m = 1; y += 1; }
        return YMD(y, m, a.day);
    }

    inline YMD prev_month(YMD a)
    {
        int y = a.year;
        int m = a.month - 1;
        if (m < 1) { m = 12; y -= 1; }
        return YMD(y, m, a.day);
    }

    inline std::string pad_int(int v, int width)
    {
        std::ostringstream oss;
        oss << std::setw(width) << std::setfill('0') << v;
        return oss.str();
    }

    inline std::string replace_all(std::string s, const std::string& from, const std::string& to)
    {
        if(from.empty()) return s;
        size_t pos = 0;
        while((pos = s.find(from, pos)) != std::string::npos) 
        {
            s.replace(pos, from.length(), to);
            pos += to.length();
        }
        return s;
    }

    inline std::string render_name(const std::string& tpl, int year, int month = 1, int day = 1)
    {
        std::string out = tpl;
        out = replace_all(out, "{YYYY}", pad_int(year, 4));
        out = replace_all(out, "{YY}",   pad_int(year % 100, 2));
        out = replace_all(out, "{MM}",   pad_int(month, 2));
        out = replace_all(out, "{DD}",   pad_int(day, 2));
        return out;
    }

    inline std::string mesh_file_for_year(const NameTemplates& tpls, int year, int month = 1, int day = 1) 
    {
        return render_name(tpls.mesh_tpl, year, month, day);
    }

    inline std::string mesh_file_for_year(const NameTemplates& tpls, int year)
    {
        return render_name(tpls.mesh_tpl, year);
    }

    inline std::string data_file_for_ymd(const NameTemplates& tpls, int year, int month = 1, int day = 1) 
    {
        return render_name(tpls.data_tpl, year, month, day);
    }

    inline std::string data_file_for_ymd(const NameTemplates& tpls, int year, int month) 
    {
        return render_name(tpls.data_tpl, year, month);
    }

    inline std::string make_temp_yaml_path(const std::string& prefix = "mpas_cfg_") 
    {
        namespace fs = std::filesystem;
        static thread_local std::mt19937_64 rng{std::random_device{}()};
        static thread_local std::uniform_int_distribution<uint64_t> dist;

        const auto tmpdir = fs::temp_directory_path(); 
        uint64_t r = dist(rng);
        fs::path p = tmpdir / (prefix + std::to_string(r) + ".yaml");
        return p.string();
    }

    inline std::string default_temp_dir() 
    {
        namespace fs = std::filesystem;
        fs::path p = fs::current_path() / ".tmp_yaml";
        std::error_code ec; fs::create_directories(p, ec); 
        return p.string();
    }

    inline std::string make_unique_yaml_path(const std::string& out_dir = default_temp_dir(), const std::string& tag = "") 
    {
        namespace fs = std::filesystem;
        static thread_local std::mt19937_64 rng{std::random_device{}()};
        uint64_t r = rng();
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();

        fs::path base = out_dir.empty() ? fs::path(default_temp_dir()) : fs::path(out_dir);
        std::ostringstream name;
        name << "mpas_cfg";
        if (!tag.empty()) name << "_" << tag;
        name << "_" << now << "_" << std::hex << r << ".yaml";
        fs::create_directories(base); 
        return (base / name.str()).string();
    }


    inline std::string write_temp_yaml(const std::string& path_prefix,
                                   const std::string& mesh_filename,
                                   const std::string& data_filename,
                                   const std::string& out_path = "") 
    {
        std::string yaml(kYAMLTemplate);
        yaml = replace_all(yaml, "__PATH_PREFIX__", path_prefix);
        yaml = replace_all(yaml, "__MESH_FILE__",   mesh_filename);
        yaml = replace_all(yaml, "__DATA_FILE__",   data_filename);

        std::string out = out_path.empty() ? make_unique_yaml_path() : out_path;

        std::filesystem::create_directories(std::filesystem::path(out).parent_path());

        std::ofstream ofs(out, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error("Failed to open temp yaml for write: " + out);
        }
        ofs << yaml;
        ofs.close();
        return out;
    }

    struct YamlPair
    {
        std::string a_path;
        std::string b_path;
    };

    inline YamlPair make_yaml_pair(const std::string& path_prefix, const std::string& mesh_A_filename, const std::string& mesh_B_filename, const std::string& data_A_filename, const std::string& data_B_filename, const std::string& out_dir = "") 
    {
        const std::string dir  = out_dir.empty() ? default_temp_dir() : out_dir;
        const std::string outA = make_unique_yaml_path(dir, "A");
        const std::string outB = make_unique_yaml_path(dir, "B");

        YamlPair pair;
        pair.a_path = write_temp_yaml(path_prefix, mesh_A_filename, data_A_filename, outA);
        pair.b_path = write_temp_yaml(path_prefix, mesh_B_filename, data_B_filename, outB);
        return pair;
    }

    inline void cleanup_temp_yamls(const std::vector<std::string>& paths) 
    {
        namespace fs = std::filesystem;
        for (const auto& p : paths) 
        {
            std::error_code ec;
            fs::remove(p, ec);
            if (ec) 
            {
                std::cerr << "[CleanTempYamls]::Warning: failed to remove temp yaml: " << p << ", error: " << ec.message() << std::endl;
            } 
        }
    }

    inline std::string make_date_tag(int year, int month, int day=1) 
    {
            std::ostringstream oss;
            oss << std::setfill('0') << std::setw(4) << year
                << "-" << std::setw(2) << month
                << "-" << std::setw(2) << day;
            return oss.str(); // example: 0015-04-01
    }

    inline std::vector<std::pair<std::string, std::string>> make_forward_month_pairs(int start_year, int start_month,
                            int end_year,   int end_month)
    {
            using namespace MOPS_IO;
            std::vector<std::pair<std::string, std::string>> out;

            YMD A{start_year, start_month, 1};
            YMD E{end_year,   end_month,   1};
            if (!(A <= E)) return out;

            for (; A <= E; A = next_month(A)) {
                YMD B = next_month(A);
                if (!(B <= E)) break;
                out.emplace_back(
                    make_date_tag(A.year, A.month, A.day),
                    make_date_tag(B.year, B.month, B.day)
                );
            }
            return out;
    }


    inline std::vector<std::pair<std::string, std::string>> make_backward_month_pairs(int start_year, int start_month,
                            int end_year,   int end_month)
    {
        using namespace MOPS_IO;
        std::vector<std::pair<std::string, std::string>> out;

        YMD A{start_year, start_month, 1};
        YMD E{end_year,   end_month,   1};
        if (!(E <= A)) return out;

        for (; E <= A; A = prev_month(A)) {
            YMD B = prev_month(A);
            if (B < E) break;
            out.emplace_back(
                make_date_tag(A.year, A.month, A.day),
                make_date_tag(B.year, B.month, B.day)
            );
        }
        return out;
    }

    
};


namespace verify_io 
{

    struct YMD { int y, m, d; }; // d == 1
    inline YMD next_month(YMD a) { if (++a.m>12){a.m=1;++a.y;} return a; }
    inline YMD prev_month(YMD a) { if (--a.m<1){a.m=12;--a.y;} return a; }

    struct ParsedPaths {
        std::string mesh; // relative to path_prefix
        std::string data; // relative to path_prefix
    };

    // given a yaml path, parse out the mesh and data filenames
    inline ParsedPaths parse_yaml_paths(const std::string& yaml_path) {
        std::ifstream ifs(yaml_path);
        if (!ifs) throw std::runtime_error("open yaml failed: " + yaml_path);
        std::string line;
        ParsedPaths out;
        int found = 0;
       
        static const std::regex re(R"(filenames:\s*\"([^\"]+)\")");
        while (std::getline(ifs, line)) {
            std::smatch m;
            if (std::regex_search(line, m, re)) {
                if (++found == 1) out.mesh = m[1].str();
                else if (found == 2) { out.data = m[1].str(); break; }
            }
        }
        if (found < 2) throw std::runtime_error("filenames not found twice in yaml: " + yaml_path);
        return out;
    }

    // given data filename, parse out YYYY and MM
    inline bool parse_data_ym(const std::string& data_name, int& yy, int& mm) {
        // ...Monthly.0018-02-01.nc
        static const std::regex r(R"((\d{4})-(\d{2})-01\.nc$)");
        std::smatch m;
        if (!std::regex_search(data_name, m, r)) return false;
        yy = std::stoi(m[1].str());
        mm = std::stoi(m[2].str());
        return true;
    }

    // given mesh filename, parse out the year (YYYY)
    inline bool parse_mesh_y(const std::string& mesh_name, int& yy) {
        static const std::regex r(R"((\d{4})-01-01_00000\.nc$)");
        std::smatch m;
        if (!std::regex_search(mesh_name, m, r)) return false;
        yy = std::stoi(m[1].str());
        return true;
    }

    struct CheckResult {
        size_t total_pairs = 0;
        size_t ok_pairs = 0;
        size_t missing_files = 0;
        size_t date_mismatch = 0;
        size_t mesh_mismatch = 0;
    };

    template <typename PairVec>
    CheckResult verify_forward_batch(const PairVec& pairs,
                                    const std::string& path_prefix,
                                    int start_year, int end_year)
    {
        using std::filesystem::exists;
        CheckResult stat{};
        stat.total_pairs = pairs.size();

        // expected: B = next(A)
        for (size_t i = 0; i < pairs.size(); ++i) {
            const auto& pr = pairs[i];
           
            ParsedPaths A = parse_yaml_paths(pr.a_path);
            ParsedPaths B = parse_yaml_paths(pr.b_path);

         
            bool ok_exist = true;
            if (!exists(std::filesystem::path(path_prefix) / A.mesh)) ok_exist = false;
            if (!exists(std::filesystem::path(path_prefix) / A.data)) ok_exist = false;
            if (!exists(std::filesystem::path(path_prefix) / B.mesh)) ok_exist = false;
            if (!exists(std::filesystem::path(path_prefix) / B.data)) ok_exist = false;
            if (!ok_exist) { ++stat.missing_files; continue; }

         
            int Ay, Am, By, Bm;
            if (!parse_data_ym(A.data, Ay, Am) || !parse_data_ym(B.data, By, Bm)) {
                ++stat.date_mismatch;
                std::cerr << "[DATE_PARSE_FAIL] #" << i << " A.data="<<A.data<<" B.data="<<B.data<<"\n";
                continue;
            }
           
            auto Aexp = YMD{Ay,Am,1};
            auto Bexp = next_month(Aexp);
            if (!(By==Bexp.y && Bm==Bexp.m)) {
                ++stat.date_mismatch;
                std::cerr << "[DATE_SEQ_FAIL] #" << i << " got A="<<Ay<<"-"<<Am<<" B="<<By<<"-"<<Bm
                        << " expected B="<<Bexp.y<<"-"<<Bexp.m << "\n";
                continue;
            }

            
            int AmeshY=0, BmeshY=0;
            if (!parse_mesh_y(A.mesh, AmeshY) || !parse_mesh_y(B.mesh, BmeshY)) {
                ++stat.mesh_mismatch;
                std::cerr << "[MESH_PARSE_FAIL] #" << i << " A.mesh="<<A.mesh<<" B.mesh="<<B.mesh<<"\n";
                continue;
            }
            if (AmeshY != Ay || BmeshY != By) {
                ++stat.mesh_mismatch;
                std::cerr << "[MESH_YEAR_FAIL] #" << i
                        << " meshA="<<AmeshY<<" vs Ay="<<Ay
                        << " | meshB="<<BmeshY<<" vs By="<<By << "\n";
                continue;
            }

            ++stat.ok_pairs;
        }
        return stat;
    }

    template <typename PairVec>
    CheckResult verify_backward_batch(const PairVec& pairs,
                                    const std::string& path_prefix,
                                    int start_year, int end_year)
    {
        using std::filesystem::exists;
        CheckResult stat{};
        stat.total_pairs = pairs.size();

        for (size_t i = 0; i < pairs.size(); ++i) {
            const auto& pr = pairs[i];
            ParsedPaths A = parse_yaml_paths(pr.a_path);
            ParsedPaths B = parse_yaml_paths(pr.b_path);

            bool ok_exist = true;
            if (!exists(std::filesystem::path(path_prefix) / A.mesh)) ok_exist = false;
            if (!exists(std::filesystem::path(path_prefix) / A.data)) ok_exist = false;
            if (!exists(std::filesystem::path(path_prefix) / B.mesh)) ok_exist = false;
            if (!exists(std::filesystem::path(path_prefix) / B.data)) ok_exist = false;
            if (!ok_exist) { ++stat.missing_files; continue; }

            int Ay, Am, By, Bm;
            if (!parse_data_ym(A.data, Ay, Am) || !parse_data_ym(B.data, By, Bm)) {
                ++stat.date_mismatch;
                std::cerr << "[DATE_PARSE_FAIL] #" << i << " A.data="<<A.data<<" B.data="<<B.data<<"\n";
                continue;
            }
            // backward：B = prev(A)
            auto Aexp = YMD{Ay,Am,1};
            auto Bexp = prev_month(Aexp);
            if (!(By==Bexp.y && Bm==Bexp.m)) {
                ++stat.date_mismatch;
                std::cerr << "[DATE_SEQ_FAIL] #" << i << " got A="<<Ay<<"-"<<Am<<" B="<<By<<"-"<<Bm
                        << " expected B="<<Bexp.y<<"-"<<Bexp.m << "\n";
                continue;
            }

            int AmeshY=0, BmeshY=0;
            if (!parse_mesh_y(A.mesh, AmeshY) || !parse_mesh_y(B.mesh, BmeshY)) {
                ++stat.mesh_mismatch;
                std::cerr << "[MESH_PARSE_FAIL] #" << i << " A.mesh="<<A.mesh<<" B.mesh="<<B.mesh<<"\n";
                continue;
            }
            if (AmeshY != Ay || BmeshY != By) {
                ++stat.mesh_mismatch;
                std::cerr << "[MESH_YEAR_FAIL] #" << i
                        << " meshA="<<AmeshY<<" vs Ay="<<Ay
                        << " | meshB="<<BmeshY<<" vs By="<<By << "\n";
                continue;
            }

            ++stat.ok_pairs;
        }
        return stat;
    }

    inline void print_summary(const CheckResult& r, const std::string& tag) {
        std::cout << "[VERIFY " << tag << "] total=" << r.total_pairs
                << " ok=" << r.ok_pairs
                << " missing=" << r.missing_files
                << " date_bad=" << r.date_mismatch
                << " mesh_bad=" << r.mesh_mismatch << "\n";
    }
}; // namespace verify_io