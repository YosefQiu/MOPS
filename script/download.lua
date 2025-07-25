local function get_pwd()
    local f = io.popen("pwd")
    local path = f:read("*l")
    f:close()
    return path
end

local function run(cmd)
    print("Running: " .. cmd)
    local ok, reason, code = os.execute(cmd)
    if not ok then
        error("Command failed: " .. cmd)
    end
end

local function file_exists(path)
    local f = io.open(path, "r")
    if f then f:close() return true else return false end
end

local function need_install(name, paths)
    for _, path in ipairs(paths) do
        if file_exists(path) then
            print("[" .. name .. "] Already installed: " .. path)
            return false
        end
    end
    print("[" .. name .. "] Not found. Will install.")
    return true
end

local disable_map = {}

for i = 1, #arg - 1, 2 do
    local key = arg[i]
    local val = arg[i + 1]
    if key and val and key:sub(1, 2) == "--" and val == "OFF" then
        local name = key:sub(3):upper()
        disable_map[name] = true
    end
end

local function should_skip(name)
    return disable_map[name:upper()] == true
end

local project_root = get_pwd()
local third_lib_dir = project_root .. "/third_lib"
run("mkdir -p " .. third_lib_dir)


-- NetCDF
if not should_skip("NetCDF") and need_install("NetCDF", {
    third_lib_dir .. "/lib/libnetcdf.a",
    third_lib_dir .. "/lib/libnetcdf.so",
    third_lib_dir .. "/lib/libnetcdf.dylib",
    third_lib_dir .. "/lib64/libnetcdf.a",
    third_lib_dir .. "/lib64/libnetcdf.so",
    third_lib_dir .. "/lib64/libnetcdf.dylib"
}) then
    run("rm -rf netcdf-c")
    run("git clone https://github.com/Unidata/netcdf-c.git")
    run("mkdir -p netcdf-c/build")
    run(string.format([[
    cd netcdf-c/build && cmake .. \
        -DCMAKE_INSTALL_PREFIX=%s \
        -DnetCDF_INSTALL_PREFIX=%s \
        -DBUILD_SHARED_LIBS=ON \
        -DENABLE_NETCDF_4=ON \
        -DENABLE_DAP=OFF \
        -DENABLE_TESTS=OFF
    ]], third_lib_dir, third_lib_dir))
    run("cd netcdf-c/build && make -j8 && make install")
    print("[NetCDF] Installed to: " .. third_lib_dir)
end


--Yaml-cpp
if not should_skip("Yaml-cpp") and need_install("yaml-cpp", {
    third_lib_dir .. "/lib/libyaml-cpp.a",
    third_lib_dir .. "/lib/libyaml-cpp.dylib",
    third_lib_dir .. "/lib/libyaml-cpp.so",
    third_lib_dir .. "/lib/cmake/yaml-cpp/yaml-cpp-config.cmake",
    third_lib_dir .. "/lib64/libyaml-cpp.a",
    third_lib_dir .. "/lib64/libyaml-cpp.dylib",
    third_lib_dir .. "/lib64/libyaml-cpp.so",
    third_lib_dir .. "/lib64/cmake/yaml-cpp/yaml-cpp-config.cmake"
}) then
    run("rm -rf yaml-cpp")
    run("git clone https://github.com/jbeder/yaml-cpp.git")
    run("mkdir -p yaml-cpp/build")
    run(string.format([[
    cd yaml-cpp/build && cmake .. \
      -DCMAKE_INSTALL_PREFIX=%s \
      -DBUILD_SHARED_LIBS=ON \
      -DYAML_CPP_BUILD_TESTS=OFF
    ]], third_lib_dir))
    run("cd yaml-cpp/build && make -j8 && make install")
    print("[Yaml-CPP] Installed to: " .. third_lib_dir)
end


-- ndarray
if not should_skip("ndarray") and need_install("ndarray", {
    third_lib_dir .. "/lib/libndarray.a",
    third_lib_dir .. "/lib/libndarray.so",
    third_lib_dir .. "/lib/libndarray.dylib",
    third_lib_dir .. "/lib64/libndarray.a",
    third_lib_dir .. "/lib64/libndarray.so",
    third_lib_dir .. "/lib64/libndarray.dylib"
}) then
    run("rm -rf ndarray")
    run("git clone https://github.com/hguo/ndarray.git")
    run("mkdir -p ndarray/build")

    run(string.format([[
    cd ndarray/build && cmake .. \
      -DCMAKE_INSTALL_PREFIX=%s \
      -DCMAKE_PREFIX_PATH="%s" \
      -DNDARRAY_USE_NETCDF=ON \
      -DBUILD_TESTING=OFF
    ]], third_lib_dir, third_lib_dir))

    run("cd ndarray/build && make -j8 && make install")
    print("[ndarray] Installed to: " .. third_lib_dir)
end

--pybind11
if not should_skip("pybind11") and need_install("pybind11", {
    third_lib_dir .. "/share/cmake/pybind11/pybind11Config.cmake",
}) then
    run("rm -rf pybind11")
    run("git clone https://github.com/pybind/pybind11.git")
    run("mkdir -p pybind11/build")
    run(string.format([[
        cd pybind11/build && cmake .. \
            -DCMAKE_INSTALL_PREFIX=%s \
            -DPYBIND11_TEST=OFF
    ]], third_lib_dir))
    run("cd pybind11/build && make -j8 && make install")
    print("[pybind11] Installed to: " .. third_lib_dir)
end



--vtk
if not should_skip("VTK") and need_install("VTK", {
    third_lib_dir .. "/lib/cmake/vtk-9.2/vtk-config.cmake",
    third_lib_dir .. "/lib64/cmake/vtk-9.2/vtk-config.cmake"
}) then
    run("rm -rf vtk")
    run("git clone https://gitlab.kitware.com/vtk/vtk.git")
    run("cd vtk && git checkout v9.2.6")
    run("mkdir -p vtk/build")

    run(string.format([[
    cd vtk/build && cmake .. \
      -DCMAKE_INSTALL_PREFIX=%s \
      -DCMAKE_PREFIX_PATH=%s \
      -DCMAKE_CXX_STANDARD=17 \
      -DBUILD_SHARED_LIBS=ON \
      -DVTK_BUILD_TESTING=OFF \
      -DVTK_BUILD_EXAMPLES=OFF 
    ]], third_lib_dir, third_lib_dir))
    run("cd vtk/build && make -j8 && make install")
    print("[VTK] Installed to: " .. third_lib_dir)
end


