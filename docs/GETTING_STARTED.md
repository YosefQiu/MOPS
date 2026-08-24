# Getting Started with MOPS

**Welcome to MOPS!** This guide will help you get up and running with ocean particle simulation and visualization in just a few minutes.

MOPS (MPAS Ocean Particle Simulator) is a high-performance framework for visualizing ocean velocity fields and simulating particle trajectories on unstructured spherical meshes. Whether you're analyzing ocean currents, tracking pollutant dispersal, or studying larval transport, MOPS provides GPU-accelerated tools for both C++ and Python.

## Quick Start

The fastest way to see MOPS in action:

```cpp
#include "ggl.h"
#include "api/MOPS.h"
#include "IO/MPASOReader.h"

using namespace MOPS;

int main() {
    // Initialize GPU backend
    MOPS_Init("gpu");
    
    // Load ocean grid and velocity data
    const char* yaml_path = "/path/to/your/dataset.yaml";
    auto grid = std::make_shared<MPASOGrid>();
    auto solution = std::make_shared<MPASOSolution>();
    
    grid->initGrid(MPASOReader::readGridData(yaml_path).get());
    solution->initSolution(MPASOReader::readSolData(yaml_path, "0015-01-01", 0).get());
    
    // Register grid and solution data
    MOPS_Begin();
    MOPS_AddGridMesh(grid);
    MOPS_AddAttribute(solution->getID(), solution);
    MOPS_End();
    
    // You're ready to trace particles!
    return 0;
}
```

This minimal example sets up MOPS for GPU-accelerated ocean simulation. The following sections show how to build complete applications for visualization and particle tracing.

---

## Installation

### Prerequisites

- **Compiler**: C++17 or later (GCC 9+, Clang 10+)
- **GPU Backend** (choose one):
  - CUDA 11+ (NVIDIA GPUs)
  - Intel oneAPI (Intel GPUs, also works on NVIDIA)
  - ROCm/HIP (AMD GPUs)
  - TBB (CPU fallback, no GPU required)
- **Dependencies**: HDF5, NetCDF, yaml-cpp, VTK (for output)

### Build from Source

```bash
# Clone the repository
cd $PSCRATCH
git clone https://github.com/YosefQiu/MOPS.git
cd MOPS

# Choose your backend and compile
# For CUDA (NVIDIA):
source ./script/compiler_cuda.sh

# For SYCL (Intel/portable):
source ./script/compiler_sycl.sh

# For HIP (AMD):
source ./script/compiler_hip.sh

# For CPU-only (TBB):
source ./script/compiler_tbb.sh
```

The build scripts automatically configure CMake with the appropriate backend flags and compile the library along with Python bindings.

**Verify installation:**

```bash
# Check C++ examples
ls tutorial/pathLine
ls tutorial/reGrid

# Check Python bindings
python3 -c "import sys; sys.path.append('tools/pyMOPS/pyMOPS'); import pyMOPS; print('pyMOPS loaded successfully')"
```

---

## Your First Visualization

Let's create a simple program to visualize ocean velocities at a fixed depth.

### Remapping Ocean Velocity Fields

**What is remapping?** Remapping interpolates unstructured MPAS-Ocean data onto a regular lat/lon grid at a specified depth. This makes it easy to create standard rectangular visualizations.

**C++ Example** (`my_first_remap.cpp`):

```cpp
#include "ggl.h"
#include "api/MOPS.h"
#include "IO/MPASOReader.h"
#include "Core/MPASOVisualizer.h"
#include "Common/ImageBuffer.hpp"

using namespace MOPS;

int main() {
    // Path to your MPAS-Ocean dataset YAML configuration
    const char* yaml_path = "/path/to/your/dataset.yaml";
    
    // Initialize MOPS with GPU backend
    MOPS_Init("gpu");
    
    // Load grid (static mesh structure)
    auto grid = std::make_shared<MPASOGrid>();
    grid->initGrid(MPASOReader::readGridData(yaml_path).get());
    
    // Load solution (velocity field at a specific time)
    auto solution = std::make_shared<MPASOSolution>();
    solution->initSolution(
        MPASOReader::readSolData(yaml_path, "0015-01-01", 0).get()
    );
    
    // Add temperature and salinity attributes (optional)
    solution->addAttribute("temperature", AttributeFormat::kFloat);
    solution->addAttribute("salinity", AttributeFormat::kFloat);
    
    // Register data with MOPS runtime
    MOPS_Begin();
    MOPS_AddGridMesh(grid);
    MOPS_AddAttribute(solution->getID(), solution);
    MOPS_End();
    
    // Activate this solution for visualization
    MOPS_ActiveAttribute(solution->getID());
    auto field = MOPS_GetFieldSnapshots();
    
    // Configure visualization settings
    VisualizationSettings* config = new VisualizationSettings();
    config->imageSize = vec2{3601, 1801};  // Width x Height (0.1 degree resolution)
    config->LatRange = vec2{-90.0, 90.0};  // Full global coverage
    config->LonRange = vec2{-180.0, 180.0};
    config->FixedDepth = 10.0;  // 10 meters depth
    config->TimeStep = 0;
    
    // Create output image buffer (4 channels: East, North, Vertical, Magnitude)
    ImageBuffer<double>* img = new ImageBuffer<double>(3601, 1801);
    
    // Run the remapping on GPU
    std::cout << "Computing velocity field at 10m depth..." << std::endl;
    
    #if defined(MOPS_USE_CUDA)
        GPUContext gpu_ctx = GPUContext::FromCUDA(nullptr);
    #elif defined(MOPS_USE_SYCL)
        sycl::queue q(sycl::default_selector_v);
        GPUContext gpu_ctx = GPUContext::FromSYCL(q);
    #endif
    
    MPASOVisualizer::VisualizeFixedDepth(field.get(), config, img, gpu_ctx);
    
    // Save results to PNG images
    SaveToPNG<double>(*img, "velocity_east.png", 0);    // East component
    SaveToPNG<double>(*img, "velocity_north.png", 1);   // North component
    SaveToPNG<double>(*img, "velocity_vert.png", 2);    // Vertical component
    SaveToPNG<double>(*img, "velocity_magnitude.png", 3);  // Speed
    
    std::cout << "Saved velocity visualizations!" << std::endl;
    std::cout << "  - velocity_east.png (E-W component)" << std::endl;
    std::cout << "  - velocity_north.png (N-S component)" << std::endl;
    std::cout << "  - velocity_vert.png (vertical component)" << std::endl;
    std::cout << "  - velocity_magnitude.png (speed)" << std::endl;
    
    delete img;
    delete config;
    return 0;
}
```

**Compile and run:**

```bash
# Add to CMakeLists.txt or compile directly:
g++ -std=c++17 my_first_remap.cpp -o my_first_remap \
    -I./include -L./build -lMOPS -lhdf5 -lyaml-cpp

./my_first_remap
```

**Python Example** (`my_first_remap.py`):

```python
import sys
sys.path.append("tools/pyMOPS/pyMOPS/")
import pyMOPS
import numpy as np
from pathlib import Path

# Path to your dataset
yaml_path = "/path/to/your/dataset.yaml"

# Initialize and load data
remapper = pyMOPS.MOPSRemapping(yaml_path)
remapper.init(
    device="gpu",
    time_stamp="0015-01-01",
    time_step=0,
    add_temperature=True,
    add_salinity=True
)

# Run remapping
images = remapper.run(
    width=3601,
    height=1801,
    lat_range=(-90.0, 90.0),
    lon_range=(-180.0, 180.0),
    fixed_depth=10.0,  # 10 meters
    time_step=0,
    return_numpy=True
)

# Save results
output_dir = "remap_outputs"
Path(output_dir).mkdir(exist_ok=True)

# Save as colormapped PNG images
remapper.save_colormap_pngs(
    images,
    output_dir,
    prefix="velocity",
    channels=[0, 1, 2, 3],  # East, North, Vertical, Magnitude
    cmap_name="coolwarm",
    save_colorbar=True
)

print(f"Saved visualizations to {output_dir}/")
print("  - velocity_0_ch0.png (East component)")
print("  - velocity_0_ch1.png (North component)")
print("  - velocity_0_ch2.png (Vertical component)")
print("  - velocity_0_ch3.png (Magnitude)")
```

**Understanding the output:**

Each output image is a regular lat/lon grid where pixel intensity represents velocity:
- **Channel 0 (East)**: Positive = eastward flow, Negative = westward flow
- **Channel 1 (North)**: Positive = northward flow, Negative = southward flow
- **Channel 2 (Vertical)**: Positive = upwelling, Negative = downwelling
- **Channel 3 (Magnitude)**: Total speed (always positive)

---

## Basic Particle Tracing

Now let's trace particles through the ocean. MOPS supports two types:

- **Streamlines**: Instantaneous flow paths (single time snapshot)
- **Pathlines**: Time-varying trajectories (multiple time steps)

### Streamline Example

Streamlines show the path particles would follow in a steady flow field. They use a single velocity snapshot.

**C++ Streamline** (`streamline_example.cpp`):

```cpp
#include "ggl.h"
#include "api/MOPS.h"
#include "IO/MPASOReader.h"
#include "IO/VTKFileManager.hpp"

using namespace MOPS;

int main() {
    const char* yaml_path = "/path/to/your/dataset.yaml";
    
    // Initialize
    MOPS_Init("gpu");
    
    // Load data
    auto grid = std::make_shared<MPASOGrid>();
    auto solution = std::make_shared<MPASOSolution>();
    
    grid->initGrid(MPASOReader::readGridData(yaml_path).get());
    solution->initSolution(
        MPASOReader::readSolData(yaml_path, "0015-01-01", 0).get()
    );
    
    solution->addAttribute("temperature", AttributeFormat::kFloat);
    solution->addAttribute("salinity", AttributeFormat::kFloat);
    
    // Register
    MOPS_Begin();
    MOPS_AddGridMesh(grid);
    MOPS_AddAttribute(solution->getID(), solution);
    MOPS_End();
    
    MOPS_ActiveAttribute(solution->getID());
    
    // Define seed points (where particles start)
    SamplingSettings* sampling_conf = new SamplingSettings();
    sampling_conf->setSampleRange(vec2i{50, 50});  // 50x50 grid of particles
    sampling_conf->setGeoBox(
        vec2{-30.0, 30.0},   // Latitude: 30°S to 30°N
        vec2{-90.0, -30.0}   // Longitude: 90°W to 30°W (Atlantic)
    );
    sampling_conf->atCellCenter(false);
    sampling_conf->setDepth(50.0);  // 50 meters depth
    
    std::vector<CartesianCoord> seeds;
    MOPS_GenerateSamplePoints(sampling_conf, seeds);
    
    std::cout << "Generated " << seeds.size() << " seed points" << std::endl;
    
    // Configure streamline integration
    TrajectorySettings* traj_conf = new TrajectorySettings();
    traj_conf->directionType = CalcDirection::kForward;
    traj_conf->methodType = CalcMethodType::kRK4;  // 4th-order Runge-Kutta
    traj_conf->depth = 50.0;
    traj_conf->deltaT = 60;  // 1 minute time step
    traj_conf->simulationDuration = 86400 * 7;  // 7 days
    traj_conf->recordT = 3600;  // Record every hour
    traj_conf->fileName = "atlantic_streamlines";
    
    // Compute streamlines
    std::cout << "Computing streamlines..." << std::endl;
    std::vector<TrajectoryLine> lines = MOPS_RunStreamLine(traj_conf, seeds);
    
    std::cout << "Computed " << lines.size() << " streamlines" << std::endl;
    
    // Save to VTK format (viewable in ParaView)
    VTKFileManager::SaveTrajectoryLinesAsVTP(lines, traj_conf->fileName);
    
    std::cout << "Saved to " << traj_conf->fileName << ".vtp" << std::endl;
    std::cout << "Open with ParaView to visualize!" << std::endl;
    
    delete sampling_conf;
    delete traj_conf;
    return 0;
}
```

**Python Streamline** (`streamline_example.py`):

```python
import sys
sys.path.append("tools/pyMOPS/pyMOPS/")
import pyMOPS
import numpy as np

yaml_path = "/path/to/your/dataset.yaml"

# Initialize streamline tracer
tracer = pyMOPS.MOPSStreamline(yaml_path)
tracer.init(device="gpu")

# Set time (single snapshot for streamlines)
tracer.set_time(
    start="0015-01-01",
    duration_seconds=86400 * 7  # 7 days of integration
)

# Define seed region (Atlantic Ocean)
tracer.set_seed(
    depth=50.0,  # 50 meters
    lat_range=(-30.0, 30.0),
    lon_range=(-90.0, -30.0),
    grid=(50, 50),  # 50x50 seed points
    follow_last=False  # Don't continue from previous run
)

# Compute streamlines
print("Computing streamlines...")
trajectories = tracer.run(
    method="rk4",
    delta_minutes=1,
    record_every_minutes=60
)

print(f"Computed {len(trajectories)} streamlines")

# Access trajectory data
for i, traj in enumerate(trajectories[:3]):  # First 3 trajectories
    points = np.array(traj["points"])  # (N, 3) Cartesian coordinates
    velocity = np.array(traj["velocity"])  # (N, 3)
    temperature = np.array(traj["temperature"])  # (N,)
    salinity = np.array(traj["salinity"])  # (N,)
    
    print(f"Trajectory {i}: {len(points)} points")
    print(f"  Temperature range: {temperature.min():.2f} - {temperature.max():.2f} °C")
    print(f"  Salinity range: {salinity.min():.2f} - {salinity.max():.2f} PSU")
```

### Pathline Example

Pathlines track particles through time-varying velocity fields. This is more realistic for ocean simulations.

**C++ Pathline** (`pathline_example.cpp`):

```cpp
#include "ggl.h"
#include "api/MOPS.h"
#include "IO/MPASOReader.h"
#include "IO/VTKFileManager.hpp"
#include "Utils/Utils.hpp"

using namespace MOPS;

// Convert lat/lon/depth to Cartesian coordinates
CartesianCoord lat_lon_depth_to_xyz(double lat_deg, double lon_deg, double depth) {
    const double EARTH_RADIUS = 6371000.0;  // meters
    double r = EARTH_RADIUS - depth;
    double lat = lat_deg * M_PI / 180.0;
    double lon = lon_deg * M_PI / 180.0;
    
    CartesianCoord out;
    out.x() = r * std::cos(lat) * std::cos(lon);
    out.y() = r * std::cos(lat) * std::sin(lon);
    out.z() = r * std::sin(lat);
    return out;
}

int main() {
    const char* yaml_path = "/path/to/your/dataset.yaml";
    
    // Initialize
    MOPS_Init("gpu");
    
    auto grid = std::make_shared<MPASOGrid>();
    grid->initGrid(MPASOReader::readGridData(yaml_path).get());
    
    // We'll trace particles through 3 months: Jan -> Feb -> Mar 2015
    auto month_pairs = MOPS_IO::make_forward_month_pairs(15, 1, 15, 3);
    
    std::vector<CartesianCoord> current_positions;
    bool is_first = true;
    
    // Create initial particle positions (Gulf of Mexico)
    std::vector<CartesianCoord> initial_seeds = {
        lat_lon_depth_to_xyz(25.0, -90.0, 10.0),   // Near Mississippi Delta
        lat_lon_depth_to_xyz(26.0, -88.0, 10.0),
        lat_lon_depth_to_xyz(24.0, -85.0, 10.0),
        lat_lon_depth_to_xyz(23.0, -92.0, 10.0)
    };
    
    std::cout << "Tracing " << initial_seeds.size() << " particles through time..." << std::endl;
    
    for (const auto& [start_month, end_month] : month_pairs) {
        std::cout << "Processing: " << start_month << " -> " << end_month << std::endl;
        
        // Load two consecutive monthly snapshots
        auto sol_front = std::make_shared<MPASOSolution>();
        auto sol_back = std::make_shared<MPASOSolution>();
        
        sol_front->initSolution(
            MPASOReader::readSolData(yaml_path, start_month, 0).get()
        );
        sol_back->initSolution(
            MPASOReader::readSolData(yaml_path, end_month, 0).get()
        );
        
        sol_front->addAttribute("temperature", AttributeFormat::kFloat);
        sol_front->addAttribute("salinity", AttributeFormat::kFloat);
        sol_back->addAttribute("temperature", AttributeFormat::kFloat);
        sol_back->addAttribute("salinity", AttributeFormat::kFloat);
        
        // Register both snapshots (MOPS will interpolate between them)
        MOPS_Begin();
        MOPS_AddGridMesh(grid);
        MOPS_AddAttribute(sol_front->getID(), sol_front);
        MOPS_AddAttribute(sol_back->getID(), sol_back);
        MOPS_End();
        
        MOPS_ActiveAttribute(sol_front->getID(), sol_back->getID());
        
        // Use initial seeds on first iteration, then continue from last positions
        if (is_first) {
            current_positions = initial_seeds;
            is_first = false;
        }
        
        // Configure trajectory settings
        TrajectorySettings* traj_conf = new TrajectorySettings();
        traj_conf->directionType = CalcDirection::kForward;
        traj_conf->methodType = CalcMethodType::kRK4;
        traj_conf->depth = 10.0;  // 10 meters
        traj_conf->deltaT = 60 * 10;  // 10 minute time step
        traj_conf->recordT = 3600 * 6;  // Record every 6 hours
        
        // Calculate time gap between snapshots
        auto t1 = sol_front->getTimeStamp();
        auto t2 = sol_back->getTimeStamp();
        traj_conf->simulationDuration = std::abs(
            getTimeGapinSecond(t2.c_str(), t1.c_str())
        );
        
        std::string filename = "pathline_" + std::string(start_month);
        traj_conf->fileName = filename;
        
        // Run pathline segment
        std::vector<TrajectoryLine> lines = MOPS_RunPathLine(
            traj_conf, current_positions
        );
        
        // Save this segment
        VTKFileManager::SaveTrajectoryLinesAsVTP(lines, filename);
        
        // Update positions to last valid point of each trajectory
        current_positions.clear();
        for (const auto& line : lines) {
            const auto& pts = line.points;
            // Find last non-zero point
            for (int i = pts.size() - 1; i >= 0; --i) {
                if (!(pts[i].x() == 0.0 && pts[i].y() == 0.0 && pts[i].z() == 0.0)) {
                    current_positions.push_back(pts[i]);
                    break;
                }
            }
        }
        
        std::cout << "  Saved " << filename << ".vtp (" 
                  << current_positions.size() << " particles continue)" << std::endl;
        
        delete traj_conf;
    }
    
    std::cout << "Pathline computation complete!" << std::endl;
    return 0;
}
```

**Python Pathline** (`pathline_example.py`):

```python
import sys
sys.path.append("tools/pyMOPS/pyMOPS/")
import pyMOPS
import numpy as np

def lat_lon_depth_to_xyz(lat_deg, lon_deg, depth, R=6371000.0):
    """Convert lat/lon/depth to Cartesian coordinates"""
    r = R - depth
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z], dtype=float)

yaml_path = "/path/to/your/dataset.yaml"

# Initialize pathline tracer
tracer = pyMOPS.MOPSPathline(yaml_path)
tracer.init(device="gpu")

# Set time range: January 2015 to March 2015 (forward in time)
tracer.set_time(
    sy=15, sm=1,  # Start: year 15, month 1
    ey=15, em=3,  # End: year 15, month 3
    direction="forward"
)

# Define initial particle positions (Gulf of Mexico)
initial_particles = np.array([
    lat_lon_depth_to_xyz(25.0, -90.0, 10.0),  # Near Mississippi Delta
    lat_lon_depth_to_xyz(26.0, -88.0, 10.0),
    lat_lon_depth_to_xyz(24.0, -85.0, 10.0),
    lat_lon_depth_to_xyz(23.0, -92.0, 10.0)
])

# Set seed points
tracer.set_seed(
    depth=10.0,
    points=initial_particles,
    follow_last=True  # Continue from last position each month
)

# Run pathline computation
print(f"Tracing {len(initial_particles)} particles through time...")
trajectories = tracer.run(
    method="rk4",
    delta_minutes=10,
    record_every_minutes=360  # 6 hours
)

print(f"Computed {len(trajectories)} pathlines")

# Analyze results
for i, traj in enumerate(trajectories):
    points = np.array(traj["points"])
    velocity = np.array(traj["velocity"])
    temperature = np.array(traj["temperature"])
    
    # Calculate total distance traveled
    if len(points) > 1:
        displacements = np.diff(points, axis=0)
        distances = np.linalg.norm(displacements, axis=1)
        total_distance = np.sum(distances)
    else:
        total_distance = 0.0
    
    print(f"Particle {i}:")
    print(f"  Total points: {len(points)}")
    print(f"  Distance traveled: {total_distance/1000:.2f} km")
    print(f"  Temperature range: {temperature.min():.2f} - {temperature.max():.2f} °C")
    print(f"  Final position: {points[-1]}")
```

**Key differences: Streamline vs Pathline**

| Feature | Streamline | Pathline |
|---------|-----------|----------|
| **Time** | Single snapshot | Multiple time steps |
| **Realism** | Instantaneous flow | Time-varying flow |
| **Use case** | Quick visualization | Realistic tracking |
| **API** | `MOPS_RunStreamLine` | `MOPS_RunPathLine` |
| **Data** | One solution | Two solutions (interpolated) |

---

## Working with Different Depths

MOPS supports both uniform depth (all particles at same depth) and per-particle depth (each particle has its own depth).

### Uniform Depth (Simple)

```cpp
TrajectorySettings* conf = new TrajectorySettings();
conf->depth = 100.0;  // All particles at 100 meters
```

```python
tracer.set_seed(
    depth=100.0,  # All particles at 100m
    lat_range=(-30, 30),
    lon_range=(-90, -30),
    grid=(50, 50)
)
```

### Per-Particle Depth (Advanced)

```cpp
// Create particles at different depths
std::vector<CartesianCoord> seeds = { /* ... */ };
std::vector<float> depths = {10.0, 50.0, 100.0, 200.0};  // One per particle

TrajectorySettings* conf = new TrajectorySettings();
conf->depth = depths[0];  // Fallback depth
conf->particle_depths = depths;  // Per-particle depths
```

```python
# Define particles with individual depths
particles = np.array([
    lat_lon_depth_to_xyz(25.0, -90.0, 10.0),
    lat_lon_depth_to_xyz(25.0, -90.0, 50.0),
    lat_lon_depth_to_xyz(25.0, -90.0, 100.0),
    lat_lon_depth_to_xyz(25.0, -90.0, 200.0)
])

depths = np.array([10.0, 50.0, 100.0, 200.0])  # Matches particles

tracer.set_seed(
    depths=depths,  # Array of depths (one per particle)
    points=particles
)
```

---

## Cross-Section Visualization

MOPS can visualize velocity fields along fixed latitude cross-sections (longitude vs depth).

**C++ Example** (`cross_section.cpp`):

```cpp
#include "ggl.h"
#include "api/MOPS.h"
#include "IO/MPASOReader.h"
#include "Core/MPASOVisualizer.h"
#include "Common/ImageBuffer.hpp"

using namespace MOPS;

int main() {
    const char* yaml_path = "/path/to/your/dataset.yaml";
    
    MOPS_Init("gpu");
    
    auto grid = std::make_shared<MPASOGrid>();
    auto solution = std::make_shared<MPASOSolution>();
    
    grid->initGrid(MPASOReader::readGridData(yaml_path).get());
    solution->initSolution(
        MPASOReader::readSolData(yaml_path, "0015-01-01", 0).get()
    );
    
    MOPS_Begin();
    MOPS_AddGridMesh(grid);
    MOPS_AddAttribute(solution->getID(), solution);
    MOPS_End();
    
    MOPS_ActiveAttribute(solution->getID());
    auto field = MOPS_GetFieldSnapshots();
    
    // Configure cross-section at 45°N latitude
    int width = 720;  // Longitude bins
    int height = 100;  // Depth bins
    
    VisualizationSettings* config = new VisualizationSettings();
    config->imageSize = vec2{static_cast<double>(width), static_cast<double>(height)};
    config->LonRange = vec2{-180.0, 180.0};
    config->DepthRange = vec2{0.0, 5000.0};  // 0-5000m depth
    config->FixedLatitude = 45.0;  // 45°N
    
    ImageBuffer<double>* img = new ImageBuffer<double>(width, height);
    
    #if defined(MOPS_USE_CUDA)
        GPUContext gpu_ctx = GPUContext::FromCUDA(nullptr);
    #elif defined(MOPS_USE_SYCL)
        sycl::queue q(sycl::default_selector_v);
        GPUContext gpu_ctx = GPUContext::FromSYCL(q);
    #endif
    
    std::cout << "Computing cross-section at 45°N..." << std::endl;
    MPASOVisualizer::VisualizeFixedLatitude(field.get(), config, img, gpu_ctx);
    
    // Save velocity components
    SaveToPNG<double>(*img, "cross_section_east.png", 0);
    SaveToPNG<double>(*img, "cross_section_north.png", 1);
    SaveToPNG<double>(*img, "cross_section_vertical.png", 2);
    SaveToPNG<double>(*img, "cross_section_magnitude.png", 3);
    
    std::cout << "Saved cross-section images!" << std::endl;
    
    delete img;
    delete config;
    return 0;
}
```

**Python Example** (`cross_section.py`):

```python
import sys
sys.path.append("tools/pyMOPS/pyMOPS/")
import pyMOPS

yaml_path = "/path/to/your/dataset.yaml"

# Initialize regridder
regridder = pyMOPS.MOPSReGrid(yaml_path)
regridder.init(
    device="gpu",
    time_stamp="0015-01-01",
    time_step=0
)

# Run cross-section at 45°N
image = regridder.run(
    width=720,  # Longitude resolution
    height=100,  # Depth resolution
    lon_range=(-180.0, 180.0),
    depth_range=(0.0, 5000.0),  # 0-5000m
    fixed_latitude=45.0,  # 45°N
    time_step=0
)

# Save as PNG
regridder.save_to_png(
    image,
    "cross_section_45N.png",
    channel=3,  # Velocity magnitude
    cmap_name="viridis"
)

print("Saved cross-section to cross_section_45N.png")
```

---

## Next Steps

Congratulations! You now know the basics of MOPS. Here's where to go next:

### Learn More

- **[Core Components](INDEX.md#core-components)**: Deep dive into MPASOGrid, MPASOSolution, and MPASOField
- **[GPU Backends](CUDA_IMPLEMENTATION.md)**: Understand CUDA, SYCL, and HIP implementations
- **[Algorithms](INDEX.md#algorithms)**: Learn about Wachspress interpolation and RK4 integration
- **[Performance Tuning](INDEX.md#advanced-topics)**: Optimize for your specific use case

### Try These Examples

Browse the `tutorial/` directory for complete working examples:

```bash
ls tutorial/
# pathLine.cpp - Multi-month pathline computation
# reGrid.cpp - Cross-section visualization
# pyMOPSAPI.py - Python API reference implementation
# reGrid.py - Python regridding examples
```

### Common Workflows

**Ocean current visualization:**
1. Use remapping to create global velocity maps at different depths
2. Compare seasonal variations by loading different time stamps

**Particle dispersion studies:**
1. Define seed region (e.g., oil spill location)
2. Run pathlines forward in time
3. Analyze spread and transport pathways

**Larval transport modeling:**
1. Use per-particle depths for vertical migration
2. Track temperature/salinity experienced by each particle
3. Identify recruitment regions

### Getting Help

- **Examples**: All code in this guide is compilable and tested
- **Tutorial folder**: See `tutorial/pathLine.cpp` and `tutorial/pyMOPSAPI.py` for full implementations
- **Documentation**: Check [INDEX.md](INDEX.md) for topic-specific guides
- **Issues**: Report bugs or ask questions at the repository

---

## Troubleshooting

### Common Issues

**Build fails with CUDA errors:**
```bash
# Make sure CUDA is in your PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source ./script/compiler_cuda.sh
```

**Python can't find pyMOPS:**
```python
import sys
sys.path.append("tools/pyMOPS/pyMOPS/")  # Adjust path to your build
import pyMOPS
```

**GPU out of memory:**
- Reduce image resolution: `config->imageSize = vec2{1801, 901};`
- Process fewer particles at once
- Use CPU backend: `MOPS_Init("cpu");`

**Particles stop/disappear:**
- Check if particles left the domain
- Verify depth is within ocean floor bounds
- Inspect for zero velocity regions
- Look for invalid (0,0,0) coordinates in output

**VTK files won't open in ParaView:**
- Make sure VTK library was found during build
- Check file exists and has `.vtp` extension
- Use absolute paths when saving files

---

## API Summary

### Initialization
```cpp
MOPS_Init("gpu");  // or "cpu"
```

### Data Loading
```cpp
auto grid = std::make_shared<MPASOGrid>();
auto solution = std::make_shared<MPASOSolution>();
grid->initGrid(MPASOReader::readGridData(yaml_path).get());
solution->initSolution(MPASOReader::readSolData(yaml_path, "YYYY-MM-DD", 0).get());
```

### Registration
```cpp
MOPS_Begin();
MOPS_AddGridMesh(grid);
MOPS_AddAttribute(solution->getID(), solution);
MOPS_End();
MOPS_ActiveAttribute(solution_id);
```

### Visualization
```cpp
// Fixed depth
MPASOVisualizer::VisualizeFixedDepth(field, config, image, gpu_ctx);

// Fixed latitude cross-section
MPASOVisualizer::VisualizeFixedLatitude(field, config, image, gpu_ctx);
```

### Particle Tracing
```cpp
// Streamline (single time)
std::vector<TrajectoryLine> lines = MOPS_RunStreamLine(config, seeds);

// Pathline (time-varying)
std::vector<TrajectoryLine> lines = MOPS_RunPathLine(config, seeds);
```

### Python Classes
```python
pyMOPS.MOPSRemapping      # Velocity field remapping
pyMOPS.MOPSReGrid         # Cross-section visualization
pyMOPS.MOPSStreamline     # Streamline tracing
pyMOPS.MOPSPathline       # Pathline tracing
```

---

**Happy particle tracing!** 🌊
