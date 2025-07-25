<img src="./icon.svg" alt="MOPS Logo" width="300">

<a href="./LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>

## MOPS (MPAS-O-Particles-SYCL)
### MOPS (MPAS-Ocean Particle SYCL) is a command-line tool for simulating MPAS-Ocean trajectories, supporting visualization, particle sampling, and trajectory calculation.

## [📦 Installation](#-installation) | [💻 Command Line Interface](#-command-line-interface) | [📌 Examples](#-examples) | [⚙️ Default Parameter Values](#-default-parameter-values) | [📝 Notes](#-notes) | [📩 Contact](#-contact)



## 📦 Installation
### **1️⃣ Use CMake**
```bash
cd $PSCRATCH
git clone https://github.com/YosefQiu/MOPS.git
cd MOPS
./script/compile.sh
```
### **2️⃣ Use [Spack](https://github.com/spack/spack)** (will be update use a new repo)
> ### 🔗  check [spack_test](https://github.com/YosefQiu/spack_test) for more details.

## 💻 Command Line Interface

### **Usage**

```sh
./MOPS [OPTIONS]
```

### **Available Commands**
| **Command**  | **Description** |
|-------------|----------------|
| `MOPS`      | Parses input YAML and simulates MPAS-Ocean particle trajectories. |

### **CLI Options**
| **Option** | **Description** |
|------------|----------------|
| `-i, --input <file>` | **(Required)** Path to the input YAML configuration file. |
| `-p, --prefix <path>` | Data path prefix for additional resources. |
| `--imagesize <width> <height>` | Image size in pixels (`width` and `height`). Default: `360 x 180` |
| `--longitude <min> <max>` | Longitude range (`min` and `max`). Default: `-180 180` |
| `--latitude <min> <max>` | Latitude range (`min` and `max`). Default: `-90 90` |
| `--layer <layer> ` | Ocean layer. Default: `10` |
| `--depth <meter>` | Ocean depth in meter. Default: `800` |
| `--deltat <seconds>` | Time step (ΔT) in seconds. Default: `120` |
| `--checkt <seconds>` | Check interval in seconds. Default: `60` |
| `--trajectoryt <seconds>` | Total trajectory time in seconds. Default: `86400 (1 day)` |
| `--samplerange <lon_min> <lon_max> <lat_min> <lat_max>` | Sample range for longitude and latitude. Default: `-180 180 -90 90` |
| `--samplenumber <n>` | Number of samples to generate. Default: `100` |
| `--sampletype <type>` | Sampling type:<br> `uniform` - Uniform sampling <br> `gaussian` - Gaussian sampling <br> Default: `uniform` |
| `--visualizetype <type>` | Visualization type:<br> `remap` - Re-mapping visualization <br> `trajectory` - Trajectory visualization <br> Default: `remap` |
| `-h, --help` | Display this help message. |


## 📌 Examples
### 1️⃣ **Help infomation**
```bash
./MOPS -h
```
```bash
MPAS-Ocean Particle SYCL(MPOS) Command Line Parser
Usage:
  ./MOPS [OPTION...]

  -i, --input arg          Input YAML file (Required)
  -p, --prefix arg         Data path prefix
      --imagesize arg      Image Size (width height)
      --longitude arg      Longitude Range (min max)
      --latitude arg       Latitude Range (min max)
      --layer              Fixed Layer
      --depth              Fixed Depth
      --deltat arg         Delta T
      --checkt arg         Check T
      --trajectoryt arg    Trajectory T
      --samplerange arg    Sample Range (longitude_min longitude_max
                           latitude_min latitude_max)
      --samplenumber arg   Sample Number
      --sampletype arg     Sample Type (uniform or gaussian)
      --visualizetype arg  Visualize Type (remap or trajectory)
  -h, --help               Print help message
```
---

### 2️⃣ **Basic Execution**
```sh
./MOPS -i config.yaml
```
Uses default values for all parameters except the required input YAML file.

---
### 3️⃣ **Custom Image Size & Region**
```sh
./MOPS -i config.yaml --imagesize 1920 1080 --longitude -170 170 --latitude -80 80
```
Sets image size to `1920x1080` (width x height), longitude range to `-170 170`(170W -> 170E), and latitude range to `-80 80` (80S -> 80N).

---
### 4️⃣ **Custom Sampling & Visualization**
```sh
./MOPS -i config.yaml --samplenumber 500 --sampletype gaussian --visualizetype trajectory
```
Sets the sample number to `500`, uses Gaussian sampling, and selects trajectory visualization.

---

## ⚙️ Default Parameter Values
| **Parameter** | **Default Value** |
|--------------|------------------|
| `imagesize` | `360 x 180` |
| `longitude` | `-180 180` |
| `latitude` | `-90 90` |
| `layer` | `10` |
| `depth` | `800` |
| `deltat` | `120` |
| `checkt` | `60` |
| `trajectoryt` | `86400` |
| `samplerange` | `-180 180 -90 90` |
| `samplenumber` | `100` |
| `sampletype` | `uniform` |
| `visualizetype` | `remap` |

## 📝 Notes
- The `-i, --input` argument is **mandatory**; all other arguments are optional and default values will be used if not provided.
- Ensure that the input YAML file contains a valid MPAS-Ocean particle configuration.



## 📩 Contact
For questions or contributions, please reach out via **[qiu.722@osu.edu]**.




