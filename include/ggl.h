#pragma once
//c++ header file
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <stack>
#include <queue>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <iomanip>
#include <chrono>
#include <limits>
#include <numbers>
#include <cmath>
#include <filesystem>
#include <cstddef>
#include <regex>
#include <ctime>
#include <optional>



//c header file
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <atomic>

#define LOADING_NETCDF_CXX      1

#ifndef MOPS_VTK
#define MOPS_VTK 0
#endif

#if MOPS_VTK
    //VTK Readers/Writers
    #include <vtkUnstructuredGridReader.h>
    #include <vtkXMLUnstructuredGridReader.h>
    #include <vtkXMLPolyDataReader.h>
    #include <vtkXMLImageDataWriter.h>
    #include <vtkXMLPolyDataWriter.h>

    //VTK Data Structures and Sources
    #include <vtkAppendFilter.h>
    #include <vtkSphereSource.h>
    #include <vtkUnstructuredGrid.h>
    #include <vtkImageData.h>
    #include <vtkPoints.h>
    #include <vtkPointData.h>
    #include <vtkPolyLine.h>
    #include <vtkCellArray.h>
    #include <vtkTetra.h>
    #include <vtkLine.h>

    //VTK Rendering and Visualization
    #include <vtkAppendPolyData.h>
    #include <vtkActor.h>
    #include <vtkCamera.h>
    #include <vtkDataSetMapper.h>
    #include <vtkNamedColors.h>
    #include <vtkNew.h>
    #include <vtkProperty.h>
    #include <vtkDoubleArray.h>
    #include <vtkRenderWindow.h>
    #include <vtkRenderWindowInteractor.h>
    #include <vtkRenderer.h>
    #include <vtkSmartPointer.h>
#endif


#if LOADING_NETCDF_CXX == 1
    #include "netcdf.h"
#endif

#include "Utils/BackendCompat.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif 


