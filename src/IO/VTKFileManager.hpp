#pragma once
#include "ggl.h"
#include "SYCL/ImageBuffer.hpp"
#include "Core/MPASOVisualizer.h"
#include "Utils/Utils.hpp"
#include "Utils/GeoConverter.hpp"
#include "Utils/Timer.hpp"

#if MOPS_VTK
namespace MOPS
{
	inline std::string TypeToStr(VisualizeType type)
	{
		switch (type)
		{
		case VisualizeType::kFixedLayer: return "fixed_layer_"; break;
		case VisualizeType::kFixedDepth: return "fixed_depth_"; break;
		}
	}

	class VTKFileManager
	{
	public:
		static void SaveVTI(ImageBuffer<double>* img, VisualizationSettings* config, std::string outputName = "output")
		{
			auto width = config->imageSize.x();
			auto height = config->imageSize.y();
			double latSpacing = (config->LatRange.y() - config->LatRange.x()) / (height - 1);
			double lonSpacing = (config->LonRange.y() - config->LonRange.x()) / (width - 1);

			double k = -1;
			switch (config->VisType)
			{
			case VisualizeType::kFixedDepth: k = config->FixedDepth; break;
			case VisualizeType::kFixedLayer: k = config->FixedLayer; break;
			default:
				Debug("[ERROR]::SaveVTI:: Unknow VisualizeType....."); return;
			}


			// Create ImageData object for VTK.
			vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
			imageData->SetDimensions(width, height, 1);
			imageData->AllocateScalars(VTK_DOUBLE, 3); // Each point has three attributes (X, Y, Z)
			imageData->SetOrigin(config->LonRange.x(), config->LatRange.x(), k);  // Set the origin of the data
			imageData->SetSpacing(lonSpacing, latSpacing, k);  // Set the physical size of each pixel

			// Fill data
			if (!img)
			{
				Debug("[ERROR]::SaveVTI:: ImageBuffer is nullptr....."); return;
			}

			for (int i = 0; i < height; ++i)
			{
				for (int j = 0; j < width; ++j)
				{
					if (height - 1 - i < 0 || height - 1 - i >= img->getHeight() || j < 0 || j >= img->getWidth()) 
					{
						Debug("[ERROR]::SaveVTI:: Pixel out of bounds at (%d, %d).", height - 1 - i, j);
						continue;
					}
					vec3 value = img->getPixel(height - 1 - i, j);
					double* ptr = static_cast<double*>(imageData->GetScalarPointer(j, i, 0));
					ptr[0] = value.x(); // X
					ptr[1] = value.y(); // Y
					ptr[2] = value.z(); // Z
				}
			}

			// Set writer and save file
			vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
			writer->SetFileName((outputName + "_" + TypeToStr(config->VisType) + std::to_string(k) + ".vti").c_str());
			writer->SetInputData(imageData);
			writer->Write();
			Debug("[VTKFileManager]::Finished... %s", (TypeToStr(config->VisType) + std::to_string(k) + ".vti").c_str());
		}

		static void SaveVTI(const std::vector<ImageBuffer<double>>& img_list,
                    VisualizationSettings* config,
                    const std::vector<std::string>& attribute_names,
                    const std::string& outputName = "output") 
		{

			auto width = config->imageSize.x();
			auto height = config->imageSize.y();
			double latSpacing = (config->LatRange.y() - config->LatRange.x()) / (height - 1);
			double lonSpacing = (config->LonRange.y() - config->LonRange.x()) / (width - 1);
			double k = (config->VisType == VisualizeType::kFixedDepth) ? config->FixedDepth : config->FixedLayer;

			vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
			imageData->SetDimensions(width, height, 1);
			imageData->SetOrigin(config->LonRange.x(), config->LatRange.x(), k);
			imageData->SetSpacing(lonSpacing, latSpacing, 1.0); // Avoid collapse

			int total_attr_count = img_list.size() * 3;
			if (attribute_names.size() < total_attr_count) {
				Debug("[WARNING]::Not enough attribute names (%zu < %d), auto-filling", attribute_names.size(), total_attr_count);
			}

			std::vector<vtkSmartPointer<vtkDoubleArray>> arrays(total_attr_count);

			for (int i = 0; i < total_attr_count; ++i) {
				arrays[i] = vtkSmartPointer<vtkDoubleArray>::New();
				std::string name = (i < attribute_names.size()) ? attribute_names[i] : "attr_" + std::to_string(i);
				arrays[i]->SetName(name.c_str());
				arrays[i]->SetNumberOfComponents(1);
				arrays[i]->SetNumberOfTuples(width * height);
			}

			// Fill data
			for (size_t img_idx = 0; img_idx < img_list.size(); ++img_idx) {
				const auto& img = img_list[img_idx];
				for (int i = 0; i < height; ++i) {
					for (int j = 0; j < width; ++j) {
						int row = height - 1 - i;
						vec3 val = img.getPixel(row, j);
						vtkIdType id = i * width + j;
						arrays[img_idx * 3 + 0]->SetValue(id, val.x());
						arrays[img_idx * 3 + 1]->SetValue(id, val.y());
						arrays[img_idx * 3 + 2]->SetValue(id, val.z());
					}
				}
			}

			for (auto& array : arrays) {
				imageData->GetPointData()->AddArray(array);
			}

			// Write to file
			vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
			std::string filename = (outputName + "_" + TypeToStr(config->VisType) + std::to_string(k) + ".vti");
			writer->SetFileName(filename.c_str());
			writer->SetInputData(imageData);
			writer->Write();
			Debug("[VTKFileManager]::Saved multi-attribute VTI to %s", filename.c_str());
		}
		
		static void SavePointAsVTP(std::vector<CartesianCoord>& points, std::string outpuName = "output")
		{
			vtkSmartPointer<vtkPoints> point_ptr = vtkSmartPointer<vtkPoints>::New();
			for (auto& val : points)
			{
				point_ptr->InsertNextPoint(val.x(), val.y(), val.z());
			}

			vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
			polydata->SetPoints(point_ptr);

			vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
			writer->SetFileName(checkAndModifyExtension(outpuName, "vtp").c_str());
			writer->SetInputData(polydata);
			writer->Write();
		}

		static void LineCheck(const std::vector<vtkSmartPointer<vtkPolyData>>& polyDataList, std::string outputName)
		{
			int numFiles = polyDataList.size();
			if (numFiles == 0) {
				std::cerr << "No polydata files provided." << std::endl;
				return;
			}

			vtkSmartPointer<vtkPoints> new_points = vtkSmartPointer<vtkPoints>::New();
			vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
			vtkSmartPointer<vtkPolyLine> line = vtkSmartPointer<vtkPolyLine>::New();
			int pointCount = 0;
			bool firstPoint = true;
			double previousLongitude = 0.0;

			// Iterate over points in each file and connect them into a line
			for (int i = 0; i < numFiles; ++i)
			{
				vtkPolyData* pd = polyDataList[i];
				if (pd->GetNumberOfPoints() > 0)
				{
					double point[3];
					pd->GetPoint(0, point);
					double longitude = point[0]; // Assuming x coordinate represents longitude

					// Check for longitude wraparound from -180 to 180 or 180 to -180
					if (!firstPoint) {
						if ((previousLongitude < -170 && longitude > 170) || (previousLongitude > 170 && longitude < -170)) {
							// Add the current line to the lines array and start a new line
							lines->InsertNextCell(line);
							line = vtkSmartPointer<vtkPolyLine>::New();
							pointCount = 0;
						}
					}

					// Add point to new_points and line
					vtkIdType pid = new_points->InsertNextPoint(point);
					line->GetPointIds()->InsertNextId(pid);
					pointCount++;

					// Update previousLongitude
					previousLongitude = longitude;
					firstPoint = false;
				}
			}

			// Add the last line
			if (pointCount > 0) {
				lines->InsertNextCell(line);
			}

			//Create a PolyData object and set the points and lines.
			vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
			polydata->SetPoints(new_points);
			polydata->SetLines(lines);

			//Write PolyData to file
			vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
			writer->SetFileName((outputName + ".vtp").c_str());
			writer->SetInputData(polydata);
			writer->Write();
			Debug("[VTKFileManager]::Finished....  [ %s ]", (outputName + ".vtp").c_str());
		}

		static void ConnectPointsToOneLine(const std::vector<vtkSmartPointer<vtkPolyData>>& polyDataList, const std::string& outputFileName = "output.vtp") {

			int numFiles = polyDataList.size();
			if (numFiles == 0) {
				std::cerr << "No polydata files provided." << std::endl;
				return;
			}

			vtkSmartPointer<vtkPoints> new_points = vtkSmartPointer<vtkPoints>::New();
			vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
			vtkSmartPointer<vtkPolyLine> line = vtkSmartPointer<vtkPolyLine>::New();
			int pointCount = 0;
			bool firstPoint = true;
			double previousLongitude = 0.0;

			//Iterate through the points in each file, connecting them to a line
			for (int i = 0; i < numFiles; ++i) {
				vtkPolyData* pd = polyDataList[i];
				if (pd->GetNumberOfPoints() > 0) {
					double point[3];
					pd->GetPoint(0, point);
					double longitude = point[0]; // Assuming x coordinate represents longitude

					// Check for longitude wraparound from -180 to 180 or 180 to -180
					if (!firstPoint) {
						if ((previousLongitude < -170 && longitude > 170) || (previousLongitude > 170 && longitude < -170)) {
							// Add the current line to the lines array and start a new line
							lines->InsertNextCell(line);
							line = vtkSmartPointer<vtkPolyLine>::New();
							pointCount = 0;
						}
					}

					// Add point to new_points and line
					vtkIdType pid = new_points->InsertNextPoint(point);
					line->GetPointIds()->InsertNextId(pid);
					pointCount++;

					// Update previousLongitude
					previousLongitude = longitude;
					firstPoint = false;
				}
			}

			// Add the last line
			if (pointCount > 0) {
				lines->InsertNextCell(line);
			}

		
			vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
			polydata->SetPoints(new_points);
			polydata->SetLines(lines);

			vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
			writer->SetFileName(checkAndModifyExtension(outputFileName, "vtp").c_str());
			writer->SetInputData(polydata);
			writer->Write();

			//Debug("Finished writing %s", outputFileName.c_str());
		}

		static void MergeVTPFiles(const std::vector<std::string>& fileNames, const std::string& outputFileName) {
			vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();

			for (const auto& fileName : fileNames) {
				vtkSmartPointer<vtkXMLPolyDataReader> reader = vtkSmartPointer<vtkXMLPolyDataReader>::New();
				reader->SetFileName(fileName.c_str());
				reader->Update();

				appendFilter->AddInputData(reader->GetOutput());
				// Deleting read-in files
				std::filesystem::remove(fileName);
			}

			appendFilter->Update();

			vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
			writer->SetFileName(checkAndModifyExtension(outputFileName, "vtp").c_str());
			writer->SetInputData(appendFilter->GetOutput());
			writer->Write();
			Debug("[VTKFileManager]::Finished... %s", outputFileName.c_str());
		}

		static std::string checkAndModifyExtension(const std::string& filename, const std::string& ext) 
		{
			if (filename.size() >= ext.size() &&
				filename.substr(filename.size() - ext.size()) == ext)
				return filename;
			else
				return filename + "." + ext;
		}

		
		static void SaveTrajectoryLinesAsVTP(const std::vector<TrajectoryLine>& lines, const std::string& outputFileName) 
		{
			MOPS_TIMER_START("IO::SaveVTP", TimerCategory::IO_Write);
			
			vtkSmartPointer<vtkPoints> all_points = vtkSmartPointer<vtkPoints>::New();
			vtkSmartPointer<vtkCellArray> all_lines = vtkSmartPointer<vtkCellArray>::New();

			auto tempArr   = vtkSmartPointer<vtkDoubleArray>::New();
			tempArr->SetName("temperature");
			tempArr->SetNumberOfComponents(1);

			auto salArr    = vtkSmartPointer<vtkDoubleArray>::New();
			salArr->SetName("salinity");
			salArr->SetNumberOfComponents(1);

			auto velMagArr = vtkSmartPointer<vtkDoubleArray>::New();
			velMagArr->SetName("velocity_mag");
			velMagArr->SetNumberOfComponents(1);

			const double double_NaN = std::numeric_limits<double>::quiet_NaN();

			for (const auto& traj : lines) 
			{
				if (traj.points.empty()) continue;

				vtkSmartPointer<vtkPolyLine> polyline = vtkSmartPointer<vtkPolyLine>::New();
				bool firstPoint = true;
				double previousLongitude = 0.0;

				const size_t nPts  = traj.points.size();
				const size_t nTmp  = traj.temperature.size();
				const size_t nSal  = traj.salinity.size();
				const size_t nVel  = traj.velocity.size();

				auto getTemp = [&](size_t i){ return (i < nTmp) ? traj.temperature[i] : double_NaN; };
        		auto getSal  = [&](size_t i){ return (i < nSal) ? traj.salinity[i]   : double_NaN; };


				for (size_t idx = 0; idx < nPts; ++idx) 
				{
					vec2 latlon;
					vec3 p_copy = traj.points[idx];
					GeoConverter::convertXYZToLatLonDegree(p_copy, latlon);

					double longitude = latlon.y();  // longitudes
					double latitude = latlon.x();   // latitude
					double altitude = traj.depth;

					// Wraparound check
					if (!firstPoint) {
						if ((previousLongitude < -170 && longitude > 170) || (previousLongitude > 170 && longitude < -170)) 
						{
							// Longitude wraparound detected! Need to break the line and start a new polyline
							all_lines->InsertNextCell(polyline);
							polyline = vtkSmartPointer<vtkPolyLine>::New();
						}
					}

					vtkIdType pid = all_points->InsertNextPoint(longitude, latitude, altitude);
					polyline->GetPointIds()->InsertNextId(pid);

					tempArr->InsertNextValue(getTemp(idx));
					salArr->InsertNextValue(getSal(idx));

					double vx = 0.0, vy = 0.0, vz = 0.0;
					if (idx < nVel) {
						vx = traj.velocity[idx].x(); // If it is a .x member, change to traj.velocity[idx].x
						vy = traj.velocity[idx].y();
						vz = traj.velocity[idx].z();
					}
					const double vmag = std::sqrt(vx*vx + vy*vy + vz*vz);
					velMagArr->InsertNextValue(vmag);

					previousLongitude = longitude;
					firstPoint = false;
				}

				// Add this trajectory
				if (polyline->GetNumberOfPoints() > 0) 
				{
					all_lines->InsertNextCell(polyline);
				}
			}

			// create PolyData
			vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
			polydata->SetPoints(all_points);
			polydata->SetLines(all_lines);
			polydata->GetPointData()->AddArray(tempArr);
			polydata->GetPointData()->AddArray(salArr);
			polydata->GetPointData()->AddArray(velMagArr);

			// Write to file
			vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
			writer->SetFileName(checkAndModifyExtension(outputFileName, "vtp").c_str());
			writer->SetInputData(polydata);
			writer->Write();
			
			MOPS_TIMER_STOP("IO::SaveVTP");
		}

	};

}
#endif
