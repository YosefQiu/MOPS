#pragma once
#include "ggl.h"
#include "ImageBuffer.hpp"
#include "MPASOVisualizer.h"



inline std::string TypeToStr(VisualizeType type)
{
	switch (type)
	{
	case VisualizeType::kFixedLayer: return "fixed_layer_"; break;
	case VisualizeType::kFixedDepth: return "fixed_depth_"; break;
	}
}

inline std::string checkAndModifyExtension(std::string outputName, const std::string& type) 
{
	size_t pos = outputName.find_last_of('.');
	if (pos != std::string::npos)
	{
		
		std::string currentExtension = outputName.substr(pos + 1);
		if (currentExtension == type) 
		{ 
			return outputName; 
		}
		else
		{ 
			return outputName.substr(0, pos + 1) + type; 
		}
	}
	else 
	{
		return outputName + "." + type;
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


		// 创建VTK的ImageData对象
		vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
		imageData->SetDimensions(width, height, 1);
		imageData->AllocateScalars(VTK_DOUBLE, 3); // 每个点有三个属性（X, Y, Z）
		imageData->SetOrigin(config->LonRange.x(), config->LatRange.x(), k);  // 设置数据的起始位置
		imageData->SetSpacing(lonSpacing, latSpacing, k);  // 设置每个像素的物理尺寸

		// 填充数据
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

		// 设置writer并保存文件
		vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
		writer->SetFileName((outputName + TypeToStr(config->VisType) + std::to_string(k) + ".vti").c_str());
		writer->SetInputData(imageData);
		writer->Write();
		Debug("[VTKFileManager]::Finished... %s", (TypeToStr(config->VisType) + std::to_string(k) + ".vti").c_str());
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

		// 遍历每个文件中的点，连接成一条线
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

		// 创建一个 PolyData 对象并设置 points 和 lines
		vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
		polydata->SetPoints(new_points);
		polydata->SetLines(lines);

		// 将 PolyData 写入到文件
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

		// 遍历每个文件中的点，连接成一条线
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

		// 创建一个 PolyData 对象并设置 points 和 lines
		vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
		polydata->SetPoints(new_points);
		polydata->SetLines(lines);

		// 将 PolyData 写入到文件
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
			// 删除已读入的文件
            std::filesystem::remove(fileName);
        }

        appendFilter->Update();

        vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
        writer->SetFileName(checkAndModifyExtension(outputFileName, "vtp").c_str());
        writer->SetInputData(appendFilter->GetOutput());
        writer->Write();
		Debug("[VTKFileManager]::Finished... %s", outputFileName.c_str());
    }

};
