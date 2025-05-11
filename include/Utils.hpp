#pragma once
#include "ggl.h"



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

inline std::string createDataPath(const std::string& basePath, const std::string& fileName)
{
	namespace fs = std::filesystem;
	std::string dirPath = "./" + basePath + "/" + fileName;
	if (!fs::exists(dirPath))
	{
		fs::create_directories(dirPath);
		Debug("[Data]::Created directory: %s", dirPath.c_str());
	}
	return dirPath;
}

inline std::string removeFileExtension(const std::string& filePath) 
{
	namespace fs = std::filesystem;
	fs::path pathObj(filePath);
	return (pathObj.has_extension()) ? pathObj.stem().string() : filePath;
}