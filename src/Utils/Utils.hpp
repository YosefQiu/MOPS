#pragma once
#include "ggl.h"
#include <utility>
#include "Log.hpp"

inline int toIntYMD(const std::string& s) 
{
    int y, m, d;
    char dash;
    std::stringstream ss(s);
    ss >> y >> dash >> m >> dash >> d;
    return y * 10000 + m * 100 + d;  // 150401
}

inline bool compareByDate(const std::string& a, const std::string& b)
{
    std::regex re(R"((\d{4}-\d{2}-\d{2}))");
    std::smatch ma, mb;
    std::string da, db;

    if (std::regex_search(a, ma, re)) da = ma.str(1);
    if (std::regex_search(b, mb, re)) db = mb.str(1);

    return da < db;

}

inline bool compareByNumber(const std::string& a, const std::string& b)
{
    std::regex re(R"((\d+))"); 
    std::smatch ma, mb;
    int na = 0, nb = 0;

    if (std::regex_search(a, ma, re)) na = std::stoi(ma.str(1));
    if (std::regex_search(b, mb, re)) nb = std::stoi(mb.str(1));

    return na < nb;
}

inline std::vector<std::string> getFilesWithPrefix(const std::string& folder, const std::string& prefix,
        std::function<bool(const std::string&, const std::string&)> sortFn = {})
{
    namespace fs = std::filesystem;
    std::vector<std::string> files;

    try
    {
        for (const auto& entry : fs::directory_iterator(folder))
        {
            if (entry.is_regular_file())
            {
                std::string filename = entry.path().filename().string();
                if (filename.rfind(prefix, 0) == 0)
                    files.push_back(filename);
            }
        }
    }
    catch (const std::exception& e)
    {
        Debug("[ERROR]::getFilesWithPrefix: %s", e.what());
    }

    if (sortFn)
        std::sort(files.begin(), files.end(), sortFn);
    else
        std::sort(files.begin(), files.end());

    return files;
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

inline int getTimeGapinSecond(const char* t1, const char* t2)
{
    std::tm tm1 = {}, tm2 = {};

    std::istringstream ss1{std::string(t1)};
    ss1 >> std::get_time(&tm1, "%Y-%m-%d_%H:%M:%S");

    std::istringstream ss2{std::string(t2)};
    ss2 >> std::get_time(&tm2, "%Y-%m-%d_%H:%M:%S");


    if (ss1.fail() || ss2.fail())
        throw std::runtime_error("Error in parsing time");

    std::time_t time1 = std::mktime(&tm1);
    std::time_t time2 = std::mktime(&tm2);

    return static_cast<int>(std::difftime(time1, time2));

}
