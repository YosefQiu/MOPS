#pragma once
#include "ggl.h"
#include <memory>
#include <mutex>
#include <vector>

#define _DEBUG 1
#define MAX_PATH_FOR_LOG	256
#define MAX_LOG_LENGTH		10240
#define RELEASE (!_DEBUG)

int InitEngineLog(const char* engineLogFilePath);
int FormatCurrDate(char* szTime, char* szDate);
void DebugLog(const char* file, int nLine, const char* format, ...);//debug log
void InfoLog(const char* file, int nLine, const char* format, ...);//info log
void ErrorLog(const char* file, int nLine, const char* format, ...);//normal error,need not to be displayed to user
void EditorErrorLog(const char* file, int nLine, const char* format, ...);//error log that need to display to the editor console window

void SetEngineErrorReporter(void(*foo)(const char* msg));
void ReportEngineError(const char* msg);

#if _DEBUG
#define Debug(f,...) DebugLog(__FILE__,__LINE__,f,##__VA_ARGS__)
#else
#define Debug(f,...)
#endif
#define Error(f,...) ErrorLog(__FILE__,__LINE__,f,##__VA_ARGS__)
#define Info(f,...) InfoLog(__FILE__,__LINE__,f,##__VA_ARGS__)
#define errorC(f,...) EditorErrorLog(__FILE__,__LINE__,f,##__VA_ARGS__)
#define DebugLambda(lambda) lambda()

class EngineLog
{
public:
    static char* getEngineLog()
    {
        static char szEngineLog[MAX_PATH_FOR_LOG];
        return szEngineLog;
    }
};
//char szEngineLog[MAX_PATH_FOR_LOG];
static void(*gEngineErrorReporter)(const char* msg) = nullptr;

inline void SetEngineErrorReporter(void(*foo)(const char* msg))
{
    gEngineErrorReporter = foo;
}

inline void ReportEngineError(const char* msg)
{
    if (gEngineErrorReporter != nullptr)
    {
        gEngineErrorReporter(msg);
    }
    else {
        Error("%s", msg);
    }
}

inline int  FormatCurrDate(char* szTime, char* szDate)
{
    time_t t;
    time(&t);
    struct tm* today;
    today = localtime(&t);
    strftime(szTime, 32, "%H:%M:%S", today);
    strftime(szDate, 32, "%y-%m-%d", today);
    return 0;
}

inline int InitEngineLog(const char* engineLog)
{
    int nLen = strlen(engineLog);
    if ((nLen >= MAX_PATH_FOR_LOG) || (nLen <= 0))
        return -1;
    memset(EngineLog::getEngineLog(), 0, MAX_PATH_FOR_LOG);
    strcpy(EngineLog::getEngineLog(), engineLog);
    return 0;
}

inline void WriteLog(const char* tag, const char* file, int nLine, const char* szLogContent)
{
    char szTime[32];
    char szDate[32];
    memset(szTime, 0, sizeof(szTime));
    memset(szDate, 0, sizeof(szDate));
    FormatCurrDate(szTime, szDate);
#if RELEASE
    std::string fn;
    fn = EngineLog::getEngineLog();
    fn += ".";
    fn += szDate;
    std::ofstream fd;
    try {
        fd.open(fn.c_str(), std::ios::app);
        fd << szTime << " : " << tag << " " << szLogContent << std::endl;
        fd.close();
    }
    catch (...) {
        printf("open log file exception!\n");
        return;
    }
#else
    printf("%s %s %s\n", szTime, tag, szLogContent);
#endif
}

inline void DebugLog(const char* file, int nLine, const char* format, ...)
{
    if (strlen(format) == 0)
        return;
    char szBuffer[MAX_LOG_LENGTH];
    memset(szBuffer, 0, MAX_LOG_LENGTH);
    va_list	l_va;
    va_start(l_va, format);
    vsnprintf(szBuffer, sizeof(szBuffer), format, l_va);
    va_end(l_va);
#if _DEBUG
    WriteLog("[DEBUG] ", file, nLine, szBuffer);
#endif
}

inline void InfoLog(const char* file, int nLine, const char* format, ...)
{
    if (strlen(format) == 0)
        return;
    char szBuffer[MAX_LOG_LENGTH];
    memset(szBuffer, 0, MAX_LOG_LENGTH);
    va_list	l_va;
    va_start(l_va, format);
    vsnprintf(szBuffer, sizeof(szBuffer), format, l_va);
    va_end(l_va);
    WriteLog(" [INFO] ", file, nLine, szBuffer);
}

inline void ErrorLog(const char* file, int nLine, const char* format, ...)
{
    if (strlen(format) == 0)
        return;
    char szBuffer[MAX_LOG_LENGTH];
    memset(szBuffer, 0, MAX_LOG_LENGTH);

    va_list	l_va;
    va_start(l_va, format);
    vsnprintf(szBuffer, sizeof(szBuffer), format, l_va);
    va_end(l_va);
    WriteLog("[ERROR]", file, nLine, szBuffer);
}

inline void EditorErrorLog(const char* file, int nLine, const char* format, ...)
{
    if (strlen(format) == 0)
        return;
    char szBuffer[MAX_LOG_LENGTH];
    memset(szBuffer, 0, MAX_LOG_LENGTH);
    va_list	l_va;
    va_start(l_va, format);
    vsnprintf(szBuffer, sizeof(szBuffer), format, l_va);
    va_end(l_va);
    if (gEngineErrorReporter)
    {
        char szTime[32];
        char szDate[32];
        std::string fn;
        memset(szTime, 0, sizeof(szTime));
        memset(szDate, 0, sizeof(szDate));
        FormatCurrDate(szTime, szDate);
        fn = EngineLog::getEngineLog();
        fn += ".";
        fn += szDate;
        gEngineErrorReporter(fn.c_str());
    }
    else
    {
        WriteLog("[ERRORC]", file, nLine, szBuffer);
    }
}