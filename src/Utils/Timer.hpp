#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <algorithm>

namespace MOPS {

/**
 * @brief Internal timing categories for MOPS operations
 */
enum class TimerCategory {
    IO_Read,           // Reading data from disk (NetCDF, etc.)
    IO_Write,          // Writing data to disk (VTK, etc.)
    Preprocessing,     // Grid/solution preprocessing (KDTree, velocity calc, etc.)
    MemoryCopy,        // CPU <-> GPU memory transfers
    GPUKernel,         // GPU computation kernels
    CPUCompute,        // CPU computation
    Other              // Miscellaneous
};

/**
 * @brief Convert TimerCategory to string
 */
inline const char* categoryToString(TimerCategory cat) {
    switch (cat) {
        case TimerCategory::IO_Read:        return "IO (Read)";
        case TimerCategory::IO_Write:       return "IO (Write)";
        case TimerCategory::Preprocessing:  return "Preprocessing";
        case TimerCategory::MemoryCopy:     return "Memory Copy";
        case TimerCategory::GPUKernel:      return "GPU Kernel";
        case TimerCategory::CPUCompute:     return "CPU Compute";
        case TimerCategory::Other:          return "Other";
        default:                            return "Unknown";
    }
}

/**
 * @brief Record of a single timing event
 */
struct TimingRecord {
    std::string name;
    TimerCategory category;
    double duration_ms;  // in milliseconds
};

/**
 * @brief Global timer manager for MOPS
 * Thread-safe singleton pattern
 */
class TimerManager {
public:
    static TimerManager& instance() {
        static TimerManager inst;
        return inst;
    }

    // Disable copy
    TimerManager(const TimerManager&) = delete;
    TimerManager& operator=(const TimerManager&) = delete;

    /**
     * @brief Start timing an operation
     */
    void start(const std::string& name, TimerCategory category) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_activeTimers[name] = {category, std::chrono::high_resolution_clock::now()};
    }

    /**
     * @brief Stop timing an operation and record the duration
     */
    void stop(const std::string& name) {
        auto end_time = std::chrono::high_resolution_clock::now();
        
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_activeTimers.find(name);
        if (it != m_activeTimers.end()) {
            auto duration = std::chrono::duration<double, std::milli>(
                end_time - it->second.start_time).count();
            
            m_records.push_back({name, it->second.category, duration});
            m_categoryTotals[it->second.category] += duration;
            m_totalTime += duration;
            
            m_activeTimers.erase(it);
        }
    }

    /**
     * @brief Record a duration directly (for external timing)
     */
    void record(const std::string& name, TimerCategory category, double duration_ms) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_records.push_back({name, category, duration_ms});
        m_categoryTotals[category] += duration_ms;
        m_totalTime += duration_ms;
    }

    /**
     * @brief Reset all timing data
     */
    void reset() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_records.clear();
        m_categoryTotals.clear();
        m_activeTimers.clear();
        m_totalTime = 0.0;
    }

    /**
     * @brief Print timing summary by category
     */
    void printSummary() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              MOPS Timing Summary                           ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════╣\n";
        
        // Sort categories by total time (descending)
        std::vector<std::pair<TimerCategory, double>> sorted_cats(
            m_categoryTotals.begin(), m_categoryTotals.end());
        std::sort(sorted_cats.begin(), sorted_cats.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        
        for (const auto& [cat, total] : sorted_cats) {
            double percentage = (m_totalTime > 0) ? (total / m_totalTime * 100.0) : 0.0;
            std::cout << "║ " << std::left << std::setw(20) << categoryToString(cat)
                      << std::right << std::setw(12) << std::fixed << std::setprecision(2) 
                      << total << " ms"
                      << std::setw(10) << std::fixed << std::setprecision(1) 
                      << percentage << " %    ║\n";
        }
        
        std::cout << "╠════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ " << std::left << std::setw(20) << "TOTAL"
                  << std::right << std::setw(12) << std::fixed << std::setprecision(2) 
                  << m_totalTime << " ms"
                  << std::setw(10) << "100.0" << " %    ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    }

    /**
     * @brief Print detailed timing for each operation
     */
    void printDetailed() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                        MOPS Detailed Timing                                  ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
        std::cout << "║ " << std::left << std::setw(40) << "Operation"
                  << std::setw(15) << "Category"
                  << std::right << std::setw(12) << "Time (ms)"
                  << std::setw(8) << "%" << "   ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
        
        for (const auto& rec : m_records) {
            double percentage = (m_totalTime > 0) ? (rec.duration_ms / m_totalTime * 100.0) : 0.0;
            
            // Truncate long names
            std::string name = rec.name;
            if (name.length() > 38) {
                name = name.substr(0, 35) + "...";
            }
            
            std::cout << "║ " << std::left << std::setw(40) << name
                      << std::setw(15) << categoryToString(rec.category)
                      << std::right << std::setw(12) << std::fixed << std::setprecision(2) 
                      << rec.duration_ms
                      << std::setw(7) << std::fixed << std::setprecision(1) 
                      << percentage << "%   ║\n";
        }
        
        std::cout << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
    }

    /**
     * @brief Get total time for a category
     */
    double getCategoryTime(TimerCategory category) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_categoryTotals.find(category);
        return (it != m_categoryTotals.end()) ? it->second : 0.0;
    }

    /**
     * @brief Get total time
     */
    double getTotalTime() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_totalTime;
    }

private:
    TimerManager() = default;

    struct ActiveTimer {
        TimerCategory category;
        std::chrono::high_resolution_clock::time_point start_time;
    };

    mutable std::mutex m_mutex;
    std::vector<TimingRecord> m_records;
    std::unordered_map<TimerCategory, double> m_categoryTotals;
    std::unordered_map<std::string, ActiveTimer> m_activeTimers;
    double m_totalTime = 0.0;
};

/**
 * @brief RAII-style scoped timer
 * Automatically starts timing on construction and stops on destruction
 */
class ScopedTimer {
public:
    ScopedTimer(const std::string& name, TimerCategory category)
        : m_name(name) {
        TimerManager::instance().start(name, category);
    }
    
    ~ScopedTimer() {
        TimerManager::instance().stop(m_name);
    }

    // Disable copy
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string m_name;
};

// ============================================================================
// Convenience macros for timing
// ============================================================================

#ifdef MOPS_ENABLE_TIMING

#define MOPS_TIMER_START(name, category) \
    MOPS::TimerManager::instance().start(name, category)

#define MOPS_TIMER_STOP(name) \
    MOPS::TimerManager::instance().stop(name)

#define MOPS_SCOPED_TIMER(name, category) \
    MOPS::ScopedTimer _mops_scoped_timer_##__LINE__(name, category)

#define MOPS_TIMER_RECORD(name, category, duration_ms) \
    MOPS::TimerManager::instance().record(name, category, duration_ms)

#else

#define MOPS_TIMER_START(name, category) ((void)0)
#define MOPS_TIMER_STOP(name) ((void)0)
#define MOPS_SCOPED_TIMER(name, category) ((void)0)
#define MOPS_TIMER_RECORD(name, category, duration_ms) ((void)0)

#endif

} // namespace MOPS
