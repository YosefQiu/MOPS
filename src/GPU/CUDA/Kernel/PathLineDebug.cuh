#pragma once

/**
 * @file PathLineDebug.cuh
 * @brief Debug utilities for PathLine kernel
 *
 * Usage:
 *   1. Enable debug output by defining MOPS_DEBUG_PATHLINE in CMakeLists.txt:
 *      add_compile_definitions(MOPS_DEBUG_PATHLINE=1)
 *
 *   2. Use debug macros in code:
 *      DEBUG_PATHLINE("Step %d completed", i_step);
 *      DEBUG_PATHLINE_PARTICLE(particle_id, "Position: %f,%f,%f", x, y, z);
 *
 *   3. Disable by removing the compile definition or setting to 0
 */

// ============================================================================
// Configuration
// ============================================================================

// Set debug level (can be overridden by compile definition)
#ifndef MOPS_DEBUG_PATHLINE
    #define MOPS_DEBUG_PATHLINE 0  // 0=off, 1=basic, 2=verbose, 3=very verbose
#endif

// Which particle to debug (set to -1 to debug all, or specific ID like 0)
#ifndef MOPS_DEBUG_PARTICLE_ID
    #define MOPS_DEBUG_PARTICLE_ID -1
#endif

// Which step to debug (set to -1 to debug all, or specific step)
#ifndef MOPS_DEBUG_STEP_ID
    #define MOPS_DEBUG_STEP_ID -1
#endif

// ============================================================================
// Debug Macros
// ============================================================================

#if MOPS_DEBUG_PATHLINE > 0

    /**
     * Basic debug print (only from thread 0)
     * Usage: DEBUG_PATHLINE("Message: %d", value);
     */
    #define DEBUG_PATHLINE(fmt, ...) \
        do { \
            if (threadIdx.x == 0 && blockIdx.x == 0) { \
                printf("[PathLine] " fmt "\n", ##__VA_ARGS__); \
            } \
        } while(0)

    /**
     * Debug print for specific particle (only if global_id matches)
     * Usage: DEBUG_PATHLINE_PARTICLE(particle_id, "Position: %f", x);
     */
    #define DEBUG_PATHLINE_PARTICLE(particle_id, fmt, ...) \
        do { \
            if (MOPS_DEBUG_PARTICLE_ID < 0 || (particle_id) == MOPS_DEBUG_PARTICLE_ID) { \
                printf("[PathLine P%d] " fmt "\n", (particle_id), ##__VA_ARGS__); \
            } \
        } while(0)

    /**
     * Debug print for specific step
     * Usage: DEBUG_PATHLINE_STEP(i_step, "Computing RK4");
     */
    #define DEBUG_PATHLINE_STEP(step, fmt, ...) \
        do { \
            if (MOPS_DEBUG_STEP_ID < 0 || (step) == MOPS_DEBUG_STEP_ID) { \
                if (threadIdx.x == 0 && blockIdx.x == 0) { \
                    printf("[PathLine S%d] " fmt "\n", (step), ##__VA_ARGS__); \
                } \
            } \
        } while(0)

    /**
     * Debug print for specific particle at specific step
     * Usage: DEBUG_PATHLINE_PS(particle_id, i_step, "Cell: %d", cell_id);
     */
    #define DEBUG_PATHLINE_PS(particle_id, step, fmt, ...) \
        do { \
            bool match_particle = (MOPS_DEBUG_PARTICLE_ID < 0 || (particle_id) == MOPS_DEBUG_PARTICLE_ID); \
            bool match_step = (MOPS_DEBUG_STEP_ID < 0 || (step) == MOPS_DEBUG_STEP_ID); \
            if (match_particle && match_step) { \
                printf("[PathLine P%d S%d] " fmt "\n", (particle_id), (step), ##__VA_ARGS__); \
            } \
        } while(0)

    /**
     * Verbose debug (level 2+)
     * Usage: DEBUG_PATHLINE_VERBOSE("Detailed info: %f", value);
     */
    #if MOPS_DEBUG_PATHLINE >= 2
        #define DEBUG_PATHLINE_VERBOSE(fmt, ...) DEBUG_PATHLINE(fmt, ##__VA_ARGS__)
    #else
        #define DEBUG_PATHLINE_VERBOSE(fmt, ...) do {} while(0)
    #endif

    /**
     * Very verbose debug (level 3+)
     * Usage: DEBUG_PATHLINE_VVERBOSE("Inner loop: %d", i);
     */
    #if MOPS_DEBUG_PATHLINE >= 3
        #define DEBUG_PATHLINE_VVERBOSE(fmt, ...) DEBUG_PATHLINE(fmt, ##__VA_ARGS__)
    #else
        #define DEBUG_PATHLINE_VVERBOSE(fmt, ...) do {} while(0)
    #endif

#else
    // Debug disabled - all macros become no-ops
    #define DEBUG_PATHLINE(fmt, ...) do {} while(0)
    #define DEBUG_PATHLINE_PARTICLE(particle_id, fmt, ...) do {} while(0)
    #define DEBUG_PATHLINE_STEP(step, fmt, ...) do {} while(0)
    #define DEBUG_PATHLINE_PS(particle_id, step, fmt, ...) do {} while(0)
    #define DEBUG_PATHLINE_VERBOSE(fmt, ...) do {} while(0)
    #define DEBUG_PATHLINE_VVERBOSE(fmt, ...) do {} while(0)
#endif

// ============================================================================
// Utility Macros
// ============================================================================

/**
 * Print vec3 for debugging
 * Usage: DEBUG_PATHLINE_VEC3(particle_id, "Position", pos);
 */
#define DEBUG_PATHLINE_VEC3(particle_id, label, vec) \
    DEBUG_PATHLINE_PARTICLE(particle_id, "%s: (%.6f, %.6f, %.6f)", label, (vec).x(), (vec).y(), (vec).z())

/**
 * Print array values
 * Usage: DEBUG_PATHLINE_ARRAY(particle_id, "Weights", weights, 3);
 */
#define DEBUG_PATHLINE_ARRAY(particle_id, label, arr, n) \
    do { \
        if (MOPS_DEBUG_PATHLINE > 0) { \
            printf("[PathLine P%d] %s: [", (particle_id), label); \
            for (int _i = 0; _i < (n); ++_i) { \
                printf("%s%.6f", (_i > 0 ? ", " : ""), (arr)[_i]); \
            } \
            printf("]\n"); \
        } \
    } while(0)

/**
 * Assert-like debug check (only in debug builds)
 * Usage: DEBUG_PATHLINE_CHECK(cell_id >= 0, particle_id, "Invalid cell: %d", cell_id);
 */
#define DEBUG_PATHLINE_CHECK(condition, particle_id, fmt, ...) \
    do { \
        if (MOPS_DEBUG_PATHLINE > 0) { \
            if (!(condition)) { \
                printf("[PathLine P%d ERROR] " fmt "\n", (particle_id), ##__VA_ARGS__); \
            } \
        } \
    } while(0)

// ============================================================================
// Example Usage
// ============================================================================

/*
// In your kernel code:

#include "PathLineDebug.cuh"

__global__ void KernelPathLine(...) {
    int global_id = ...;

    // Basic debug (only thread 0)
    DEBUG_PATHLINE("PathLine kernel started with %d particles", particle_count);

    // Debug specific particle
    DEBUG_PATHLINE_PARTICLE(global_id, "Starting position: %f,%f,%f", pos.x(), pos.y(), pos.z());

    // Debug specific step
    DEBUG_PATHLINE_STEP(i_step, "Computing RK4 k1");

    // Debug specific particle at specific step
    DEBUG_PATHLINE_PS(global_id, i_step, "Cell: %d, Depth: %.2f", cell_id, depth);

    // Verbose debug (only if level >= 2)
    DEBUG_PATHLINE_VERBOSE("Intermediate calculation: %f", intermediate_value);

    // Vector debug
    vec3 pos = ...;
    DEBUG_PATHLINE_VEC3(global_id, "Current position", pos);

    // Array debug
    double weights[3] = {...};
    DEBUG_PATHLINE_ARRAY(global_id, "Barycentric weights", weights, 3);

    // Conditional check
    DEBUG_PATHLINE_CHECK(cell_id >= 0, global_id, "Invalid cell ID: %d", cell_id);
}

// To enable debug output:
// 1. In CMakeLists.txt, add:
//    add_compile_definitions(MOPS_DEBUG_PATHLINE=1)
//    add_compile_definitions(MOPS_DEBUG_PARTICLE_ID=0)  # Debug particle 0
//    add_compile_definitions(MOPS_DEBUG_STEP_ID=22)     # Debug step 22
//
// 2. Or use compile flags:
//    nvcc -DMOPS_DEBUG_PATHLINE=2 -DMOPS_DEBUG_PARTICLE_ID=0 ...

*/
