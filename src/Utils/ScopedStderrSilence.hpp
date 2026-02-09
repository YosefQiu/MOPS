#pragma once
#include "ggl.h"
struct ScopedStderrSilence {
    int saved_fd;
    ScopedStderrSilence() {
        saved_fd = dup(fileno(stderr));
        freopen("/dev/null", "w", stderr);
    }
    ~ScopedStderrSilence() {
        fflush(stderr);
        dup2(saved_fd, fileno(stderr));
        close(saved_fd);
    }
};