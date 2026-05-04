#pragma once
#include "backend.h"

enum class BackendType {
    CPU,
    CPU_OPTIMIZED,
    GEMM_OPTIMIZED
};

class Device {
public:
    static void set_backend(BackendType type) {
        get_current_type() = type;
        
    }

    static Backend* get() {
        BackendType type = get_current_type();

        if (type == BackendType::CPU_OPTIMIZED) {
            static CPUBackendOptimized opt_backend;
            return &opt_backend;
        } else if (type == BackendType::GEMM_OPTIMIZED) {
            static GEMMOptimizedBackend gemm_backend;
            return &gemm_backend;
        } else {
            static CPUBackend normal_backend;
            return &normal_backend;
        }
        
    }


private:
    static BackendType& get_current_type() {
        static BackendType current = BackendType::CPU;
        return current;
    }
};