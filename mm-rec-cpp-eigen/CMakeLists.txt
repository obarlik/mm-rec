cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mm-rec-eigen VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags for maximum performance
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -funroll-loops -ffast-math -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address") # Added -fsanitize=address to debug flags

# Find OpenMP (optional for parallelization)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "✅ OpenMP found")
else()
    message(STATUS "⚠️  OpenMP not found (single-threaded)")
    message(STATUS "⚠️  OpenMP not found (single-threaded)")
endif()

# Find OpenCL (for iGPU offloading)
find_package(OpenCL)
if(OpenCL_FOUND)
    message(STATUS "✅ OpenCL found: ${OpenCL_VERSION_STRING}")
    add_definitions(-DUSE_OPENCL)
    include_directories(${OpenCL_INCLUDE_DIRS})
else()
    message(STATUS "⚠️  OpenCL not found. iGPU offloading disabled.")
endif()

# ==========================================
# GPU / CUDA Configuration (Ready for Future)
# ==========================================
# option(USE_CUDA "Enable CUDA Support" OFF)
# if(USE_CUDA)
#     enable_language(CUDA)
#     find_package(CUDAToolkit REQUIRED)
#     add_definitions(-DUSE_CUDA)
#     message(STATUS "🚀 GPU Mode: ON (CUDA Enabled)")
# else()
#     message(STATUS "ℹ️  GPU Mode: OFF (CPU Only)")
# endif()
# ==========================================

# Find Eigen3 (header-only!)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "✅ Eigen3 found: ${EIGEN3_VERSION}")

# --- MKL STATIC LINKING ---
# We manually define the paths since we found them in /usr/lib/x86_64-linux-gnu/
set(MKL_DIR "/usr/lib/x86_64-linux-gnu")
set(MKL_LIBS 
    ${MKL_DIR}/libmkl_intel_lp64.a
    ${MKL_DIR}/libmkl_gnu_thread.a
    ${MKL_DIR}/libmkl_core.a
)

# Check if they exist (sanity check)
if(EXISTS "${MKL_DIR}/libmkl_core.a")
    message(STATUS "🚀 MKL Static Libraries FOUND at ${MKL_DIR}")
    
    # Enable MKL in Eigen
    add_definitions(-DEIGEN_USE_MKL_ALL)
    
    # Link Flags (Important for static linking: --start-group ... --end-group)
    # We must also link dl, pthread, m (math), and gomp (OpenMP)
    set(MKL_LINK_FLAGS "-Wl,--start-group ${MKL_DIR}/libmkl_intel_lp64.a ${MKL_DIR}/libmkl_gnu_thread.a ${MKL_DIR}/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl -lgomp")
    
    message(STATUS "✅ Performance Mode: ON (Using MKL Static)")
else()
    message(WARNING "⚠️  MKL NOT FOUND. Falling back to generic optimizations.")
endif()
# --------------------------

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src) # Allow including from src/cli
include_directories(${EIGEN3_INCLUDE_DIR})
# Add MKL Include Directory
include_directories(${MKL_DIR}/../../include/mkl) # Try relative to lib
include_directories(/usr/include/mkl) # Try standard standard


# Source files
set(CORE_SOURCES
    src/core/tensor.cpp
    src/core/global_allocator.cpp
    src/core/linear.cpp
    src/core/gated_memory.cpp
    src/core/normalization.cpp
)

set(MODEL_SOURCES
    src/model/mm_rec_block.cpp
    src/model/mm_rec_model.cpp
    src/model/moe.cpp
)

set(TRAINING_SOURCES
    src/training/uboo_loss.cpp
    src/training/backward.cpp
    src/training/gru_backward.cpp
    src/training/gradient_utils.cpp
    src/training/scheduler.cpp
    src/training/trainer.cpp
    src/training/sample_tracker.cpp
)

set(CONFIG_SOURCES
    src/config/model_config.cpp
)

set(UTILS_SOURCES
    src/utils/session.cpp
    src/utils/checkpoint.cpp
)

set(DATA_SOURCES
    src/data/dataset.cpp
    src/data/data_loader.cpp
    src/data/tokenizer.cpp
)

# Main library
add_library(mm_rec_eigen STATIC 
    ${CORE_SOURCES}
    ${MODEL_SOURCES}
    ${TRAINING_SOURCES}
    ${CONFIG_SOURCES}
    ${UTILS_SOURCES}
    ${DATA_SOURCES}
)

target_link_libraries(mm_rec_eigen
    Eigen3::Eigen
)

if(MKL_LINK_FLAGS)
    target_link_libraries(mm_rec_eigen ${MKL_LINK_FLAGS})
elseif(OpenMP_CXX_FOUND)
    target_link_libraries(mm_rec_eigen OpenMP::OpenMP_CXX)
endif()

# Unified CLI Tool
add_executable(mm_rec 
    src/mm_rec_cli.cpp 
    src/cli/cmd_prepare.cpp 
    src/cli/cmd_train.cpp
    src/cli/cmd_infer.cpp
    src/cli/cmd_parse_metrics.cpp
    src/cli/cmd_view_trace.cpp
)
target_link_libraries(mm_rec mm_rec_eigen)

# Tests
enable_testing()
add_executable(test_moe tests/test_moe.cpp)
target_link_libraries(test_moe mm_rec_eigen)
add_test(NAME test_moe COMMAND test_moe)

add_executable(test_data_pipeline tests/test_data_pipeline.cpp)
target_link_libraries(test_data_pipeline mm_rec_eigen)
add_test(NAME test_data_pipeline COMMAND test_data_pipeline)

add_executable(test_rmsnorm tests/test_rmsnorm.cpp)
target_link_libraries(test_rmsnorm mm_rec_eigen)
add_test(NAME test_rmsnorm COMMAND test_rmsnorm)

add_executable(test_adversarial tests/test_adversarial.cpp)
target_link_libraries(test_adversarial mm_rec_eigen)
add_test(NAME test_adversarial COMMAND test_adversarial)

add_executable(hardware_probe tests/hardware_probe.cpp)
# Enable F16C and AVX2 explicitly for this probe
target_compile_options(hardware_probe PRIVATE -mf16c -mavx2)
target_link_libraries(hardware_probe mm_rec_eigen)

add_executable(test_compressed_integration tests/test_compressed_integration.cpp)
target_compile_options(test_compressed_integration PRIVATE -mf16c -mavx2)
target_link_libraries(test_compressed_integration mm_rec_eigen)
add_test(NAME test_compressed_integration COMMAND test_compressed_integration)

# Memory Manager Benchmark
add_executable(test_memory_benchmark tests/test_memory_benchmark.cpp)
target_link_libraries(test_memory_benchmark mm_rec_eigen)

# Pool Allocator Benchmark
add_executable(test_pool_benchmark tests/test_pool_benchmark.cpp)
target_link_libraries(test_pool_benchmark mm_rec_eigen)
add_test(NAME test_pool_benchmark COMMAND test_pool_benchmark)

add_executable(test_system_optimizer tests/test_system_optimizer.cpp)
target_link_libraries(test_system_optimizer mm_rec_eigen)
add_test(NAME test_system_optimizer COMMAND test_system_optimizer)

add_executable(test_optimizer_benchmark tests/test_optimizer_benchmark.cpp)
target_link_libraries(test_optimizer_benchmark mm_rec_eigen)
add_test(NAME test_optimizer_benchmark COMMAND test_optimizer_benchmark)

add_executable(test_igpu tests/test_igpu.cpp)
target_link_libraries(test_igpu mm_rec_eigen ${CMAKE_DL_LIBS}) # Link dl for dlopen
add_test(NAME test_igpu COMMAND test_igpu)

add_executable(test_vulkan_probe tests/test_vulkan_probe.cpp)
target_link_libraries(test_vulkan_probe ${CMAKE_DL_LIBS})
add_test(NAME test_vulkan_probe COMMAND test_vulkan_probe)

add_executable(test_compression_benchmark tests/test_compression_benchmark.cpp)
target_compile_options(test_compression_benchmark PRIVATE -mf16c -mavx2)
target_link_libraries(test_compression_benchmark mm_rec_eigen)
add_test(NAME test_compression_benchmark COMMAND test_compression_benchmark)

add_executable(test_opengl_probe tests/test_opengl_probe.cpp)
target_link_libraries(test_opengl_probe ${CMAKE_DL_LIBS})
add_test(NAME test_opengl_probe COMMAND test_opengl_probe)

add_executable(benchmark_final_suite tests/benchmark_final_suite.cpp)
target_link_libraries(benchmark_final_suite mm_rec_eigen ${CMAKE_DL_LIBS})
add_test(NAME benchmark_final_suite COMMAND benchmark_final_suite)

# add_executable(test_vulkan_memory tests/test_vulkan_memory.cpp)
# target_link_libraries(test_vulkan_memory ${CMAKE_DL_LIBS})
# add_test(NAME test_vulkan_memory COMMAND test_vulkan_memory)
add_executable(inference_demo src/inference_demo.cpp)
target_link_libraries(inference_demo mm_rec_eigen)

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer mm_rec_eigen)
add_test(NAME test_tokenizer COMMAND test_tokenizer)

add_executable(benchmark_tokenizer tests/benchmark_tokenizer.cpp)
target_link_libraries(benchmark_tokenizer mm_rec_eigen)

add_executable(demo_training_cpp src/demo_training_cpp.cpp)
target_link_libraries(demo_training_cpp mm_rec_eigen)

add_executable(train_kernel src/train_kernel.cpp)
target_link_libraries(train_kernel mm_rec_eigen)

add_executable(test_hybrid_stability tests/test_hybrid_stability.cpp)
target_link_libraries(test_hybrid_stability mm_rec_eigen)

add_executable(debug_dataloader tests/debug_dataloader.cpp)
target_link_libraries(debug_dataloader mm_rec_eigen)

add_executable(debug_overfit tests/debug_overfit.cpp)
target_link_libraries(debug_overfit mm_rec_eigen)

# Metrics Tests
add_executable(benchmark_metrics_overhead tests/benchmark_metrics_overhead.cpp)
target_link_libraries(benchmark_metrics_overhead mm_rec_eigen)
add_test(NAME benchmark_metrics_overhead COMMAND benchmark_metrics_overhead)

add_executable(test_binary_metrics test_binary_metrics.cpp)
target_link_libraries(test_binary_metrics mm_rec_eigen)
add_test(NAME test_binary_metrics COMMAND test_binary_metrics)

# Logger Tests
add_executable(benchmark_logger_overhead tests/benchmark_logger_overhead.cpp)
target_link_libraries(benchmark_logger_overhead mm_rec_eigen)
add_test(NAME benchmark_logger_overhead COMMAND benchmark_logger_overhead)

add_executable(test_logger tests/test_logger.cpp)
target_link_libraries(test_logger mm_rec_eigen)
add_test(NAME test_logger COMMAND test_logger)

# UI Demo
add_executable(demo_ui tests/demo_ui.cpp)
target_link_libraries(demo_ui mm_rec_eigen)




# Config/Info
message(STATUS "========================================")
message(STATUS "MM-Rec Eigen (Combined CLI)")
message(STATUS "MM-Rec Eigen (Zero Runtime Deps!)")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Eigen version: ${EIGEN3_VERSION}")
message(STATUS "OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "Runtime deps: ZERO! (Eigen is header-only)")
message(STATUS "========================================")
