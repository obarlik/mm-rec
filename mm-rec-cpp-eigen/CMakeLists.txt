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
endif()

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

# Memory Manager Benchmark
add_executable(test_memory_benchmark tests/test_memory_benchmark.cpp)
target_link_libraries(test_memory_benchmark mm_rec_eigen)

# Pool Allocator Benchmark
add_executable(test_pool_benchmark tests/test_pool_benchmark.cpp)
target_link_libraries(test_pool_benchmark mm_rec_eigen)
add_test(NAME test_pool_benchmark COMMAND test_pool_benchmark)
add_executable(inference_demo src/inference_demo.cpp)
target_link_libraries(inference_demo mm_rec_eigen)

add_executable(demo_training_cpp src/demo_training_cpp.cpp)
target_link_libraries(demo_training_cpp mm_rec_eigen)

add_executable(train_kernel src/train_kernel.cpp)
target_link_libraries(train_kernel mm_rec_eigen)

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
