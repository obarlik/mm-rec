cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mm-rec-cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags for maximum performance
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Find LibTorch
list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find OpenMP
find_package(OpenMP REQUIRED)

# Optional: MKL for optimized BLAS
option(USE_MKL "Use Intel MKL" ON)
if(USE_MKL)
    find_package(MKL QUIET)
    if(MKL_FOUND)
        message(STATUS "✅ Intel MKL found: ${MKL_VERSION}")
        add_definitions(-DUSE_MKL)
        include_directories(${MKL_INCLUDE_DIRS})
        link_directories(${MKL_LIBRARY_DIRS})
        set(BLAS_LIBRARIES ${MKL_LIBRARIES})
    else()
        message(STATUS "⚠️  Intel MKL not found, trying OpenBLAS")
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
            message(STATUS "✅ OpenBLAS found")
            add_definitions(-DUSE_OPENBLAS)
            set(BLAS_LIBRARIES ${BLAS_LIBRARIES})
        else()
            message(STATUS "⚠️  No BLAS library found, using LibTorch")
        endif()
    endif()
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${TORCH_INCLUDE_DIRS})

# Source files
set(CORE_SOURCES
    src/core/gated_memory.cpp
    src/core/log_sum_exp.cpp
)

set(LAYER_SOURCES
    src/layers/mm_rec_block.cpp
    src/layers/memory_state.cpp
    src/layers/uboo_output.cpp
)

set(MODEL_SOURCES
    src/model/mm_rec_model.cpp
    src/model/session.cpp
    src/model/uboo_loss.cpp
)

set(UTILS_SOURCES
    src/utils/checkpoint.cpp
    src/utils/profiler.cpp
)

# Main library
add_library(mm_rec_lib SHARED
    ${CORE_SOURCES}
    ${LAYER_SOURCES}
    ${MODEL_SOURCES}
    ${UTILS_SOURCES}
)

target_link_libraries(mm_rec_lib
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
)

# Set RPATH for LibTorch
set_target_properties(mm_rec_lib PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Inference executable
add_executable(inference src/inference_main.cpp)
target_link_libraries(inference mm_rec_lib ${TORCH_LIBRARIES})

# Tests
enable_testing()
add_subdirectory(tests)

# Installation
install(TARGETS mm_rec_lib LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(TARGETS inference DESTINATION bin)

# Print configuration
message(STATUS "========================================")
message(STATUS "MM-Rec C++ Configuration")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "LibTorch version: ${Torch_VERSION}")
message(STATUS "LibTorch libraries: ${TORCH_LIBRARIES}")
message(STATUS "OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "BLAS: ${BLAS_LIBRARIES}")
message(STATUS "========================================")
