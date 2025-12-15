cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mm-rec-pure VERSION 0.1.0 LANGUAGES CXX)

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
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find MKL (Intel Math Kernel Library)
option(USE_MKL "Use Intel MKL" ON)

if(USE_MKL)
    find_package(MKL QUIET)
    if(MKL_FOUND)
        message(STATUS "✅ Intel MKL found: ${MKL_VERSION}")
        set(BLAS_LIBRARIES ${MKL_LIBRARIES})
        include_directories(${MKL_INCLUDE_DIRS})
        link_directories(${MKL_LIBRARY_DIRS})
    else()
        # Try pkg-config
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
            pkg_check_modules(MKL QUIET mkl-dynamic-lp64-seq)
            if(MKL_FOUND)
                message(STATUS "✅ MKL via pkg-config")
                set(BLAS_LIBRARIES ${MKL_LIBRARIES})
                include_directories(${MKL_INCLUDE_DIRS})
                link_directories(${MKL_LIBRARY_DIRS})
            endif()
        endif()
    endif()
    
    # Fallback: Check system paths
    if(NOT MKL_FOUND)
        find_library(MKL_CORE mkl_core PATHS /usr/lib/x86_64-linux-gnu)
        if(MKL_CORE)
            message(STATUS "✅ MKL found in system path")
            set(BLAS_LIBRARIES mkl_intel_lp64 mkl_sequential mkl_core pthread m dl)
        endif()
    endif()
endif()

# Fallback to OpenBLAS if MKL not found
if(NOT BLAS_LIBRARIES)
    message(STATUS "⚠️  MKL not found, trying OpenBLAS")
    find_package(BLAS REQUIRED)
    set(BLAS_LIBRARIES ${BLAS_LIBRARIES})
    message(STATUS "✅ Using BLAS: ${BLAS_LIBRARIES}")
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(CORE_SOURCES
    src/core/tensor.cpp
    src/core/linear.cpp
    src/core/gated_memory.cpp
)

# Main library
add_library(mm_rec_pure SHARED ${CORE_SOURCES})

target_link_libraries(mm_rec_pure
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# Tests
enable_testing()

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor mm_rec_pure)
add_test(NAME test_tensor COMMAND test_tensor)

add_executable(test_gated_memory tests/test_gated_memory.cpp)
target_link_libraries(test_gated_memory mm_rec_pure)
add_test(NAME test_gated_memory COMMAND test_gated_memory)

# Print configuration
message(STATUS "========================================")
message(STATUS "MM-Rec Pure C++ Configuration")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "BLAS: ${BLAS_LIBRARIES}")
message(STATUS "========================================")
