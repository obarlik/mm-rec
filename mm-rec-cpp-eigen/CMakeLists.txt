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
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

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
message(STATUS "   Eigen is HEADER-ONLY - zero runtime dependency!")

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${EIGEN3_INCLUDE_DIR})

# Source files
set(CORE_SOURCES
    src/core/tensor.cpp
    src/core/linear.cpp
    src/core/gated_memory.cpp
)

set(MODEL_SOURCES
    src/model/mm_rec_block.cpp
    src/model/mm_rec_model.cpp
)

set(TRAINING_SOURCES
    src/training/uboo_loss.cpp
)

set(CONFIG_SOURCES
    src/config/model_config.cpp
)

# Main library
add_library(mm_rec_eigen SHARED 
    ${CORE_SOURCES}
    ${MODEL_SOURCES}
    ${TRAINING_SOURCES}
    ${CONFIG_SOURCES}
)

target_link_libraries(mm_rec_eigen
    Eigen3::Eigen
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(mm_rec_eigen OpenMP::OpenMP_CXX)
endif()

# Tests
enable_testing()

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor mm_rec_eigen)
add_test(NAME test_tensor COMMAND test_tensor)

add_executable(test_gated_memory tests/test_gated_memory.cpp)
target_link_libraries(test_gated_memory mm_rec_eigen)
add_test(NAME test_gated_memory COMMAND test_gated_memory)

add_executable(test_model tests/test_model.cpp)
target_link_libraries(test_model mm_rec_eigen)
add_test(NAME test_model COMMAND test_model)

add_executable(test_uboo_loss tests/test_uboo_loss.cpp)
target_link_libraries(test_uboo_loss mm_rec_eigen)
add_test(NAME test_uboo_loss COMMAND test_uboo_loss)

add_executable(test_error_handling tests/test_error_handling.cpp)
target_link_libraries(test_error_handling mm_rec_eigen)
add_test(NAME test_error_handling COMMAND test_error_handling)

# Inference executable
add_executable(inference_demo src/inference_demo.cpp)
target_link_libraries(inference_demo mm_rec_eigen)

# Print configuration
message(STATUS "========================================")
message(STATUS "MM-Rec Eigen (Zero Runtime Deps!)")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Eigen version: ${EIGEN3_VERSION}")
message(STATUS "OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "Runtime deps: ZERO! (Eigen is header-only)")
message(STATUS "========================================")
