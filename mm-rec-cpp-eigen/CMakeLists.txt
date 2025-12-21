cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mm-rec-eigen VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXECUTABLE_SUFFIX ".bin")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==========================================
# AUTO-CLEANUP: Kill residual processes before build
# ==========================================
# This prevents port conflicts and zombie processes during development
# Only runs once per configure to avoid interrupting builds
set(CLEANUP_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/cleanup_silent.sh")
set(CLEANUP_MARKER "${CMAKE_BINARY_DIR}/.cleanup_done")

if(EXISTS ${CLEANUP_SCRIPT} AND NOT EXISTS ${CLEANUP_MARKER})
    message(STATUS "🧹 Cleaning up residual processes...")
    execute_process(
        COMMAND bash ${CLEANUP_SCRIPT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE CLEANUP_RESULT
    )
    if(CLEANUP_RESULT EQUAL 0)
        message(STATUS "✅ Pre-build cleanup done")
        # Create marker file to skip cleanup on subsequent make calls
        file(WRITE ${CLEANUP_MARKER} "cleanup completed")
    endif()
elseif(EXISTS ${CLEANUP_MARKER})
    # Cleanup already done, skip silently
else()
    message(STATUS "ℹ️  Cleanup script not found, skipping")
endif()
# ==========================================

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
    message(STATUS "ℹ️  OpenCL not found. (Optional: Vulkan is used for main engine)")
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

# Helper to embed binary resources (e.g. SPIR-V shaders) directly into the binary
function(embed_resource target input_file)
    get_filename_component(input_name ${input_file} NAME)
    get_filename_component(input_dir ${input_file} DIRECTORY)
    
    # Target object file
    set(output_file "${CMAKE_CURRENT_BINARY_DIR}/${input_name}.o")
    
    # Run ld in the input directory so symbols don't include the full path
    # Symbol will be: _binary_matmul_spv_start (dots become underscores)
    add_custom_command(
        OUTPUT ${output_file}
        COMMAND ld -r -b binary -z noexecstack -o ${output_file} ${input_name}
        DEPENDS ${input_file}
        WORKING_DIRECTORY ${input_dir}
        COMMENT "Embedding Resource: ${input_name} -> Object"
    )
    
    set_source_files_properties(${output_file} PROPERTIES GENERATED TRUE)
    target_sources(${target} PRIVATE ${output_file})
endfunction()

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
    src/core/auto_tuner.cpp
    src/core/dynamic_balancer.cpp
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

set(INFRASTRUCTURE_SOURCES
    src/infrastructure/config.cpp
)

set(BUSINESS_SOURCES
    src/business/checkpoint.cpp
)

set(APPLICATION_SOURCES
    src/application/dashboard_manager.cpp
    src/application/run_manager.cpp
    src/application/session.cpp
)

set(JOB_SOURCES
    src/jobs/job_training.cpp
)

set(DATA_SOURCES
    src/data/dataset.cpp
    src/data/data_loader.cpp
    src/data/tokenizer.cpp
)
# Embed dashboard assets
add_subdirectory(src/dashboard)


# Main library
add_library(mm_rec_eigen STATIC 
    ${CORE_SOURCES}
    ${MODEL_SOURCES}
    ${TRAINING_SOURCES}
    ${CONFIG_SOURCES}
    ${INFRASTRUCTURE_SOURCES}
    ${BUSINESS_SOURCES}
    ${APPLICATION_SOURCES}
    ${JOB_SOURCES}
    ${DATA_SOURCES}
)

target_link_libraries(mm_rec_eigen dashboard_assets)


target_link_libraries(mm_rec_eigen
    Eigen3::Eigen
)

# ------------------------------------------
# AUTOMATIC SHADER COMPILATION & EMBEDDING
# ------------------------------------------
# 1. Find the glslc compiler (from Vulkan SDK)
find_program(GLSLC_EXECUTABLE NAMES glslc HINTS $ENV{VULKAN_SDK}/bin)

if(GLSLC_EXECUTABLE)
    message(STATUS "✅ Found glslc: ${GLSLC_EXECUTABLE}")
else()
    message(WARNING "⚠️  glslc not found! Shaders will not be recompiled. Expecting existing .spv files.")
endif()

# 2. Find all source .comp shaders
file(GLOB RAW_SHADERS "${CMAKE_SOURCE_DIR}/src/shaders/*.comp")
set(COMPILED_SHADERS "")

# Create output directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/src/shaders)

foreach(shader_src ${RAW_SHADERS})
    get_filename_component(shader_name ${shader_src} NAME_WE)
    set(shader_spv "${CMAKE_BINARY_DIR}/src/shaders/${shader_name}.spv")
    
    # If glslc is present, compile .comp -> .spv
    if(GLSLC_EXECUTABLE)
        add_custom_command(
            OUTPUT ${shader_spv}
            COMMAND ${GLSLC_EXECUTABLE} -O ${shader_src} -o ${shader_spv}
            DEPENDS ${shader_src}
            COMMENT "Compiling Shader: ${shader_name}.comp -> .spv"
        )
    endif()
    
    # 3. Add to list for embedding
    # Note: If glslc missing, we assume .spv might already exist in source or we fail later
    if(GLSLC_EXECUTABLE OR EXISTS ${shader_spv})
        list(APPEND COMPILED_SHADERS ${shader_spv})
    else()
        # Fallback: Check if .spv exists in source tree
        set(source_spv "${CMAKE_SOURCE_DIR}/src/shaders/${shader_name}.spv")
        if(EXISTS ${source_spv})
             file(COPY ${source_spv} DESTINATION ${CMAKE_BINARY_DIR}/src/shaders)
             list(APPEND COMPILED_SHADERS "${CMAKE_BINARY_DIR}/src/shaders/${shader_name}.spv")
        endif()
    endif()

endforeach()

# 4. Embed Every Shader found/compiled
foreach(shader_spv ${COMPILED_SHADERS})
    embed_resource(mm_rec_eigen ${shader_spv})
endforeach()

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
    src/cli/cmd_server.cpp
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

# Thread Safety Test
add_executable(test_pool_thread_safety tests/test_pool_thread_safety.cpp)
target_link_libraries(test_pool_thread_safety mm_rec_eigen)
add_test(NAME test_pool_thread_safety COMMAND test_pool_thread_safety)

# Config verification test
add_executable(test_config_driven_server tests/test_config_driven_server.cpp)
target_link_libraries(test_config_driven_server mm_rec_eigen)
add_test(NAME test_config_driven_server COMMAND test_config_driven_server)

# REAL DashboardManager Test
add_executable(test_real_dashboard tests/test_real_dashboard.cpp)
target_link_libraries(test_real_dashboard mm_rec_eigen)
add_test(NAME test_real_dashboard COMMAND test_real_dashboard)

add_executable(test_system_optimizer tests/test_system_optimizer.cpp)
target_link_libraries(test_system_optimizer mm_rec_eigen)

# Hardware Benchmark
add_executable(benchmark_hardware tests/benchmark_hardware.cpp)
target_link_libraries(benchmark_hardware mm_rec_eigen)

add_executable(benchmark_shader_opt tests/benchmark_shader_opt.cpp)
target_link_libraries(benchmark_shader_opt mm_rec_eigen)

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

# End-to-End Integration Test
add_executable(test_e2e_server tests/test_e2e_server.cpp)
target_link_libraries(test_e2e_server mm_rec_eigen)
add_test(NAME test_e2e_server COMMAND test_e2e_server)

# add_executable(test_vulkan_memory tests/test_vulkan_memory.cpp)
# target_link_libraries(test_vulkan_memory ${CMAKE_DL_LIBS})
# add_test(NAME test_vulkan_memory COMMAND test_vulkan_memory)
# Legacy targets removed (mm_rec_train, demo_training_cpp, inference_demo)
# Consolidated into mm_rec CLI





# ==========================================
# SHADER DEPLOYMENT
# ==========================================
# Copy shaders to the build directory structure expected by the app (src/shaders/*.spv)
# This ensures ./benchmark_hardware can find "src/shaders/matmul.spv" relative to CWD if run from build root,
# or we can adjust CWD.
# ACTUALLY: Let's copy to ${CMAKE_BINARY_DIR}/src/shaders so existing relative paths work.

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/src/shaders)
file(GLOB SHADER_FILES "${CMAKE_SOURCE_DIR}/src/shaders/*.spv")
file(COPY ${SHADER_FILES} DESTINATION ${CMAKE_BINARY_DIR}/src/shaders)

message(STATUS "✅ Deployed ${SHADER_FILES} to build dir.")

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
