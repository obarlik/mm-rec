# Test executable macro
macro(add_mm_rec_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} mm_rec_lib ${TORCH_LIBRARIES})
    add_test(NAME ${test_name} COMMAND ${test_name})
endmacro()

# Core tests
add_mm_rec_test(test_gated_memory)
add_mm_rec_test(test_log_sum_exp)

# Layer tests
add_mm_rec_test(test_uboo)
add_mm_rec_test(test_per_layer_state)

# Model tests
add_mm_rec_test(test_session)
add_mm_rec_test(test_mm_rec_model)

# Benchmark (not a test, just builds)
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark mm_rec_lib ${TORCH_LIBRARIES})
