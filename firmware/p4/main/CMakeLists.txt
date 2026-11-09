idf_component_register(
    SRCS "main.c" "kernels.c" "pie.S" "bench_mem.c"
    INCLUDE_DIRS "."
    REQUIRES esp_timer esp_psram esp_hw_support)
target_compile_options(${COMPONENT_LIB} PRIVATE -O3)
