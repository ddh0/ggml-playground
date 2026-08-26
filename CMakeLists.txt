cmake_minimum_required(VERSION 3.14)
project(ggml-playground C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(GGML_METAL "use Metal backend on Apple Silicon" ON)

add_subdirectory(vendor/ggml)

add_executable(ggml-playground
    src/ggml-playground.cpp
    src/logger.cpp
)

target_include_directories(ggml-playground PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ggml-playground PRIVATE
    ggml
)
