cmake_minimum_required(VERSION 3.14)
project(ggml-playground C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

add_subdirectory(ggml)

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
