cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

set(NAME bps-sim)
project(${NAME} LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 17)

find_package(ZLIB REQUIRED)

add_subdirectory(external)

pybind11_add_module(bps_sim
    simulator.cpp)

target_compile_options(bps_sim PRIVATE -Wall -Wextra -Wshadow)

add_dependencies(bps_sim habitat_sim_geodesic preprocess)
target_link_libraries(bps_sim
    PRIVATE bps3D habitat_sim_geodesic ZLIB::ZLIB simdjson cpp20sync)
