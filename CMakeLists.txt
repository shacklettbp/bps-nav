cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

set(NAME ddppo-fastrollout)
project(${NAME} LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 17)

add_subdirectory(external)

pybind11_add_module(ddppo_fastrollout fastrollout.cpp)
add_dependencies(ddppo_fastrollout v4r_headless habitat_sim_geodesic)
target_link_libraries(ddppo_fastrollout PRIVATE v4r_headless habitat_sim_geodesic)
