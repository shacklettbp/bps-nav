 find_program(PYTHON_EXECUTABLE
    # macOS still defaults to `python` being Python 2, so look for `python3`
    # first
    NAMES python3 python
    PATHS ENV PATH   # look in the PATH environment variable
    NO_DEFAULT_PATH  # do not look anywhere else...
  )

# Let the Find module do proper version checks on what we found (it uses the
# same PYTHON_EXECUTABLE variable, will pick it up from the cache)
find_package(PythonInterp 3.6 REQUIRED)

message(STATUS "Bindings being generated for python at ${PYTHON_EXECUTABLE}")

add_subdirectory(bps3D EXCLUDE_FROM_ALL)
add_subdirectory(pybind11)

if (NOT TARGET simdjson)
    set(SIMDJSON_JUST_LIBRARY ON CACHE INTERNAL "")
    set(SIMDJSON_BUILD_STATIC ON CACHE INTERNAL "")
    add_subdirectory(simdjson EXCLUDE_FROM_ALL)
endif()

add_library(habitat_sim_geodesic SHARED
    habitat-sim-geodesic/habitat_sim_geodesic/csrc/recastnavigation-master/Detour/Source/DetourNode.cpp
    habitat-sim-geodesic/habitat_sim_geodesic/csrc/recastnavigation-master/Detour/Source/DetourAlloc.cpp
    habitat-sim-geodesic/habitat_sim_geodesic/csrc/recastnavigation-master/Detour/Source/DetourAssert.cpp
    habitat-sim-geodesic/habitat_sim_geodesic/csrc/recastnavigation-master/Detour/Source/DetourCommon.cpp
    habitat-sim-geodesic/habitat_sim_geodesic/csrc/recastnavigation-master/Detour/Source/DetourNavMesh.cpp
    habitat-sim-geodesic/habitat_sim_geodesic/csrc/recastnavigation-master/Detour/Source/DetourNavMeshQuery.cpp
    habitat-sim-geodesic/habitat_sim_geodesic/csrc/recastnavigation-master/Detour/Source/DetourNavMeshBuilder.cpp
    habitat-sim-geodesic/habitat_sim_geodesic/csrc/PathFinder.cpp
    habitat-sim-geodesic/habitat_sim_geodesic/csrc/PathFinder.h
)

target_include_directories(habitat_sim_geodesic
    PRIVATE
        habitat-sim-geodesic/habitat_sim_geodesic/csrc/recastnavigation-master/Detour/Include
        habitat-sim-geodesic/habitat_sim_geodesic/csrc/eigen/Eigen
    PUBLIC
        habitat-sim-geodesic/habitat_sim_geodesic/csrc/eigen
        habitat-sim-geodesic/habitat_sim_geodesic/csrc)

target_compile_definitions(habitat_sim_geodesic PUBLIC DT_VIRTUAL_QUERYFILTER)

add_library(cpp20sync INTERFACE)
target_include_directories(cpp20sync INTERFACE cpp20sync)
