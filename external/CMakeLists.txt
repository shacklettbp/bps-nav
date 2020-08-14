add_subdirectory(v4r EXCLUDE_FROM_ALL)
add_subdirectory(pybind11)
add_subdirectory(simdjson EXCLUDE_FROM_ALL)

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
