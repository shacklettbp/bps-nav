add_subdirectory(v4r EXCLUDE_FROM_ALL)
add_subdirectory(pybind11)

add_library(habitat_sim_geodesic 
    habitat_sim_geodesic/csrc/recastnavigation-master/Detour/Source/DetourNode.cpp
    habitat_sim_geodesic/csrc/recastnavigation-master/Detour/Source/DetourAlloc.cpp
    habitat_sim_geodesic/csrc/recastnavigation-master/Detour/Source/DetourAssert.cpp
    habitat_sim_geodesic/csrc/recastnavigation-master/Detour/Source/DetourCommon.cpp
    habitat_sim_geodesic/csrc/recastnavigation-master/Detour/Source/DetourNavMesh.cpp
    habitat_sim_geodesic/csrc/recastnavigation-master/Detour/Source/DetourNavMeshQuery.cpp
    habitat_sim_geodesic/csrc/recastnavigation-master/Detour/Source/DetourNavMeshBuilder.cpp
    habitat_sim_geodesic/csrc/PathFinder.cpp
    habitat_sim_geodesic/csrc/PathFinder.h
)

target_include_directories(habitat_sim_geodesic
    PRIVATE
        /habitat_sim_geodesic/csrc/recastnavigation-master/Detour/Include
        /habitat_sim_geodesic/csrc/eigen
        /habitat_sim_geodesic/csrc/eigen/Eigen
    PUBLIC
        habitat_sim_geodesic/csrc)

target_compile_definitions(habitat_sim_geodesic DT_VIRTUAL_QUERYFILTER)
