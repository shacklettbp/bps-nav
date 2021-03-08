/*cppimport
<%
setup_pybind11(cfg)

cfg['compiler_args'] = ['-std=c++14', '-O2', '-g', '-DDT_VIRTUAL_QUERYFILTER']

cfg['sources'] = ['./csrc/recastnavigation-master/Detour/Source/DetourNode.cpp',
                  './csrc/recastnavigation-master/Detour/Source/DetourAlloc.cpp',
                  './csrc/recastnavigation-master/Detour/Source/DetourAssert.cpp',
                  './csrc/recastnavigation-master/Detour/Source/DetourCommon.cpp',
                  './csrc/recastnavigation-master/Detour/Source/DetourNavMesh.cpp',
                  './csrc/recastnavigation-master/Detour/Source/DetourNavMeshQuery.cpp',
                  './csrc/recastnavigation-master/Detour/Source/DetourNavMeshBuilder.cpp',
                  './csrc/PathFinder.cpp',]

cfg['include_dirs'] = ['./csrc/recastnavigation-master/Detour/Include',
                       './csrc/eigen',
                       './csrc/eigen/Eigen',
                       './csrc',]

cfg['dependencies'] = ['./csrc/PathFinder.h']
%>
*/
#include <pybind11/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "PathFinder.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace esp;
using namespace esp::nav;

PYBIND11_MODULE(bindings, m) {
  py::class_<ShortestPath, ShortestPath::ptr>(m, "ShortestPath")
      .def(py::init(&ShortestPath::create<>))
      .def_readwrite("requested_start", &ShortestPath::requestedStart)
      .def_readwrite("requested_end", &ShortestPath::requestedEnd)
      .def_readwrite("points", &ShortestPath::points)
      .def_readwrite("geodesic_distance", &ShortestPath::geodesicDistance);

  py::class_<PathFinder, PathFinder::ptr>(m, "PathFinder")
      .def(py::init(&PathFinder::create<>))
      .def("get_bounds", &PathFinder::bounds)
      .def("seed", &PathFinder::seed)
      .def("get_topdown_view", &PathFinder::getTopDownView,
           R"(Returns the topdown view of the PathFinder's navmesh.)",
           "pixelsPerMeter"_a, "height"_a)
      .def("get_random_navigable_point", &PathFinder::getRandomNavigablePoint)
      .def("find_path", py::overload_cast<ShortestPath&>(&PathFinder::findPath),
           "path"_a)
      .def("try_step", &PathFinder::tryStep<vec3f>, "start"_a, "end"_a)
      .def("try_step_no_sliding", &PathFinder::tryStepNoSliding<vec3f>,
           "start"_a, "end"_a)
      .def("snap_point", &PathFinder::snapPoint<vec3f>)
      .def("island_radius", &PathFinder::islandRadius, "pt"_a)
      .def_property_readonly("is_loaded", &PathFinder::isLoaded)
      .def("load_nav_mesh", &PathFinder::loadNavMesh)
      .def("is_navigable", &PathFinder::isNavigable,
           R"(Checks to see if the agent can stand at the specified point.)",
           "pt"_a, "max_y_delta"_a = 0.5);
}
