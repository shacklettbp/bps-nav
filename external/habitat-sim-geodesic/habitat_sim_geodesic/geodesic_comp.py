import os.path as osp

import cppimport.import_hook

from habitat_sim_geodesic.bindings import PathFinder, ShortestPath


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class GeodesicDistanceComputer(metaclass=Singleton):
    def __init__(self):
        self._pathfinders = {}

    def _get_pathfinder(self, scene_id) -> PathFinder:
        scene_name = osp.splitext(osp.basename(scene_id))[0]

        if scene_name not in self._pathfinders:
            navmesh = osp.join(
                osp.dirname(__file__), "navmeshes", scene_name + ".navmesh"
            )
            pf = PathFinder()
            pf.load_nav_mesh(navmesh)

            if not pf.is_loaded:
                raise RuntimeError(
                    f"Could not find navmesh to load for scene_id '{scene_id}'\n"
                    + f"Tried to load from '{navmesh}'"
                )

            self._pathfinders[scene_name] = pf

        return self._pathfinders[scene_name]

    def compute_distance(self, scene_id, start_pt, end_pt):
        path = ShortestPath()

        self._get_pathfinder(scene_id).find_path(path)

        return path.geodesic_distance


def compute_geodesic_distance(scene_id, start_pt, end_pt):
    r"""Comptues the geodesic distance between two points

    :param scene_id: The ID of the scene to compute the distance in.
        Assumes scene_id is a file name in the format /anything/<scene_name>.<ext>
        where <scene_name> is the hash name of an mp3d scene. i.e. `17DRP5sb8fy`

        Note that providing the hash name directly will also work

    :param start_pt: The starting point.  Assumed to already be in habitat's coordinate frame
    :param end_pt: The ending point. Assumed to already be in habitat's coordinate frame

    :return: The geodesic distance between the two points
    """
    return GeodesicDistanceComputer().compute_distance(scene_id, start_pt, end_pt)
