# habitat-sim-geodesic

## Install

```bash
pip install git+https://github.com/erikwijmans/habitat-sim-geodesic.git#egg=habitat_sim_geodesic
```


# Usage


```python
from habitat_sim_geodesic import compute_geodesic_distance

# Now you can compute the geodesic between two points!
compute_geodesic_distance(
    "17DRP5sb8fy",
    np.array([3.76632, 0.072447, 0.30173]),
    np.array([0.403801, 0.072447, -0.242499]),
)

# The docstring on this method describes the arguments
# python -c "from habitat_sim_geodesic import compute_geodesic_distance; help(compute_geodesic_distance)"
# will show it

# The points are assumed to already be in habitat's coordinate frame,
# if they are in the mp3d coordinate frame instead, you can use
# the mp3d_to_habitat helper function
from habitat_sim_geodesic import mp3d_to_habitat

pt_mp3d = ...
pt_habitat = mp3d_to_habitat(pt_mp3d)

```
