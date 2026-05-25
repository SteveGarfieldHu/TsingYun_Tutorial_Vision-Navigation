"""Costmap generation: obstacle inflation and lidar-based dynamic costmap."""

from __future__ import annotations

from typing import Tuple

import numpy as np

_static_costmap_cache = None


def compute_costmap(
    static_map: np.ndarray,
) -> np.ndarray:
    """
    Build the global costmap by inflating static obstacles.

    Parameters
    ----------
    static_map : np.ndarray, shape (rows, cols), dtype int8
        0 = free cell, 1 = obstacle cell.

    Returns
    -------
    costmap : np.ndarray, shape (rows, cols), dtype uint8
        Per-cell cost in [0, 255]:
        - obstacle cells get the maximum lethal value, so the planner
          treats them as impassable.
        - free cells near an obstacle get a non-zero cost that decays with
          distance, creating a "buffer" so the planned path keeps clear of
          walls instead of grazing them.
        - free cells far from any obstacle get cost 0.

    Notes
    -----
    - The classical recipe: compute the Euclidean distance from each free cell
      to the nearest obstacle (`scipy.ndimage.distance_transform_edt` does this
      in one call), then map distance → cost so that distance 0 is lethal and
      cost falls off smoothly out to some `inflation_radius`. Beyond that
      radius, cost should be 0.
    - The shape of the decay (linear, exponential, ...) and the magnitude of
      the inflation radius are tuning knobs. Pick something that visibly biases
      the path away from walls without making narrow passages impassable. The
      inflation radius that is too large will also cause the robot to take a
      longer route, wasting time.
    """
    # TODO: Implement a function to compute a costmap from the static map by inflating obstacles.
    LETHAL_COST: int = 254
    INFLATION_RADIUS: float = 3.0

    # Obstacle coordinates
    obs_r, obs_c = np.where(static_map > 0)
    if obs_r.size == 0:
        return np.zeros_like(static_map, dtype=np.uint8)

    rows, cols = static_map.shape
    r_grid, c_grid = np.mgrid[0:rows, 0:cols]

    # Min Euclidean distance from each cell to nearest obstacle (chunked to save memory)
    min_dist_sq = np.full((rows, cols), np.inf)
    chunk = 500
    for i in range(0, len(obs_r), chunk):
        dr = r_grid[:, :, np.newaxis] - obs_r[i : i + chunk]
        dc = c_grid[:, :, np.newaxis] - obs_c[i : i + chunk]
        d_sq = dr * dr + dc * dc
        np.minimum(min_dist_sq, d_sq.min(axis=2), out=min_dist_sq)
    dist = np.sqrt(min_dist_sq)

    # Build costmap
    costmap = np.zeros((rows, cols), dtype=np.uint8)
    costmap[static_map > 0] = LETHAL_COST

    mask = (static_map == 0) & (dist < INFLATION_RADIUS)
    costmap[mask] = (LETHAL_COST * (1.0 - dist[mask] / INFLATION_RADIUS)).round().astype(np.uint8)

    return costmap


def update_local_costmap(
    static_map: np.ndarray,
    robot_pos: Tuple[float, float],
    lidar_scan: np.ndarray,
    lidar_range: float,
    lidar_num_rays: int,
) -> np.ndarray:
    """
    Produce the per-frame costmap by adding a dynamic layer on top of the
    static inflation.

    Parameters
    ----------
    static_map : np.ndarray, shape (rows, cols), dtype int8
        The same static map passed to `compute_costmap`. Re-inflate it (or
        cache the result) to get the static layer.
    robot_pos : Tuple[float, float], (x, y)
        Current robot position in world (grid-unit) coordinates. Lidar rays
        originate from this point.
    lidar_scan : np.ndarray, shape (lidar_num_rays,)
        Hit distance for each ray, in grid units. A value equal to `lidar_range`
        means the ray did not hit anything within range.
    lidar_range : float
        Maximum sensing distance of the lidar, in grid units.
    lidar_num_rays : int
        Number of rays in the scan; the i-th ray is at angle
        `2*pi * i / lidar_num_rays` measured from the +x axis.

    Returns
    -------
    costmap : np.ndarray, shape (rows, cols), dtype uint8
        Static-inflation layer merged with a dynamic layer that marks lidar
        hits as lethal and inflates them with a (smaller) buffer. Use a
        per-cell `max` to combine the two layers so the most conservative
        cost wins.

    Notes
    -----
    - Convert each ray hit `(angle_i, lidar_scan[i])` into a world point
      `(x + d*cos(a), y + d*sin(a))`, then to a grid cell. Mark that cell
      lethal and inflate it.
    - Skip rays where `lidar_scan[i] >= lidar_range` (no hit).
    - Optional but useful: skip hits that land on a cell that is *already*
      a static obstacle; otherwise the lidar's view of a wall keeps
      re-inflating the same area.
    """
    # TODO: Implement a function to update the global costmap with a local dynamic layer based on the lidar scan.
    # --- static layer (cached) ---
    global _static_costmap_cache
    map_id = id(static_map)
    if _static_costmap_cache is None or _static_costmap_cache[0] != map_id:
        _static_costmap_cache = (map_id, compute_costmap(static_map))
    static_costmap = _static_costmap_cache[1]

    # --- dynamic layer: lidar hits ---
    rows, cols = static_map.shape
    x, y = robot_pos

    DYNAMIC_LETHAL: int = 254
    DYNAMIC_RADIUS: float = 2.0

    hit_r, hit_c = [], []
    for i in range(lidar_num_rays):
        d = lidar_scan[i]
        if d >= lidar_range:
            continue
        angle = 2.0 * np.pi * i / lidar_num_rays
        hx = x + d * np.cos(angle)
        hy = y + d * np.sin(angle)
        hr, hc = int(round(hy)), int(round(hx))
        if 0 <= hr < rows and 0 <= hc < cols and static_map[hr, hc] == 0:
            hit_r.append(hr)
            hit_c.append(hc)

    if not hit_r:
        return static_costmap.copy()

    # Inflate dynamic hits
    hit_r = np.array(hit_r)
    hit_c = np.array(hit_c)
    r_grid, c_grid = np.mgrid[0:rows, 0:cols]
    min_dist_sq = np.full((rows, cols), np.inf)
    dr = r_grid[:, :, np.newaxis] - hit_r
    dc = c_grid[:, :, np.newaxis] - hit_c
    d_sq = dr * dr + dc * dc
    np.minimum(min_dist_sq, d_sq.min(axis=2), out=min_dist_sq)
    dist = np.sqrt(min_dist_sq)

    dynamic_costmap = np.zeros((rows, cols), dtype=np.uint8)
    for r, c in zip(hit_r, hit_c):
        dynamic_costmap[r, c] = DYNAMIC_LETHAL
    mask = (dynamic_costmap == 0) & (dist < DYNAMIC_RADIUS)
    dynamic_costmap[mask] = (DYNAMIC_LETHAL * (1.0 - dist[mask] / DYNAMIC_RADIUS)).round().astype(np.uint8)

    return np.maximum(static_costmap, dynamic_costmap)
