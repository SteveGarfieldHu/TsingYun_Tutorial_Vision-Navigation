"""Local planner: Pure Pursuit controller."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def local_plan(
    current_pose: Tuple[float, float],
    max_speed: float,
    max_accel: float,
    global_path: List[Tuple[float, float]],
    costmap: np.ndarray = None,
) -> Tuple[float, float]:
    """
    Convert the next chunk of the global path into a velocity command.

    Parameters
    ----------
    current_pose : Tuple[float, float], (x, y)
        Current robot position in world (grid-unit) coordinates.
    max_speed : float
        Maximum allowed speed magnitude (grid units / second). The returned
        command vector should not exceed this length.
    max_accel : float
        Maximum allowed acceleration. You may ignore this if the world's
        `step()` already enforces a ramp; otherwise use it to compute a
        feasible command from the current velocity.
    global_path : List[Tuple[float, float]], list of (x, y) waypoints from start to goal
        Waypoints from the global planner, ordered from current pose to goal.
        May be empty if no path was found — in that case return `(0.0, 0.0)`.

    Returns
    -------
    cmd_vx, cmd_vy : float, float
        Desired world-frame velocity in grid units per second. The world step
        will clip this to `max_speed` and ramp toward it at `max_accel`, so
        returning a "pointing at the look-ahead" vector scaled to `max_speed`
        is usually the right move.

    Notes
    -----
    - Pure Pursuit recipe:
        1. Find the look-ahead point on `global_path`: walk forward from the
           closest waypoint to `current_pose` until the cumulative distance
           exceeds a look-ahead radius `Ld` (a tuning constant, e.g. 1.5-2.5
           grid units). If you reach the last waypoint first, use it.
        2. The command direction is `(look_ahead - current_pose)`, normalized.
        3. The command speed is `max_speed` (or a slowed value if the
           remaining path length is short, to ease into the goal).
    - Optional: More complex local programming methods (such as Dynamic
      Window Approach) can be used, or more complex model prediction methods
      (such as MPPI) can be tried.
    """
    # TODO: Implement Pure Pursuit controller.
    if not global_path:
        return 0.0, 0.0

    x, y = current_pose

    # 1. Find the closest waypoint on the path
    dists_sq = np.array([(px - x) ** 2 + (py - y) ** 2 for px, py in global_path])
    closest_idx = int(np.argmin(dists_sq))

    # 2. Walk forward from closest waypoint to find the look-ahead point
    Ld = 2.0  # look-ahead radius (grid units)
    look_ahead = global_path[-1]  # default to goal
    cum_dist = 0.0
    for i in range(closest_idx, len(global_path) - 1):
        x1, y1 = global_path[i]
        x2, y2 = global_path[i + 1]
        seg_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if cum_dist + seg_len >= Ld:
            frac = (Ld - cum_dist) / seg_len if seg_len > 1e-9 else 0.0
            look_ahead = (x1 + frac * (x2 - x1), y1 + frac * (y2 - y1))
            break
        cum_dist += seg_len

    # 3. Direction toward look-ahead point
    dx = look_ahead[0] - x
    dy = look_ahead[1] - y
    dist = (dx * dx + dy * dy) ** 0.5
    if dist < 1e-6:
        return 0.0, 0.0

    # 4. Speed: full speed far from goal, slow down when near
    goal = global_path[-1]
    remaining = ((goal[0] - x) ** 2 + (goal[1] - y) ** 2) ** 0.5
    speed = min(max_speed, max(2.0, remaining))

    return dx / dist * speed, dy / dist * speed
