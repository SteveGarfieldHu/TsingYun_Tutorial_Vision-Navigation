"""Global path planner: A* search on a costmap grid."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def global_plan(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    costmap: np.ndarray,
) -> List[Tuple[float, float]]:
    """
    Run path search over `costmap` to find a path from `start` to `goal`.

    Parameters
    ----------
    start : Tuple[float, float], (x, y)
        Start position in world (grid-unit) coordinates. `costmap[int(y), int(x)]`
        is the cell containing this point.
    goal : Tuple[float, float], (x, y)
        Goal position in the same coordinate system.
    costmap : np.ndarray, shape (rows, cols), dtype uint8
        Per-cell traversal cost. Cells with large cost are treated as impassable
        (lethal). Otherwise the cell's cost is added to the step cost so the
        planner is biased away from inflated/dangerous areas.

    Returns
    -------
    path : List[Tuple[float, float]], list of (x, y) waypoints from start to goal.
        World-coordinate waypoints from start to goal, inclusive of both ends.
        Returns [] if no path exists or if start/goal lie inside a lethal cell.

    Notes
    -----
    - Use 8-connectivity (N/S/E/W + 4 diagonals). Step cost between adjacent
      cells should be `dist + cell_cost`, where `dist` is 1.0 for cardinal moves
      and sqrt(2) for diagonals.
    - Use either a shortest path algorithm (like Dijkstra) or a heuristic search
      algorithm (like A*). If using A*, a good heuristic is the octile distance
      (diagonal distance) or Euclidean distance.
    """
    # TODO: Implement path search on the costmap grid to find a path from start to goal.
    rows, cols = costmap.shape
    LETHAL = 100  # treat cells with cost >= this as impassable (robot radius ≈ 1.6)

    # Convert start / goal to grid cells
    sc, sr = int(round(start[0])), int(round(start[1]))
    gc, gr = int(round(goal[0])), int(round(goal[1]))

    # Validate start and goal
    if not (0 <= sr < rows and 0 <= sc < cols):
        return []
    if not (0 <= gr < rows and 0 <= gc < cols):
        return []
    if costmap[sr, sc] >= LETHAL or costmap[gr, gc] >= LETHAL:
        return []

    # 8-connectivity: (dr, dc), cardinal dist=1, diagonal dist=sqrt(2)
    SQRT2 = 2.0 ** 0.5
    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
    ]

    # Octile distance heuristic
    def heuristic(r, c):
        dr = abs(r - gr)
        dc = abs(c - gc)
        return max(dr, dc) + (SQRT2 - 1) * min(dr, dc)

    # A* search
    import heapq
    open_heap = [(heuristic(sr, sc), 0.0, sr, sc)]
    g_cost = np.full((rows, cols), np.inf)
    g_cost[sr, sc] = 0.0
    came_from = {}  # (r, c) -> (prev_r, prev_c)
    closed = set()

    while open_heap:
        f, g, r, c = heapq.heappop(open_heap)

        if (r, c) in closed:
            continue
        closed.add((r, c))

        if r == gr and c == gc:
            # Reconstruct path
            path = []
            cr, cc = gr, gc
            while (cr, cc) != (sr, sc):
                path.append((float(cc), float(cr)))
                cr, cc = came_from[(cr, cc)]
            path.append((float(sc), float(sr)))
            path.reverse()
            return path

        for dr, dc, step_dist in neighbors:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if (nr, nc) in closed:
                continue
            cell_cost = costmap[nr, nc]
            if cell_cost >= LETHAL:
                continue
            new_g = g + step_dist + cell_cost
            if new_g < g_cost[nr, nc]:
                g_cost[nr, nc] = new_g
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(open_heap, (new_g + heuristic(nr, nc), new_g, nr, nc))

    return []  # no path found
