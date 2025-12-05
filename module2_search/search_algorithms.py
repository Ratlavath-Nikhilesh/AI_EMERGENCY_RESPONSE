# module2_search/search_algorithms.py

from typing import Dict, List, Tuple, Callable, Optional
import heapq

import networkx as nx


def uniform_cost_search(
    G: nx.Graph,
    start: str,
    goal: str,
    weight_attr: str = "effective_time_min",
) -> Tuple[List[str], float]:
    """
    Uninformed Uniform Cost Search (Dijkstra-like) on the road network.

    Returns:
        path: list of node ids from start to goal
        cost: total travel time along that path
    """
    # (cost_so_far, current_node, path)
    frontier = [(0.0, start, [start])]
    best_cost: Dict[str, float] = {start: 0.0}

    while frontier:
        cost, node, path = heapq.heappop(frontier)

        if node == goal:
            return path, cost

        # If we already have a better cost for this node, skip
        if cost > best_cost.get(node, float("inf")):
            continue

        for neighbor in G.neighbors(node):
            edge_data = G[node][neighbor]
            # Skip blocked edges
            if edge_data.get("blocked", False):
                continue

            w = edge_data.get(weight_attr, float("inf"))
            if w == float("inf"):
                continue

            new_cost = cost + w
            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                heapq.heappush(frontier, (new_cost, neighbor, path + [neighbor]))

    raise ValueError(f"No path found from {start} to {goal} with UCS.")


def a_star_search(
    G: nx.Graph,
    start: str,
    goal: str,
    heuristic: Callable[[str], float],
    weight_attr: str = "effective_time_min",
) -> Tuple[List[str], float]:
    """
    A* search on the road network.

    heuristic(node) should return an estimate of remaining travel time
    from 'node' to 'goal' (in minutes). For admissibility, it should not
    overestimate the true minimal remaining cost under the same weights.

    Returns:
        path: list of node ids from start to goal
        cost: total travel time along that path
    """

    # (f = g + h, g, node, path)
    frontier = [(heuristic(start), 0.0, start, [start])]
    best_g: Dict[str, float] = {start: 0.0}

    while frontier:
        f, g, node, path = heapq.heappop(frontier)

        if node == goal:
            return path, g

        if g > best_g.get(node, float("inf")):
            continue

        for neighbor in G.neighbors(node):
            edge_data = G[node][neighbor]
            if edge_data.get("blocked", False):
                continue

            w = edge_data.get(weight_attr, float("inf"))
            if w == float("inf"):
                continue

            g_new = g + w
            if g_new < best_g.get(neighbor, float("inf")):
                best_g[neighbor] = g_new
                f_new = g_new + heuristic(neighbor)
                heapq.heappush(frontier, (f_new, g_new, neighbor, path + [neighbor]))

    raise ValueError(f"No path found from {start} to {goal} with A*.")


def build_time_heuristic_from_graph(
    G: nx.Graph,
    goal: str,
    weight_attr: str = "effective_time_min",
) -> Callable[[str], float]:
    """
    Build a heuristic h(node) = shortest travel time from node to goal
    in the given graph (under the specified weight attribute).

    This is effectively an 'ideal' heuristic: it never overestimates the
    true cost in G, so A* behaves like an informed Dijkstra.
    """

    # Compute shortest-path distances from all nodes TO the goal.
    # We run from goal as source for convenience.
    lengths = nx.single_source_dijkstra_path_length(
        G,
        source=goal,
        weight=weight_attr,
    )

    def h(node: str) -> float:
        return float(lengths.get(node, float("inf")))

    return h
