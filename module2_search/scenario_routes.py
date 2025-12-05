# module2_search/scenario_routes.py

from typing import Tuple, List

import networkx as nx

from common.constants import Weather
from common.road_network import build_base_road_network, apply_weather_and_diversions, get_travel_time
from .search_algorithms import (
    uniform_cost_search,
    a_star_search,
    build_time_heuristic_from_graph,
)


def print_route(label: str, path: List[str], time_min: float) -> None:
    print(f"{label}")
    print("  Path:", " -> ".join(path))
    print(f"  Total estimated travel time: {time_min:.2f} minutes\n")


def build_now_and_future_graphs() -> Tuple[nx.Graph, nx.Graph]:
    """
    Build two graph configurations:
    - G_now: rainy but no stadium diversions yet (edges open, moderate congestion).
    - G_future: rainy, heavier congestion, and stadium edges blocked due to diversions.
    """

    # Base graph (no weather, no diversions)
    base_G_now = build_base_road_network()
    base_G_future = build_base_road_network()

    # "Now" graph: rain slowdown, moderate congestion, no blocks yet
    G_now = apply_weather_and_diversions(
        base_G_now,
        weather=Weather.RAIN,
        blocked_edges=(),       # assume diversions not formally active yet
        congestion_factor=1.2,  # some congestion but not peak
    )

    # "Future" graph: we expect full diversion activation near stadium
    blocked_edges = (
        ("stadium", "jaydev_vihar_flyover"),
        ("stadium", "rupali_square"),
    )

    G_future = apply_weather_and_diversions(
        base_G_future,
        weather=Weather.RAIN,
        blocked_edges=blocked_edges,
        congestion_factor=1.5,  # heavier congestion as crowd leaves
    )

    return G_now, G_future


def run_module2_demo():
    """
    Demonstration for Module 2:
    - One ambulance starts at H1 (trauma hospital).
    - Accident ACC3 (SUV rollover) is near 'airport_approach'.
    - We compare:
        1) Naive UCS on current graph (G_now)
        2) Predictive A* on future graph (G_future)
    """

    G_now, G_future = build_now_and_future_graphs()

    start_node = "hospital_h1"
    goal_node = "airport_approach"

    print("=== Module 2 Demo: Predictive Search-Based Ambulance Route Optimization ===\n")
    print("Start (ambulance):", start_node)
    print("Goal (accident location):", goal_node)
    print()

    # 1) Uninformed Uniform Cost Search on "now" graph (no stadium blocks yet)
    ucs_path, ucs_cost = uniform_cost_search(G_now, start=start_node, goal=goal_node)
    print_route("1) UCS on current conditions (rain, moderate congestion, no active diversions):",
                ucs_path, ucs_cost)

    # 2) Informed A* Search on "future" graph (stadium diversions + heavier congestion)
    heuristic = build_time_heuristic_from_graph(
        G_future,
        goal=goal_node,
        weight_attr="effective_time_min",
    )
    astar_path, astar_cost = a_star_search(
        G_future,
        start=start_node,
        goal=goal_node,
        heuristic=heuristic,
    )
    print_route("2) A* on predicted future conditions (rain, heavy congestion, stadium diversions):",
                astar_path, astar_cost)

    # For clarity: compute naive UCS path cost under future graph as well (optional)
    # (What if we stubbornly kept using the UCS-found path when diversions activate?)
    try:
        # Re-compute travel time of UCS path under future graph (not re-run search, just sum costs)
        ucs_path_future_cost = get_travel_time(G_future, tuple(ucs_path))
        print("Re-evaluating UCS path under future conditions:")
        print("  Path:", " -> ".join(ucs_path))
        print(f"  Travel time if diversions activate: {ucs_path_future_cost:.2f} minutes\n")
    except Exception as e:
        print("Could not re-evaluate UCS path under future graph (possibly broken due to blocks).")
        print("Reason:", e)

    # Short textual summary
    print("Summary:")
    print("  - UCS on the 'now' graph tends to prefer the currently fastest route,")
    print("    which may pass through stadium-connected roads that are expected to be blocked.")
    print("  - A* on the 'future' graph avoids these high-risk edges and finds a route")
    print("    that is slightly longer in the present, but more reliable once diversions start.")


if __name__ == "__main__":
    run_module2_demo()
