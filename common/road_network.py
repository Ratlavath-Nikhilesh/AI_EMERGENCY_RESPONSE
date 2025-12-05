# common/road_network.py

from typing import Tuple, Dict, Any
import networkx as nx

from .constants import Weather, RoadCondition, RAIN_SPEED_REDUCTION


def build_base_road_network() -> nx.Graph:
    """
    Build a simplified road network around the stadium / Jaydev Vihar / Rupali / Airport / hospitals.
    Edge weights are base travel times in minutes under normal conditions (no rain, no match traffic).
    """

    G = nx.Graph()

    # Nodes
    nodes = [
        "stadium",
        "jaydev_vihar_flyover",
        "rupali_square",
        "airport_approach",
        "hospital_h1",
        "hospital_h2",
        "nayapalli_chowk",
        "acharya_vihar",
        "khandagiri_chowk",
    ]
    for n in nodes:
        G.add_node(n)

    # Edges: (u, v, base_time_minutes)
    # Values chosen to create two realistic competing routes:
    # - A fast but risky route via stadium (likely to be blocked after match).
    # - A slightly longer but more robust route via Khandagiri / Nayapalli / Rupali.
    edges = [
        ("stadium", "jaydev_vihar_flyover", 4.0),
        ("stadium", "rupali_square", 6.0),
        ("stadium", "nayapalli_chowk", 8.0),

        ("jaydev_vihar_flyover", "acharya_vihar", 4.0),
        ("acharya_vihar", "nayapalli_chowk", 6.0),
        ("nayapalli_chowk", "khandagiri_chowk", 10.0),

        ("rupali_square", "nayapalli_chowk", 5.0),
        ("rupali_square", "hospital_h2", 6.0),

        ("airport_approach", "rupali_square", 8.0),
        ("airport_approach", "hospital_h2", 10.0),

        ("khandagiri_chowk", "hospital_h1", 12.0),
        ("jaydev_vihar_flyover", "hospital_h1", 10.0),
        ("stadium", "hospital_h2", 8.0),
    ]

    for u, v, t in edges:
        G.add_edge(
            u,
            v,
            base_time_min=t,
            effective_time_min=t,            # will be updated by weather/traffic
            condition=RoadCondition.NORMAL,
            blocked=False
        )

    return G


def apply_weather_and_diversions(
    G: nx.Graph,
    weather: Weather,
    blocked_edges: Tuple[Tuple[str, str], ...] = (),
    congestion_factor: float = 1.0,
) -> nx.Graph:
    """
    Modify edge attributes in-place based on weather, events (rainy, diversions near stadium),
    and overall congestion factor.

    effective_time_min = base_time_min * (1 + rain_slowdown) * congestion_factor
    Blocked edges are marked and given infinite effective time.
    """

    for u, v, data in G.edges(data=True):
        base_time = data.get("base_time_min", 1.0)

        # Start from base time
        effective = base_time

        # Rain reduces speed -> increases travel time
        if weather == Weather.RAIN:
            effective *= (1 + RAIN_SPEED_REDUCTION)  # e.g. 30% slower

        # Overall congestion multiplier (e.g., match crowd leaving)
        effective *= congestion_factor

        # Check if this edge is blocked due to diversions
        if (u, v) in blocked_edges or (v, u) in blocked_edges:
            data["blocked"] = True
            data["condition"] = RoadCondition.BLOCKED
            # Represent blocked by making time huge
            effective = float("inf")
        else:
            data["blocked"] = False

        data["effective_time_min"] = effective

    return G


def get_travel_time(G: nx.Graph, path: Tuple[str, ...]) -> float:
    """
    Compute total effective travel time for a path (sequence of nodes).
    """
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge_data: Dict[str, Any] = G[u][v]
        t = edge_data.get("effective_time_min", edge_data.get("base_time_min", 0.0))
        total += t
    return total
