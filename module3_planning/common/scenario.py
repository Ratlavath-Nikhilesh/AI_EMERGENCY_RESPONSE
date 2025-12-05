# common/scenario.py
from datetime import datetime, timedelta
from typing import List, Dict, Any

from .constants import (
    Weather,
    AccidentSeverity,
    VehicleType,
    HospitalCTStatus,
)
from .entities import Accident, Hospital, Ambulance
from .road_network import build_base_road_network, apply_weather_and_diversions


def create_base_scenario() -> Dict[str, Any]:
    """
    Build the fixed scenario described in the project statement:
    - Rainy Saturday, 8:15 PM
    - 3 accidents
    - 2 ambulances
    - 2 hospitals (H1 trauma, H2 closer but CT offline initially)
    - Road network with rain slowdown and diversions near stadium
    """

    # 1. Global context
    scenario_time = datetime(
        year=2025, month=7, day=12, hour=20, minute=15  # arbitrary Saturday date
    )
    weather = Weather.RAIN

    # 2. Road network with weather & diversions
    G = build_base_road_network()

    # "Two major diversion roads near the stadium are blocked"
    blocked_edges = (
        ("stadium", "jaydev_vihar_flyover"),
        ("stadium", "rupali_square"),
    )

    # Example congestion factor (1.5x slower due to match crowd)
    G = apply_weather_and_diversions(
        G,
        weather=weather,
        blocked_edges=blocked_edges,
        congestion_factor=1.5,
    )

    # 3. Hospitals
    h1 = Hospital(
        id="H1",
        name="H1 Trauma Hospital",
        node="hospital_h1",
        is_trauma_center=True,
        base_distance_km_from_stadium=11.0,
        ct_status=HospitalCTStatus.AVAILABLE,  # trauma hospital, assume CT working
        capacity_beds=40,
        current_load=30,
    )

    h2_ct_offline_until = scenario_time + timedelta(minutes=40)

    h2 = Hospital(
        id="H2",
        name="H2 General Hospital",
        node="hospital_h2",
        is_trauma_center=False,
        base_distance_km_from_stadium=6.0,
        ct_status=HospitalCTStatus.OFFLINE,
        ct_offline_until=h2_ct_offline_until,
        capacity_beds=30,
        current_load=20,
    )

    hospitals: List[Hospital] = [h1, h2]

    # 4. Ambulances
    # You can adjust starting positions later if needed.
    a1 = Ambulance(
        id="A1",
        current_node="hospital_h1",
        is_available=True,
    )

    a2 = Ambulance(
        id="A2",
        current_node="hospital_h2",
        is_available=True,
    )

    ambulances: List[Ambulance] = [a1, a2]

    # 5. Accidents (within next 25 minutes)
    # (i) high-speed bike collision near the Jaydev Vihar flyover
    acc1 = Accident(
        id="ACC1",
        location_node="jaydev_vihar_flyover",
        description="High-speed bike collision near Jaydev Vihar flyover",
        time_reported=scenario_time + timedelta(minutes=5),
        vehicle_types=[VehicleType.BIKE],
        estimated_speed_kmph=80.0,
        caller_urgency="panicked",
        weather=weather,
    )

    # (ii) car–auto crash at Rupali Square
    acc2 = Accident(
        id="ACC2",
        location_node="rupali_square",
        description="Car–auto crash at Rupali Square",
        time_reported=scenario_time + timedelta(minutes=12),
        vehicle_types=[VehicleType.CAR, VehicleType.AUTO],
        estimated_speed_kmph=50.0,
        caller_urgency="tense",
        weather=weather,
    )

    # (iii) SUV rollover near the Airport approach road with airbags deployed
    acc3 = Accident(
        id="ACC3",
        location_node="airport_approach",
        description="SUV rollover near Airport approach road with airbags deployed",
        time_reported=scenario_time + timedelta(minutes=20),
        vehicle_types=[VehicleType.SUV],
        estimated_speed_kmph=70.0,
        caller_urgency="tense",
        weather=weather,
    )

    accidents: List[Accident] = [acc1, acc2, acc3]

    scenario = {
        "time": scenario_time,
        "weather": weather,
        "graph": G,
        "hospitals": hospitals,
        "ambulances": ambulances,
        "accidents": accidents,
        "blocked_edges": blocked_edges,
    }

    return scenario


if __name__ == "__main__":
    # Quick sanity check printout
    scenario = create_base_scenario()
    print("Scenario time:", scenario["time"])
    print("Weather:", scenario["weather"])
    print("Hospitals:", [h.name for h in scenario["hospitals"]])
    print("Ambulances:", [a.id for a in scenario["ambulances"]])
    print("Accidents:", [a.id + "@" + a.location_node for a in scenario["accidents"]])
