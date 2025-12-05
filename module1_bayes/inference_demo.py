# module1_bayes/inference_demo.py

from typing import Dict

from common.scenario import create_base_scenario
from common.constants import Weather
from common.entities import Accident
from .inference_engine import EmergencyBayesInference


def accident_to_evidence(accident: Accident, traffic_delay_level: str) -> Dict[str, str]:
    """
    Convert an Accident object + traffic delay level
    into BN evidence dictionary.
    """
    # Speed bucket
    if accident.estimated_speed_kmph is None:
        speed_state = "medium"
    else:
        v = accident.estimated_speed_kmph
        if v < 40:
            speed_state = "low"
        elif v <= 70:
            speed_state = "medium"
        else:
            speed_state = "high"

    # Vehicle type: use first vehicle in list
    if not accident.vehicle_types:
        vehicle_state = "car"
    else:
        vehicle_state = accident.vehicle_types[0].value  # e.g. "bike"

    # Weather
    weather_state = "rain" if accident.weather == Weather.RAIN else "clear"

    # Caller urgency
    caller_state = accident.caller_urgency or "tense"

    return {
        "ImpactSpeed": speed_state,
        "VehicleType": vehicle_state,
        "Weather": weather_state,
        "CallerUrgency": caller_state,
        "TrafficDelayProb": traffic_delay_level,
    }


def run_demo_for_acc1():
    scenario = create_base_scenario()
    accidents = scenario["accidents"]

    # Find ACC1 (high-speed bike crash near Jaydev Vihar flyover)
    acc1 = next(a for a in accidents if a.id == "ACC1")

    # High delay due to rain + diversions
    evidence = accident_to_evidence(acc1, traffic_delay_level="high")

    engine = EmergencyBayesInference()
    posteriors = engine.infer_severity_and_risk(evidence)

    print("=== Module 1 Demo: Bayesian Severity & Future Risk ===")
    print(f"Accident ID: {acc1.id}")
    print(f"Description: {acc1.description}")
    print(f"Location node: {acc1.location_node}")
    print(f"Evidence used: {evidence}")
    print("\nPosterior distributions:")
    for var, dist in posteriors.items():
        print(f"  {var}:")
        for state, prob in dist.items():
            print(f"    P({var} = {state} | evidence) = {prob:.3f}")

    # Store back into the accident object for later modules if needed
    acc1.severity_distribution = posteriors["CurrentSeverity"]
    acc1.future_risk_distribution = posteriors["FutureRisk"]

    most_probable_severity = max(posteriors["CurrentSeverity"], key=posteriors["CurrentSeverity"].get)
    most_probable_risk = max(posteriors["FutureRisk"], key=posteriors["FutureRisk"].get)

    print("\nMost probable severity:", most_probable_severity)
    print("Most probable future risk:", most_probable_risk)


if __name__ == "__main__":
    run_demo_for_acc1()
