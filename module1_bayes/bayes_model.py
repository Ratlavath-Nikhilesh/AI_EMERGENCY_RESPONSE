# module1_bayes/bayes_model.py

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import numpy as np


def build_emergency_bayes_model() -> BayesianModel:
    """
    Build and return the Bayesian Network model for accident severity and future risk.
    """

    # 1. Graph structure
    model = BayesianModel([
        ("ImpactSpeed", "CurrentSeverity"),
        ("VehicleType", "CurrentSeverity"),
        ("CallerUrgency", "CurrentSeverity"),

        ("Weather", "TrafficDelayProb"),
        ("ImpactSpeed", "TrafficDelayProb"),

        ("CurrentSeverity", "FutureRisk"),
        ("TrafficDelayProb", "FutureRisk"),
    ])

    # --------------------------
    # 2. CPDs (Conditional Probability Distributions)
    # --------------------------

    # 2.1 ImpactSpeed prior
    # Skewed towards medium speeds, but with decent high-speed mass due to Indian traffic.
    cpd_impactspeed = TabularCPD(
        variable="ImpactSpeed",
        variable_card=3,
        values=[[0.20],  # low
                [0.55],  # medium
                [0.25]], # high
        state_names={"ImpactSpeed": ["low", "medium", "high"]}
    )

    # 2.2 VehicleType prior (aggregated categories)
    cpd_vehicletype = TabularCPD(
        variable="VehicleType",
        variable_card=4,
        values=[[0.30],  # bike / 2-wheeler
                [0.45],  # car / small private vehicle
                [0.15],  # suv / larger private vehicle
                [0.10]], # auto / small commercial
        state_names={"VehicleType": ["bike", "car", "suv", "auto"]}
    )

    # 2.3 Weather prior
    cpd_weather = TabularCPD(
        variable="Weather",
        variable_card=2,
        values=[[0.7],  # clear
                [0.3]], # rain
        state_names={"Weather": ["clear", "rain"]}
    )

    # 2.4 CallerUrgency prior
    cpd_caller = TabularCPD(
        variable="CallerUrgency",
        variable_card=3,
        values=[[0.3],  # calm
                [0.4],  # tense
                [0.3]], # panicked
        state_names={"CallerUrgency": ["calm", "tense", "panicked"]}
    )

    # 2.5 TrafficDelayProb | Weather, ImpactSpeed
    # Parents: [Weather, ImpactSpeed]
    cpd_traffic_delay = TabularCPD(
        variable="TrafficDelayProb",
        variable_card=3,
        values=[
            # low delay
            [0.7, 0.5, 0.4,  0.4, 0.3, 0.2],
            # medium delay
            [0.2, 0.3, 0.3,  0.3, 0.3, 0.3],
            # high delay
            [0.1, 0.2, 0.3,  0.3, 0.4, 0.5],
        ],
        evidence=["Weather", "ImpactSpeed"],
        evidence_card=[2, 3],
        state_names={
            "TrafficDelayProb": ["low", "medium", "high"],
            "Weather": ["clear", "rain"],
            "ImpactSpeed": ["low", "medium", "high"],
        },
    )

    # 2.6 CurrentSeverity | ImpactSpeed, VehicleType, CallerUrgency
    # Parents: [ImpactSpeed, VehicleType, CallerUrgency]
    # We construct 36 columns programmatically based on domain heuristics,
    # calibrated so that "minor" is most likely overall (similar to the CSV),
    # but high-speed / heavier-vehicle / panicked caller pushes towards critical.

    severity_rows = []
    speeds = ["low", "medium", "high"]
    vehicles = ["bike", "car", "suv", "auto"]
    urgencies = ["calm", "tense", "panicked"]

    columns = []
    for sp in speeds:
        for vt in vehicles:
            for ur in urgencies:
                # Base pattern by speed
                if sp == "low":
                    base = np.array([0.80, 0.17, 0.03])  # minor, moderate, critical
                elif sp == "medium":
                    base = np.array([0.65, 0.27, 0.08])
                else:  # high speed
                    base = np.array([0.45, 0.35, 0.20])

                # Adjust for vehicle
                if vt in ["car", "suv"]:
                    # slightly more mass in critical
                    base = base + np.array([-0.03, 0.0, 0.03])
                else:  # bike/auto slightly lighter, but still risky
                    base = base + np.array([0.01, 0.0, -0.01])

                # Adjust for caller urgency
                if ur == "tense":
                    base = base + np.array([-0.02, 0.01, 0.01])
                elif ur == "panicked":
                    base = base + np.array([-0.05, 0.0, 0.05])

                # Clamp and normalize
                base = np.clip(base, 0.01, 0.95)
                base = base / base.sum()

                columns.append(base)

    columns = np.array(columns).T  # shape (3, 36)

    cpd_current_severity = TabularCPD(
        variable="CurrentSeverity",
        variable_card=3,
        values=columns,
        evidence=["ImpactSpeed", "VehicleType", "CallerUrgency"],
        evidence_card=[3, 4, 3],
        state_names={
            "CurrentSeverity": ["minor", "moderate", "critical"],
            "ImpactSpeed": ["low", "medium", "high"],
            "VehicleType": ["bike", "car", "suv", "auto"],
            "CallerUrgency": ["calm", "tense", "panicked"],
        },
    )

    # 2.7 FutureRisk | CurrentSeverity, TrafficDelayProb
    cpd_future_risk = TabularCPD(
        variable="FutureRisk",
        variable_card=3,
        values=[
            # low risk
            [0.8, 0.6, 0.4,  0.6, 0.4, 0.2,  0.4, 0.2, 0.1],
            # medium risk
            [0.15, 0.3, 0.3,  0.3, 0.4, 0.3,  0.4, 0.4, 0.3],
            # high risk
            [0.05, 0.1, 0.3,  0.1, 0.2, 0.5,  0.2, 0.4, 0.6],
        ],
        evidence=["CurrentSeverity", "TrafficDelayProb"],
        evidence_card=[3, 3],
        state_names={
            "FutureRisk": ["low", "medium", "high"],
            "CurrentSeverity": ["minor", "moderate", "critical"],
            "TrafficDelayProb": ["low", "medium", "high"],
        },
    )

    # 3. Add CPDs to model and check
    model.add_cpds(
        cpd_impactspeed,
        cpd_vehicletype,
        cpd_weather,
        cpd_caller,
        cpd_traffic_delay,
        cpd_current_severity,
        cpd_future_risk,
    )

    model.check_model()
    return model


if __name__ == "__main__":
    m = build_emergency_bayes_model()
    print("Bayesian model nodes:", m.nodes())
    print("Bayesian model edges:", m.edges())
