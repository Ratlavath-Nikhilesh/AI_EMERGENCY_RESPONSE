# module1_bayes/inference_engine.py

from typing import Dict, Any

from pgmpy.inference import VariableElimination

from .bayes_model import build_emergency_bayes_model


class EmergencyBayesInference:
    """
    Wrapper around pgmpy VariableElimination for our emergency BN.
    """

    def __init__(self):
        self.model = build_emergency_bayes_model()
        self.infer = VariableElimination(self.model)

    def infer_severity_and_risk(self, evidence: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Compute posterior distributions over CurrentSeverity and FutureRisk.

        :param evidence: dict variable -> state string
        :return: {
            "CurrentSeverity": {state: prob, ...},
            "FutureRisk": {state: prob, ...},
        }
        """
        query_result = self.infer.query(
            variables=["CurrentSeverity", "FutureRisk"],
            evidence=evidence,
            show_progress=False
        )

        result = {}
        for var in ["CurrentSeverity", "FutureRisk"]:
            dist = query_result[var]
            states = self.model.get_cpds(var).state_names[var]
            probs = dist.values
            result[var] = {state: float(p) for state, p in zip(states, probs)}

        return result


if __name__ == "__main__":
    engine = EmergencyBayesInference()

    example_evidence = {
        "ImpactSpeed": "high",
        "VehicleType": "bike",
        "Weather": "rain",
        "CallerUrgency": "panicked",
        "TrafficDelayProb": "high",
    }

    out = engine.infer_severity_and_risk(example_evidence)
    print("Posterior distributions with example evidence:")
    for var, dist in out.items():
        print(var, ":", dist)
