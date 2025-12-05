# common/entities.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

from .constants import (
    Weather,
    AccidentSeverity,
    FutureRisk,
    VehicleType,
    HospitalCTStatus,
)


@dataclass
class Accident:
    id: str
    location_node: str            # node id in the road network graph
    description: str
    time_reported: datetime

    vehicle_types: List[VehicleType]
    estimated_speed_kmph: Optional[float] = None
    caller_urgency: Optional[str] = None     # e.g. "calm", "tense", "panicked"
    weather: Weather = Weather.CLEAR

    # Outputs from Module 1
    predicted_severity: Optional[AccidentSeverity] = None
    severity_distribution: Optional[Dict[str, float]] = None
    future_risk_distribution: Optional[Dict[str, float]] = None


@dataclass
class Hospital:
    id: str
    name: str
    node: str                     # node id in the road network graph
    is_trauma_center: bool
    base_distance_km_from_stadium: float     # just for documentation/report
    ct_status: HospitalCTStatus = HospitalCTStatus.AVAILABLE
    ct_offline_until: Optional[datetime] = None
    capacity_beds: int = 20
    current_load: int = 0         # patients currently admitted


@dataclass
class Ambulance:
    id: str
    current_node: str             # node id in the road network
    is_available: bool = True
    assigned_accident_id: Optional[str] = None

    # For Module 4 RL & Module 2 route tracking
    en_route_to_node: Optional[str] = None
    eta_minutes: Optional[float] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
