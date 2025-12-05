# common/constants.py
from enum import Enum


class Weather(str, Enum):
    CLEAR = "clear"
    RAIN = "rain"


class AccidentSeverity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    CRITICAL = "critical"


class FutureRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VehicleType(str, Enum):
    BIKE = "bike"
    CAR = "car"
    SUV = "suv"
    AUTO = "auto"
    TRUCK = "truck"
    OTHER = "other"


class HospitalCTStatus(str, Enum):
    AVAILABLE = "available"
    OFFLINE = "offline"


class RoadCondition(str, Enum):
    NORMAL = "normal"
    CONGESTED = "congested"
    BLOCKED = "blocked"


# Global settings from the problem statement
RAIN_SPEED_REDUCTION = 0.30  # 30% decrease in average speed when raining
