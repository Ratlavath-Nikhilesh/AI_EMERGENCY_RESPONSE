# module4_rl/env.py

import random
from typing import Tuple, Dict, Any

import numpy as np


# ----------------- Discrete codes for state -----------------

# Ambulance status / region indices for A1
A1_IDLE_H1 = 0
A1_IDLE_STADIUM = 1
A1_IDLE_AIRPORT = 2
A1_BUSY = 3

# Ambulance status / region indices for A2
A2_IDLE_H2 = 0
A2_IDLE_RUPALI = 1
A2_IDLE_AIRPORT = 2
A2_BUSY = 3

# Accident locations (coarse regions)
ACC_NONE = 0
ACC_JAYDEV = 1   # near stadium
ACC_RUPALI = 2
ACC_AIRPORT = 3

# Accident severities
SEV_NONE = 0
SEV_MINOR = 1
SEV_MODERATE = 2
SEV_CRITICAL = 3

# Predicted hotspot
HOTSPOT_NONE = 0
HOTSPOT_STADIUM = 1
HOTSPOT_RUPALI = 2
HOTSPOT_AIRPORT = 3

# Road conditions
ROAD_NORMAL = 0
ROAD_CONGESTED = 1

# Hospital load / capacity forecasts
HLOAD_BOTH_OK = 0
HLOAD_H1_BUSY = 1
HLOAD_H2_BUSY = 2

# Actions
ACT_PREPOSITION_STADIUM = 0
ACT_PREPOSITION_AIRPORT = 1
ACT_IMMEDIATE_DISPATCH = 2
ACT_DELAY_DISPATCH = 3
ACT_REROUTE_MID = 4

N_ACTIONS = 5


# Regions for travel time approximation
REGION_H1 = 0
REGION_H2 = 1
REGION_STADIUM = 2
REGION_RUPALI = 3
REGION_AIRPORT = 4

# Base travel times (minutes) between regions under normal conditions
# Indices: [from_region][to_region]
BASE_TRAVEL_TIME = np.array([
    #   H1  H2  STAD  RUPALI  AIRPORT
    [ 0, 20, 10, 18, 25],   # from H1
    [20,  0, 22, 10, 18],   # from H2
    [10, 22,  0,  8, 20],   # from STADIUM
    [18, 10,  8,  0, 12],   # from RUPALI
    [25, 18, 20, 12,  0],   # from AIRPORT
])


def accident_location_to_region(acc_loc: int) -> int:
    if acc_loc == ACC_JAYDEV:
        return REGION_STADIUM
    elif acc_loc == ACC_RUPALI:
        return REGION_RUPALI
    elif acc_loc == ACC_AIRPORT:
        return REGION_AIRPORT
    else:
        raise ValueError("Cannot map ACC_NONE to region")


def a1_status_to_region(status: int) -> int:
    if status == A1_IDLE_H1:
        return REGION_H1
    elif status == A1_IDLE_STADIUM:
        return REGION_STADIUM
    elif status == A1_IDLE_AIRPORT:
        return REGION_AIRPORT
    elif status == A1_BUSY:
        # when busy, approximate region as STADIUM/RUPALI/AIRPORT; we will use current accident region instead
        return None
    else:
        raise ValueError(f"Unknown A1 status: {status}")


def a2_status_to_region(status: int) -> int:
    if status == A2_IDLE_H2:
        return REGION_H2
    elif status == A2_IDLE_RUPALI:
        return REGION_RUPALI
    elif status == A2_IDLE_AIRPORT:
        return REGION_AIRPORT
    elif status == A2_BUSY:
        return None
    else:
        raise ValueError(f"Unknown A2 status: {status}")


class EmergencyResponseEnv:
    """
    Simplified emergency response MDP environment for Q-learning.

    - At most one active accident at a time.
    - Two ambulances (A1 at H1, A2 at H2 initially).
    - Time is discretised into steps (e.g., 5 minutes each).
    """

    def __init__(self, max_steps: int = 10, seed: int = 0):
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        # State variables
        self.time_step = 0
        self.a1_status = A1_IDLE_H1
        self.a2_status = A2_IDLE_H2
        self.active_acc_loc = ACC_NONE
        self.active_severity = SEV_NONE
        self.predicted_hotspot = HOTSPOT_STADIUM  # after stadium match, expect near stadium
        self.road_condition = ROAD_CONGESTED     # rainy + stadium crowd
        self.hospital_load = HLOAD_H2_BUSY       # H2 CT offline; H1 OK

        # For stats
        self.last_response_time = 0.0
        self.total_response_time = 0.0
        self.served_accidents = 0
        self.critical_within_threshold = 0
        self.critical_total = 0

    # ------------- Helper functions -------------

    def _encode_state(self) -> Tuple[int, ...]:
        return (
            self.time_step,
            self.a1_status,
            self.a2_status,
            self.active_acc_loc,
            self.active_severity,
            self.predicted_hotspot,
            self.road_condition,
            self.hospital_load,
        )

    def _sample_new_accident_if_none(self):
        """Generate a new accident with probability depending on predicted hotspot."""
        if self.active_acc_loc != ACC_NONE:
            return  # already an active accident

        # Base arrival probability
        if self.predicted_hotspot == HOTSPOT_NONE:
            p_new = 0.1
        else:
            p_new = 0.4

        if self.rng.random() > p_new:
            return  # no new accident this step

        # Sample location biased by hotspot
        r = self.rng.random()
        if self.predicted_hotspot == HOTSPOT_STADIUM:
            # more likely near Jaydev or Rupali
            if r < 0.6:
                self.active_acc_loc = ACC_JAYDEV
            elif r < 0.9:
                self.active_acc_loc = ACC_RUPALI
            else:
                self.active_acc_loc = ACC_AIRPORT
        elif self.predicted_hotspot == HOTSPOT_RUPALI:
            if r < 0.2:
                self.active_acc_loc = ACC_JAYDEV
            elif r < 0.7:
                self.active_acc_loc = ACC_RUPALI
            else:
                self.active_acc_loc = ACC_AIRPORT
        elif self.predicted_hotspot == HOTSPOT_AIRPORT:
            if r < 0.1:
                self.active_acc_loc = ACC_JAYDEV
            elif r < 0.3:
                self.active_acc_loc = ACC_RUPALI
            else:
                self.active_acc_loc = ACC_AIRPORT
        else:
            # no strong hotspot
            self.active_acc_loc = self.rng.choice([ACC_JAYDEV, ACC_RUPALI, ACC_AIRPORT])

        # Sample severity roughly consistent with India dataset
        r_sev = self.rng.random()
        if r_sev < 0.8:
            self.active_severity = SEV_MINOR
        elif r_sev < 0.95:
            self.active_severity = SEV_MODERATE
        else:
            self.active_severity = SEV_CRITICAL

        if self.active_severity == SEV_CRITICAL:
            self.critical_total += 1

    def _severity_weight(self, sev: int) -> int:
        if sev == SEV_MINOR:
            return 1
        elif sev == SEV_MODERATE:
            return 2
        elif sev == SEV_CRITICAL:
            return 3
        else:
            return 0

    def _compute_travel_time(self, amb_region: int, acc_loc: int) -> float:
        acc_region = accident_location_to_region(acc_loc)
        base = BASE_TRAVEL_TIME[amb_region, acc_region]
        if self.road_condition == ROAD_CONGESTED:
            return base * 1.3  # 30% slower due to rain + match traffic
        else:
            return base

    def _dispatch_nearest(self) -> float:
        """
        Dispatch the nearest available ambulance to current active accident.
        Returns the estimated response time (minutes).
        """
        if self.active_acc_loc == ACC_NONE:
            return 0.0

        candidates = []

        if self.a1_status in (A1_IDLE_H1, A1_IDLE_STADIUM, A1_IDLE_AIRPORT):
            r1 = a1_status_to_region(self.a1_status)
            t1 = self._compute_travel_time(r1, self.active_acc_loc)
            candidates.append(("A1", t1))

        if self.a2_status in (A2_IDLE_H2, A2_IDLE_RUPALI, A2_IDLE_AIRPORT):
            r2 = a2_status_to_region(self.a2_status)
            t2 = self._compute_travel_time(r2, self.active_acc_loc)
            candidates.append(("A2", t2))

        if not candidates:
            # No free ambulance: huge penalty
            return 60.0

        amb_id, travel_time = min(candidates, key=lambda x: x[1])

        # After service, assume ambulance becomes idle at the nearest hospital
        # If accident near airport/rupali -> nearest is H2; if near Jaydev -> nearest is H1
        acc_region = accident_location_to_region(self.active_acc_loc)
        if acc_region in (REGION_AIRPORT, REGION_RUPALI):
            end_region = REGION_H2
        else:
            end_region = REGION_H1

        if amb_id == "A1":
            # mark busy only for this step, then idle at end_region
            self.a1_status = A1_BUSY
        else:
            self.a2_status = A2_BUSY

        # We won't simulate multiple steps of travel; we treat it as completed this step.
        # Update final positions:
        if amb_id == "A1":
            self.a1_status = (
                A1_IDLE_H1 if end_region == REGION_H1 else A1_IDLE_AIRPORT
            )
        else:
            self.a2_status = (
                A2_IDLE_H2 if end_region == REGION_H2 else A2_IDLE_AIRPORT
            )

        # Clear accident
        sev = self.active_severity
        self.active_acc_loc = ACC_NONE
        self.active_severity = SEV_NONE

        self.last_response_time = travel_time
        self.total_response_time += travel_time
        self.served_accidents += 1

        # Check critical within threshold (e.g. 15 minutes)
        if sev == SEV_CRITICAL and travel_time <= 15.0:
            self.critical_within_threshold += 1

        return travel_time

    def _compute_service_reward(self, severity: int, response_time: float) -> float:
        """
        Reward for serving an accident, prioritising critical cases and low time.
        """
        w = self._severity_weight(severity)
        if w == 0:
            return 0.0

        # Base: +60 for critical, +40 for moderate, +20 for minor (w * 20)
        base = 20.0 * w

        # Penalty proportional to time
        reward = base - 1.5 * response_time

        # Extra penalty if response time exceeds severity-specific threshold
        if severity == SEV_CRITICAL:
            threshold = 15.0
        elif severity == SEV_MODERATE:
            threshold = 20.0
        else:
            threshold = 25.0

        if response_time > threshold:
            reward -= 10.0

        return reward

    # ------------- Public RL interface -------------

    def reset(self) -> Tuple[int, ...]:
        self.time_step = 0
        self.a1_status = A1_IDLE_H1
        self.a2_status = A2_IDLE_H2
        self.active_acc_loc = ACC_NONE
        self.active_severity = SEV_NONE
        self.predicted_hotspot = HOTSPOT_STADIUM
        self.road_condition = ROAD_CONGESTED
        self.hospital_load = HLOAD_H2_BUSY

        self.last_response_time = 0.0
        self.total_response_time = 0.0
        self.served_accidents = 0
        self.critical_within_threshold = 0
        self.critical_total = 0

        # Possibly start with an accident in first step
        self._sample_new_accident_if_none()
        return self._encode_state()

    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool, Dict[str, Any]]:
        """
        Apply action and advance one time step.

        Returns:
            next_state, reward, done, info
        """
        reward = 0.0

        # Time penalty (we want to finish episode quickly with good service)
        reward -= 1.0

        # 1. Handle action effects w.r.t. current accident & ambulances
        if self.active_acc_loc != ACC_NONE:
            # There is an active accident
            if action == ACT_IMMEDIATE_DISPATCH:
                sev = self.active_severity
                t = self._dispatch_nearest()
                reward += self._compute_service_reward(sev, t)
            elif action == ACT_DELAY_DISPATCH:
                # penalty for delaying when accident exists, especially if critical
                if self.active_severity == SEV_CRITICAL:
                    reward -= 8.0
                elif self.active_severity == SEV_MODERATE:
                    reward -= 4.0
                else:
                    reward -= 2.0
                # accident remains, might get worse indirectly (not modelled explicitly)
            elif action in (ACT_PREPOSITION_STADIUM, ACT_PREPOSITION_AIRPORT):
                # We are wasting time moving ambulances instead of responding
                reward -= 5.0
                self._apply_preposition(action)
            elif action == ACT_REROUTE_MID:
                # Simple approximation: if critical + H2 not busy, give a small bonus;
                # else penalise for unnecessary complexity.
                if self.active_severity == SEV_CRITICAL and self.hospital_load != HLOAD_H2_BUSY:
                    reward += 3.0
                else:
                    reward -= 1.0
        else:
            # No active accident; prepositioning can be useful
            if action in (ACT_PREPOSITION_STADIUM, ACT_PREPOSITION_AIRPORT):
                self._apply_preposition(action)
                # small penalty for movement, but might pay off later
                reward -= 0.5
            elif action == ACT_DELAY_DISPATCH:
                # basically a no-op when no accident
                reward -= 0.5
            elif action == ACT_REROUTE_MID:
                # meaningless when no accident
                reward -= 0.5
            elif action == ACT_IMMEDIATE_DISPATCH:
                # also meaningless when no accident
                reward -= 0.5

        # 2. Update hotspot prediction & hospital load in a simple way
        self._update_predictions()

        # 3. Advance time and possibly spawn a new accident
        self.time_step += 1
        self._sample_new_accident_if_none()

        done = self.time_step >= self.max_steps

        next_state = self._encode_state()
        info = {
            "served_accidents": self.served_accidents,
            "total_response_time": self.total_response_time,
            "critical_total": self.critical_total,
            "critical_within_threshold": self.critical_within_threshold,
        }
        return next_state, reward, done, info

    def _apply_preposition(self, action: int):
        # Move nearest idle ambulance towards stadium or airport cluster
        if action == ACT_PREPOSITION_STADIUM:
            # move whichever idle ambulance is nearer (roughly)
            if self.a1_status == A1_IDLE_H1:
                self.a1_status = A1_IDLE_STADIUM
            elif self.a2_status == A2_IDLE_H2:
                self.a2_status = A2_IDLE_RUPALI
        elif action == ACT_PREPOSITION_AIRPORT:
            if self.a1_status in (A1_IDLE_H1, A1_IDLE_STADIUM):
                self.a1_status = A1_IDLE_AIRPORT
            elif self.a2_status in (A2_IDLE_H2, A2_IDLE_RUPALI):
                self.a2_status = A2_IDLE_AIRPORT

    def _update_predictions(self):
        """
        Simple heuristic evolution of predicted hotspot and hospital load.
        """
        # Update hotspot based on last accident
        if self.active_acc_loc == ACC_JAYDEV:
            self.predicted_hotspot = HOTSPOT_STADIUM
        elif self.active_acc_loc == ACC_RUPALI:
            self.predicted_hotspot = HOTSPOT_RUPALI
        elif self.active_acc_loc == ACC_AIRPORT:
            self.predicted_hotspot = HOTSPOT_AIRPORT
        else:
            # if no accident, keep previous hotspot but with small chance of NONE
            if self.rng.random() < 0.1:
                self.predicted_hotspot = HOTSPOT_NONE

        # Hospital load: very simple dynamics
        # H2 starts busy (CT offline); small chance to become ok later in episode
        if self.hospital_load == HLOAD_H2_BUSY and self.rng.random() < 0.1:
            self.hospital_load = HLOAD_BOTH_OK
        elif self.hospital_load == HLOAD_BOTH_OK and self.rng.random() < 0.05:
            self.hospital_load = HLOAD_H1_BUSY
