"""
Multi-Model Fusion Engine per PDF §3.1.
Combines fatigue, distraction, blink/PERCLOS into unified driver state and alerts.
Example logic: PERCLOS > threshold & blink duration > threshold → Sleep; head turned > 3s → Distraction.
"""

from dataclasses import dataclass, field
from typing import Literal

# Alert types per PDF: fatigue, distraction, sleep
AlertType = Literal["fatigue", "distraction", "sleep"]


@dataclass
class ModelOutputs:
    """Raw outputs from monitoring models (fatigue, distraction, blink/PERCLOS)."""

    # Fatigue: low blink rate, high PERCLOS
    perclos: float = 0.0  # 0–1, % eye closure over time window
    blink_duration_sec: float = 0.0
    blink_rate_low: bool = False
    fatigue_score: float = 0.0  # 0–1

    # Distraction: head pose
    head_turned_away_sec: float = 0.0
    distraction_score: float = 0.0  # 0–1

    # Sleep: prolonged eyelid closure
    eye_closure_duration_sec: float = 0.0


@dataclass
class FusionResult:
    """Unified driver state and optional alert after fusion."""

    driver_state: str  # "normal" | "fatigue" | "distraction" | "sleep"
    alert_type: AlertType | None = None
    confidence_score: float = 0.0
    message: str = ""


class FusionEngine:
    """
    Fusion logic per PDF:
    - If PERCLOS > threshold and blink duration > threshold → Sleep alert
    - If head turned > threshold (e.g. 3s) → Distraction alert
    - If fatigue indicators → Fatigue alert
    """

    def __init__(
        self,
        perclos_sleep_threshold: float = 0.4,
        eye_closure_sleep_sec: float = 2.0,
        head_turned_distraction_sec: float = 3.0,
        fatigue_threshold: float = 0.6,
    ):
        self.perclos_sleep_threshold = perclos_sleep_threshold
        self.eye_closure_sleep_sec = eye_closure_sleep_sec
        self.head_turned_distraction_sec = head_turned_distraction_sec
        self.fatigue_threshold = fatigue_threshold

    def fuse(self, outputs: ModelOutputs) -> FusionResult:
        # Priority: sleep > distraction > fatigue > normal
        if (
            outputs.perclos >= self.perclos_sleep_threshold
            and outputs.eye_closure_duration_sec >= self.eye_closure_sleep_sec
        ):
            return FusionResult(
                driver_state="sleep",
                alert_type="sleep",
                confidence_score=min(1.0, outputs.perclos + 0.2),
                message="Prolonged eye closure detected",
            )
        if outputs.head_turned_away_sec >= self.head_turned_distraction_sec:
            return FusionResult(
                driver_state="distraction",
                alert_type="distraction",
                confidence_score=min(1.0, outputs.distraction_score or 0.8),
                message="Head turned away from road",
            )
        if outputs.fatigue_score >= self.fatigue_threshold or outputs.blink_rate_low:
            return FusionResult(
                driver_state="fatigue",
                alert_type="fatigue",
                confidence_score=outputs.fatigue_score or 0.7,
                message="Fatigue indicators detected",
            )
        return FusionResult(driver_state="normal", message="")
