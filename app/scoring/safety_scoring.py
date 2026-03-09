"""
Daily safety scoring 
score = 100 - (fatigue_count × 5) - (distraction_count × 3) - (sleep_events × 10)
90–100 → Safe, 70–89 → Moderate Risk, <70 → High Risk
"""

from datetime import date


def risk_level_from_score(score: float) -> str:
    if score >= 90:
        return "Safe"
    if score >= 70:
        return "Moderate Risk"
    return "High Risk"


class SafetyScoring:
    """Compute daily safety score from alert counts per PDF formula."""

    FATIGUE_PENALTY = 5
    DISTRACTION_PENALTY = 3
    SLEEP_PENALTY = 10

    @classmethod
    def compute(
        cls,
        fatigue_count: int,
        distraction_count: int,
        sleep_count: int,
    ) -> float:
        score = 100.0
        score -= fatigue_count * cls.FATIGUE_PENALTY
        score -= distraction_count * cls.DISTRACTION_PENALTY
        score -= sleep_count * cls.SLEEP_PENALTY
        return max(0.0, min(100.0, score))
