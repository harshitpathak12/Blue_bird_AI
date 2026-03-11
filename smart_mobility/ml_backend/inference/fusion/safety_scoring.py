"""
Safety Score Engine - Calculates driver and road safety scores
"""
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque


class SafetyScoreEngine:
    """
    Calculates safety scores for:
    - Driver safety (based on behavior)
    - Road safety (based on road conditions)
    - Overall safety score
    """
    
    def __init__(self, time_window_seconds: int = 300):
        """
        Initialize safety score engine
        
        Args:
            time_window_seconds: Time window for score calculation (default 5 minutes)
        """
        self.time_window = timedelta(seconds=time_window_seconds)
        self.driver_weights = {
            "fatigue": 0.3,
            "distraction": 0.25,
            "harsh_braking": 0.15,
            "overspeeding": 0.15,
            "swerving": 0.15
        }
        self.road_weights = {
            "pothole": 0.3,
            "crack": 0.2,
            "roughness": 0.25,
            "water_logging": 0.25
        }
    
    def calculate_driver_safety_score(self, events: List[Dict]) -> Dict:
        """
        Calculate driver safety score from events
        
        Args:
            events: List of driver safety events
            
        Returns:
            Dictionary with safety score and breakdown
        """
        if not events:
            return {
                "score": 100.0,
                "category": "excellent",
                "breakdown": {}
            }
        
        # Filter events by time window
        now = datetime.now().timestamp()
        recent_events = [
            e for e in events 
            if now - e.get("timestamp", 0) <= self.time_window.total_seconds()
        ]
        
        if not recent_events:
            return {
                "score": 100.0,
                "category": "excellent",
                "breakdown": {}
            }
        
        # Calculate penalty for each event type
        penalties = {}
        total_penalty = 0.0
        
        for event_type, weight in self.driver_weights.items():
            type_events = [e for e in recent_events if e.get("event_type") == event_type]
            
            if type_events:
                # Calculate penalty based on frequency and severity
                count = len(type_events)
                avg_confidence = np.mean([e.get("confidence", 0.5) for e in type_events])
                
                # Severity multiplier
                severities = [e.get("severity", "low") for e in type_events]
                severity_mult = {
                    "low": 1.0,
                    "medium": 1.5,
                    "high": 2.0,
                    "critical": 3.0
                }
                max_severity_mult = max([severity_mult.get(s, 1.0) for s in severities])
                
                penalty = count * avg_confidence * weight * max_severity_mult * 10
                penalties[event_type] = penalty
                total_penalty += penalty
        
        # Calculate score (100 - penalty, minimum 0)
        score = max(0.0, 100.0 - total_penalty)
        
        # Categorize score
        if score >= 90:
            category = "excellent"
        elif score >= 75:
            category = "good"
        elif score >= 60:
            category = "fair"
        elif score >= 40:
            category = "poor"
        else:
            category = "critical"
        
        return {
            "score": float(score),
            "category": category,
            "breakdown": penalties,
            "event_count": len(recent_events),
            "time_window_seconds": self.time_window.total_seconds()
        }
    
    def calculate_road_safety_score(self, events: List[Dict]) -> Dict:
        """
        Calculate road safety score from road condition events
        
        Args:
            events: List of road safety events
            
        Returns:
            Dictionary with road safety score
        """
        if not events:
            return {
                "score": 100.0,
                "category": "excellent",
                "breakdown": {}
            }
        
        # Filter by time window
        now = datetime.now().timestamp()
        recent_events = [
            e for e in events 
            if now - e.get("timestamp", 0) <= self.time_window.total_seconds()
        ]
        
        if not recent_events:
            return {
                "score": 100.0,
                "category": "excellent",
                "breakdown": {}
            }
        
        # Calculate penalty
        penalties = {}
        total_penalty = 0.0
        
        for event_type, weight in self.road_weights.items():
            type_events = [e for e in recent_events if e.get("event_type") == event_type]
            
            if type_events:
                count = len(type_events)
                avg_confidence = np.mean([e.get("confidence", 0.5) for e in type_events])
                
                # Severity multiplier
                severities = [e.get("severity", "low") for e in type_events]
                severity_mult = {
                    "low": 1.0,
                    "medium": 1.5,
                    "high": 2.0,
                    "critical": 3.0
                }
                max_severity_mult = max([severity_mult.get(s, 1.0) for s in severities])
                
                penalty = count * avg_confidence * weight * max_severity_mult * 15
                penalties[event_type] = penalty
                total_penalty += penalty
        
        score = max(0.0, 100.0 - total_penalty)
        
        # Categorize
        if score >= 85:
            category = "excellent"
        elif score >= 70:
            category = "good"
        elif score >= 55:
            category = "fair"
        elif score >= 40:
            category = "poor"
        else:
            category = "critical"
        
        return {
            "score": float(score),
            "category": category,
            "breakdown": penalties,
            "event_count": len(recent_events)
        }
    
    def calculate_overall_safety_score(self, driver_events: List[Dict], 
                                      road_events: List[Dict],
                                      driver_weight: float = 0.6,
                                      road_weight: float = 0.4) -> Dict:
        """
        Calculate overall safety score combining driver and road safety
        
        Args:
            driver_events: Driver safety events
            road_events: Road safety events
            driver_weight: Weight for driver safety (default 0.6)
            road_weight: Weight for road safety (default 0.4)
            
        Returns:
            Dictionary with overall safety score
        """
        driver_score = self.calculate_driver_safety_score(driver_events)
        road_score = self.calculate_road_safety_score(road_events)
        
        # Weighted average
        overall_score = (
            driver_score["score"] * driver_weight +
            road_score["score"] * road_weight
        )
        
        # Categorize
        if overall_score >= 90:
            category = "excellent"
        elif overall_score >= 75:
            category = "good"
        elif overall_score >= 60:
            category = "fair"
        elif overall_score >= 40:
            category = "poor"
        else:
            category = "critical"
        
        return {
            "overall_score": float(overall_score),
            "category": category,
            "driver_score": driver_score["score"],
            "road_score": road_score["score"],
            "driver_category": driver_score["category"],
            "road_category": road_score["category"],
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_trend(self, scores: List[float], window_size: int = 10) -> Dict:
        """
        Calculate safety score trend
        
        Args:
            scores: List of historical scores
            window_size: Window size for trend calculation
            
        Returns:
            Dictionary with trend information
        """
        if len(scores) < 2:
            return {
                "trend": "stable",
                "change": 0.0,
                "slope": 0.0
            }
        
        # Use recent scores
        recent_scores = scores[-window_size:] if len(scores) > window_size else scores
        
        # Calculate linear regression slope
        x = np.arange(len(recent_scores))
        slope = np.polyfit(x, recent_scores, 1)[0]
        
        # Determine trend
        if slope > 1.0:
            trend = "improving"
        elif slope < -1.0:
            trend = "declining"
        else:
            trend = "stable"
        
        # Calculate change
        change = recent_scores[-1] - recent_scores[0]
        
        return {
            "trend": trend,
            "change": float(change),
            "slope": float(slope),
            "current_score": float(recent_scores[-1]),
            "previous_score": float(recent_scores[0]) if len(recent_scores) > 1 else None
        }
