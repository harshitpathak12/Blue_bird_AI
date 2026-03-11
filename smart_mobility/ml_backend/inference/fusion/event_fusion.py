"""
Event Fusion & Scoring Engine
Combines outputs from multiple AI models and calculates safety scores
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import h3
from ml_backend.config.model_configs import FUSION_CONFIG


class EventFusionEngine:
    """
    Fuses events from multiple AI models:
    - Driver Safety models (fatigue, distraction, etc.)
    - Road Safety models (potholes, cracks, etc.)
    
    Performs:
    - Temporal alignment
    - Spatial clustering (H3)
    - Confidence aggregation
    - Severity classification
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize event fusion engine
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = FUSION_CONFIG
        
        self.temporal_window = config.get("temporal_window", 5.0)  # seconds
        self.spatial_threshold = config.get("spatial_threshold", 50.0)  # meters
        self.confidence_weights = config.get("confidence_weights", {
            "driver_safety": 0.6,
            "road_safety": 0.4
        })
        self.severity_levels = config.get("severity_levels", {
            "low": 0.0,
            "medium": 0.3,
            "high": 0.6,
            "critical": 0.8
        })
        self.h3_resolution = config.get("h3_resolution", 9)
    
    def fuse_events(self, events: List[Dict]) -> List[Dict]:
        """
        Fuse multiple events into consolidated events
        
        Args:
            events: List of event dictionaries with:
                - timestamp
                - latitude, longitude (GPS)
                - event_type
                - confidence
                - model_type (driver_safety/road_safety)
                - severity (optional)
                
        Returns:
            List of fused events
        """
        if not events:
            return []
        
        # Group events by temporal and spatial proximity
        fused_events = []
        processed = set()
        
        for i, event in enumerate(events):
            if i in processed:
                continue
            
            # Find related events
            related_events = [event]
            related_indices = [i]
            
            for j, other_event in enumerate(events[i+1:], start=i+1):
                if j in processed:
                    continue
                
                if self._are_related(event, other_event):
                    related_events.append(other_event)
                    related_indices.append(j)
            
            # Mark as processed
            processed.update(related_indices)
            
            # Fuse related events
            fused_event = self._fuse_event_group(related_events)
            fused_events.append(fused_event)
        
        return fused_events
    
    def _are_related(self, event1: Dict, event2: Dict) -> bool:
        """
        Check if two events are related (temporally and spatially)
        
        Args:
            event1: First event
            event2: Second event
            
        Returns:
            True if events are related
        """
        # Temporal check
        time_diff = abs(event1.get("timestamp", 0) - event2.get("timestamp", 0))
        if time_diff > self.temporal_window:
            return False
        
        # Spatial check
        if "latitude" in event1 and "latitude" in event2:
            distance = self._calculate_distance(
                event1["latitude"], event1["longitude"],
                event2["latitude"], event2["longitude"]
            )
            if distance > self.spatial_threshold:
                return False
        
        return True
    
    def _fuse_event_group(self, events: List[Dict]) -> Dict:
        """
        Fuse a group of related events into a single event
        
        Args:
            events: List of related events
            
        Returns:
            Fused event dictionary
        """
        if len(events) == 1:
            return events[0]
        
        # Aggregate confidences
        confidences = [e.get("confidence", 0.0) for e in events]
        avg_confidence = np.mean(confidences)
        max_confidence = np.max(confidences)
        
        # Weighted confidence (higher weight for higher confidence)
        weighted_confidence = np.average(confidences, weights=confidences)
        
        # Aggregate GPS (average)
        latitudes = [e.get("latitude") for e in events if "latitude" in e]
        longitudes = [e.get("longitude") for e in events if "longitude" in e]
        
        avg_lat = np.mean(latitudes) if latitudes else None
        avg_lon = np.mean(longitudes) if longitudes else None
        
        # Get H3 cell
        h3_cell = None
        if avg_lat and avg_lon:
            h3_cell = h3.geo_to_h3(avg_lat, avg_lon, self.h3_resolution)
        
        # Aggregate event types
        event_types = [e.get("event_type", "unknown") for e in events]
        primary_event_type = max(set(event_types), key=event_types.count)
        
        # Determine severity
        severities = [e.get("severity", "low") for e in events]
        max_severity = self._get_max_severity(severities)
        
        # Aggregate model types
        model_types = [e.get("model_type", "unknown") for e in events]
        
        return {
            "event_id": self._generate_event_id(events),
            "event_type": primary_event_type,
            "event_types": list(set(event_types)),
            "confidence": float(weighted_confidence),
            "avg_confidence": float(avg_confidence),
            "max_confidence": float(max_confidence),
            "severity": max_severity,
            "latitude": float(avg_lat) if avg_lat else None,
            "longitude": float(avg_lon) if avg_lon else None,
            "h3_cell": h3_cell,
            "timestamp": events[0].get("timestamp"),
            "model_types": list(set(model_types)),
            "event_count": len(events),
            "fused": True
        }
    
    def _get_max_severity(self, severities: List[str]) -> str:
        """Get maximum severity from list"""
        severity_order = ["low", "medium", "high", "critical"]
        severity_values = {s: i for i, s in enumerate(severity_order)}
        
        max_sev = "low"
        max_val = -1
        
        for severity in severities:
            val = severity_values.get(severity.lower(), -1)
            if val > max_val:
                max_val = val
                max_sev = severity
        
        return max_sev
    
    def _calculate_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS points (Haversine)"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371000  # Earth radius in meters
        
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)
        
        a = (sin(delta_lat / 2) ** 2 +
             cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2)
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        return R * c
    
    def _generate_event_id(self, events: List[Dict]) -> str:
        """Generate unique event ID"""
        import hashlib
        event_str = str(sorted([e.get("event_type", "") for e in events]))
        return hashlib.md5(event_str.encode()).hexdigest()[:12]
    
    def classify_severity(self, event: Dict) -> str:
        """
        Classify event severity based on confidence and event type
        
        Args:
            event: Event dictionary
            
        Returns:
            Severity level (low, medium, high, critical)
        """
        confidence = event.get("confidence", 0.0)
        event_type = event.get("event_type", "")
        
        # Critical events
        critical_types = ["drunk_driving", "crash", "critical_fatigue"]
        if event_type in critical_types and confidence > 0.7:
            return "critical"
        
        # High severity
        if confidence >= self.severity_levels["high"]:
            return "high"
        elif confidence >= self.severity_levels["medium"]:
            return "medium"
        else:
            return "low"
    
    def cluster_by_location(self, events: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Cluster events by H3 cell location
        
        Args:
            events: List of events
            
        Returns:
            Dictionary mapping H3 cells to events
        """
        clusters = defaultdict(list)
        
        for event in events:
            if "h3_cell" in event and event["h3_cell"]:
                clusters[event["h3_cell"]].append(event)
            elif "latitude" in event and "longitude" in event:
                h3_cell = h3.geo_to_h3(
                    event["latitude"], 
                    event["longitude"], 
                    self.h3_resolution
                )
                event["h3_cell"] = h3_cell
                clusters[h3_cell].append(event)
        
        return dict(clusters)
