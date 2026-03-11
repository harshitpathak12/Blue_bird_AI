"""
Metrics Tracker - Track all success metrics
"""
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
import json
from pathlib import Path
from ml_backend.config import settings


class MetricsTracker:
    """Track metrics for monitoring"""
    
    def __init__(self):
        self.metrics_dir = settings.DATA_DIR / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = defaultdict(list)
    
    def track_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Track a metric value"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.metrics[metric_name].append({
            "value": value,
            "timestamp": timestamp.isoformat()
        })
    
    def get_metrics(self, metric_name: str) -> List[Dict]:
        """Get historical metrics"""
        return self.metrics.get(metric_name, [])
    
    def save_metrics(self):
        """Save metrics to file"""
        metrics_file = self.metrics_dir / f"metrics_{datetime.now().strftime('%Y%m%d')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
