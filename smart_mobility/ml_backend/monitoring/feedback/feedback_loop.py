"""
Feedback Loop - Collect and process user feedback
"""
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path
from ml_backend.config import settings


class FeedbackLoop:
    """Manage feedback for model improvement"""
    
    def __init__(self):
        self.feedback_dir = settings.DATA_DIR / "feedback"
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_feedback(self, event_id: str, feedback: Dict) -> str:
        """Collect user feedback on an event"""
        feedback_id = f"{event_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        feedback_data = {
            "feedback_id": feedback_id,
            "event_id": event_id,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        
        feedback_file = self.feedback_dir / f"{feedback_id}.json"
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        return feedback_id
    
    def get_feedback_for_event(self, event_id: str) -> List[Dict]:
        """Get all feedback for an event"""
        feedback_files = list(self.feedback_dir.glob(f"{event_id}_*.json"))
        feedbacks = []
        
        for file in feedback_files:
            with open(file, 'r') as f:
                feedbacks.append(json.load(f))
        
        return feedbacks
