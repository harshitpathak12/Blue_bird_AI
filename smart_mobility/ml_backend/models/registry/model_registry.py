"""
Model Registry - Version and manage ML models
"""
import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from ml_backend.config import settings


class ModelRegistry:
    """Manage model versions and metadata"""
    
    def __init__(self):
        self.registry_path = settings.MODELS_DIR / "registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load model registry"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save model registry"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_name: str, model_path: str, 
                      metadata: Dict) -> str:
        """Register a new model version"""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if model_name not in self.registry:
            self.registry[model_name] = {}
        
        self.registry[model_name][version] = {
            "model_path": str(model_path),
            "metadata": metadata,
            "registered_at": datetime.now().isoformat()
        }
        
        self._save_registry()
        return version
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get latest version of a model"""
        if model_name not in self.registry:
            return None
        versions = sorted(self.registry[model_name].keys(), reverse=True)
        return versions[0] if versions else None
