"""
Configuration management for ML backend
"""
import os
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        # Fallback for older pydantic versions
        from pydantic import BaseSettings
    except ImportError:
        # If both fail, create a simple BaseSettings
        from pydantic import BaseModel as BaseSettings

from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = DATA_DIR / "models"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    # Database Settings
    DATABASE_URL: str = Field(
        default="postgresql://user:password@localhost:5432/safety_db",
        env="DATABASE_URL"
    )
    MONGODB_URL: str = Field(
        default="mongodb://localhost:27017/safety_db",
        env="MONGODB_URL"
    )
    
    # Redis Settings
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    
    # Storage Settings
    S3_BUCKET_NAME: Optional[str] = Field(default=None, env="S3_BUCKET_NAME")
    S3_REGION: str = Field(default="us-east-1", env="S3_REGION")
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    
    # MLflow Settings
    MLFLOW_TRACKING_URI: str = Field(
        default="sqlite:///mlflow.db",
        env="MLFLOW_TRACKING_URI"
    )
    
    # Model Settings
    MODEL_CACHE_SIZE: int = 5  # Number of models to keep in memory
    INFERENCE_BATCH_SIZE: int = 32
    INFERENCE_DEVICE: str = "cpu"  # "cpu" or "cuda"
    
    # Performance Settings
    MAX_WORKERS: int = 4
    TIMEOUT_SECONDS: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Global settings instance
settings = Settings()

# Create necessary directories
settings.DATA_DIR.mkdir(exist_ok=True)
settings.MODELS_DIR.mkdir(exist_ok=True)
settings.RAW_DATA_DIR.mkdir(exist_ok=True)
settings.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
