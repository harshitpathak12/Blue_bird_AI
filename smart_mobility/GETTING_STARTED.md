# Getting Started Guide - ML/AI Backend

## For Beginners: Step-by-Step Implementation

This guide will help you build the ML/AI backend step by step, even if you're new to machine learning.

---

## Prerequisites

Before starting, make sure you have:
- Python 3.8 or higher installed
- Basic understanding of Python
- Git (optional, for version control)
- 8GB+ RAM recommended
- GPU (optional, but helpful for faster training)

---

## Step 1: Environment Setup (Day 1)

### 1.1 Create Virtual Environment

```bash
# Navigate to project directory
cd /home/ia-437/Downloads/transport_pjct

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

### 1.2 Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install requirements (this may take 10-15 minutes)
pip install -r requirements.txt
```

**Note**: If you encounter errors:
- For `dlib`: You may need to install system dependencies first
  ```bash
  # Ubuntu/Debian
  sudo apt-get install cmake libopenblas-dev liblapack-dev
  
  # Then install dlib
  pip install dlib
  ```

- For `insightface`: May require additional setup
  ```bash
  pip install insightface --no-deps
  pip install onnxruntime onnx
  ```

### 1.3 Verify Installation

```bash
python -c "import torch; import cv2; import numpy as np; print('All imports successful!')"
```

---

## Step 2: Understanding the Project Structure (Day 1-2)

### 2.1 Explore the Code

1. **Read the main plan**: `ML_AI_BACKEND_PLAN.md`
2. **Check configuration**: `ml_backend/config/settings.py`
3. **Review model configs**: `ml_backend/config/model_configs.py`

### 2.2 Key Concepts

- **Driver Safety Models**: Detect unsafe driver behavior
- **Road Safety Models**: Detect road defects and conditions
- **Event Fusion**: Combine multiple model outputs
- **Inference**: Running models on new data

---

## Step 3: Start with Simple Models (Week 1)

### 3.1 Test Face Detection

Create a test script: `test_face_detection.py`

```python
import cv2
import numpy as np
from ml_backend.models.driver_safety.face_detection import FaceDetector

# Initialize detector
detector = FaceDetector()

# Load test image (or use webcam)
# For webcam:
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces
    faces = detector.detect(frame)
    
    # Draw bounding boxes
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Run it:
```bash
python test_face_detection.py
```

### 3.2 Test Eye Landmark Detection

Create: `test_eye_landmark.py`

```python
import cv2
from ml_backend.models.driver_safety.eye_landmark import EyeLandmarkDetector

detector = EyeLandmarkDetector()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = detector.detect_landmarks(frame)
    if result:
        print(f"EAR: {result['avg_ear']:.3f}, Eye Open: {result['is_eye_open']}")
    
    cv2.imshow('Eye Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Step 4: Build Data Pipeline (Week 2)

### 4.1 Create Data Ingestion API

Create: `ml_backend/data/ingestion/api.py`

```python
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import Optional
import numpy as np
import cv2

app = FastAPI()

class IMUData(BaseModel):
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    timestamp: float

class GPSData(BaseModel):
    latitude: float
    longitude: float
    altitude: float
    timestamp: float

@app.post("/ingest/imu")
async def ingest_imu(data: IMUData):
    """Receive IMU data from mobile app"""
    # Store data (implement storage later)
    return {"status": "received", "data": data}

@app.post("/ingest/image")
async def ingest_image(file: UploadFile = File(...)):
    """Receive image from mobile app"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process image (implement later)
    return {"status": "received", "shape": image.shape}

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

Run the API:
```bash
uvicorn ml_backend.data.ingestion.api:app --reload
```

Test it:
```bash
curl http://localhost:8000/health
```

---

## Step 5: Implement Road Safety Models (Week 3-4)

### 5.1 IMU Classifier

Create: `ml_backend/models/road_safety/imu_classifier.py`

```python
import numpy as np
import pickle
from pathlib import Path
from xgboost import XGBClassifier
from ml_backend.config.model_configs import ROAD_SAFETY_CONFIGS

class IMUClassifier:
    """Classify road events from IMU data"""
    
    def __init__(self):
        config = ROAD_SAFETY_CONFIGS["imu_classifier"]
        self.model = XGBClassifier(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", 6),
            learning_rate=config.get("learning_rate", 0.1)
        )
        
        # Load model if exists
        model_path = config.get("model_path")
        if model_path and Path(model_path).exists():
            self.model.load_model(str(model_path))
    
    def extract_features(self, imu_data: list):
        """Extract features from IMU data sequence"""
        data = np.array(imu_data)
        
        features = []
        for axis in range(6):  # 3 accel + 3 gyro
            axis_data = data[:, axis]
            features.extend([
                np.mean(axis_data),
                np.std(axis_data),
                np.max(np.abs(axis_data)),
                np.min(axis_data)
            ])
        
        return np.array(features)
    
    def predict(self, imu_sequence: list):
        """Predict road event type"""
        features = self.extract_features(imu_sequence)
        prediction = self.model.predict([features])
        probability = self.model.predict_proba([features])
        
        return {
            "event_type": prediction[0],
            "confidence": float(np.max(probability[0]))
        }
```

### 5.2 Pothole Detection

Create: `ml_backend/models/road_safety/pothole_detection.py`

```python
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from ml_backend.config.model_configs import ROAD_SAFETY_CONFIGS

class PotholeDetector:
    """Detect potholes in road images"""
    
    def __init__(self):
        config = ROAD_SAFETY_CONFIGS["pothole_vision"]
        model_path = config.get("model_path")
        
        if model_path and Path(model_path).exists():
            self.model = YOLO(str(model_path))
        else:
            # Use pre-trained YOLOv8 and fine-tune later
            self.model = YOLO("yolov8m.pt")
        
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
    
    def detect(self, image: np.ndarray):
        """Detect potholes in image"""
        results = self.model(image, conf=self.confidence_threshold)
        
        detections = []
        if len(results) > 0:
            result = results[0]
            for box in result.boxes:
                if box.conf[0] > self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(box.conf[0])
                    })
        
        return {
            "potholes_detected": len(detections),
            "detections": detections
        }
```

---

## Step 6: Training Your First Model (Week 5)

### 6.1 Prepare Training Data

1. **Collect data**: Use mobile app to collect IMU and image data
2. **Label data**: Create labels for potholes, fatigue, etc.
3. **Organize data**: Put in `data/raw/` directory

### 6.2 Train IMU Classifier

Create: `notebooks/training/train_imu_classifier.ipynb`

```python
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = pd.read_csv("data/processed/imu_events.csv")

# Prepare features and labels
X = data.drop("event_type", axis=1).values
y = data["event_type"].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = XGBClassifier(n_estimators=100, max_depth=6)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# Save model
model.save_model("data/models/imu_classifier.pkl")
```

---

## Step 7: Create Inference API (Week 6)

### 7.1 Unified Inference Service

Create: `ml_backend/inference/api.py`

```python
from fastapi import FastAPI, File, UploadFile
from ml_backend.models.driver_safety import (
    FaceDetector, FatigueDetector, DistractionDetector
)
from ml_backend.models.road_safety import PotholeDetector
import cv2
import numpy as np

app = FastAPI()

# Initialize models
face_detector = FaceDetector()
fatigue_detector = FatigueDetector()
distraction_detector = DistractionDetector()
pothole_detector = PotholeDetector()

@app.post("/inference/driver-safety")
async def driver_safety_inference(file: UploadFile = File(...)):
    """Run driver safety models on image"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run models
    faces = face_detector.detect(image)
    fatigue = fatigue_detector.update(image)
    distraction = distraction_detector.detect(image)
    
    return {
        "faces_detected": len(faces),
        "fatigue": fatigue,
        "distraction": distraction
    }

@app.post("/inference/road-safety")
async def road_safety_inference(file: UploadFile = File(...)):
    """Run road safety models on image"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    potholes = pothole_detector.detect(image)
    
    return potholes
```

---

## Step 8: Evaluation & Monitoring (Week 7-8)

### 8.1 Create Evaluation Script

Create: `ml_backend/training/evaluation/evaluate_models.py`

```python
from ml_backend.config.model_configs import SUCCESS_METRICS
import json

def evaluate_driver_safety(true_positives, false_positives, false_negatives):
    """Evaluate driver safety models"""
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    }
    
    # Check against success metrics
    thresholds = SUCCESS_METRICS["driver_safety"]
    metrics["meets_fatigue_precision"] = precision >= thresholds["fatigue_precision_min"]
    
    return metrics

# Use this to evaluate your models
```

---

## Common Issues & Solutions

### Issue 1: Out of Memory
**Solution**: Reduce batch size, use smaller models, or process data in chunks

### Issue 2: Model Not Loading
**Solution**: Check file paths, ensure models are downloaded/trained first

### Issue 3: Slow Inference
**Solution**: 
- Use GPU if available
- Optimize image sizes
- Use model quantization
- Implement caching

### Issue 4: Low Accuracy
**Solution**:
- Collect more training data
- Improve data quality
- Tune hyperparameters
- Use data augmentation

---

## Next Steps

1. ✅ Complete basic models
2. ✅ Test with real data
3. ⏭️ Build training pipeline
4. ⏭️ Implement event fusion
5. ⏭️ Deploy to production

---

## Learning Resources

- **PyTorch Tutorial**: https://pytorch.org/tutorials/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **YOLOv8 Docs**: https://docs.ultralytics.com/
- **MediaPipe**: https://google.github.io/mediapipe/

---

## Getting Help

- Check the main plan: `ML_AI_BACKEND_PLAN.md`
- Review code comments
- Test each component individually
- Start simple, then add complexity

