"""
Crack Segmentation using UNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from pathlib import Path
from ml_backend.config.model_configs import ROAD_SAFETY_CONFIGS


class UNet(nn.Module):
    """UNet architecture for crack segmentation"""
    
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder
        self.dec4 = self._conv_block(1024, 512)
        self.dec3 = self._conv_block(512, 256)
        self.dec2 = self._conv_block(256, 128)
        self.dec1 = self._conv_block(128, 64)
        
        # Output
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _conv_block(self, in_channels: int, out_channels: int):
        """Convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        d4 = self.upsample(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upsample(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upsample(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upsample(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output
        out = self.final(d1)
        return out


class CrackSegmenter:
    """
    Segment road cracks using UNet
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize crack segmenter
        
        Args:
            config: Configuration dictionary
        """
        if config is None:
            config = ROAD_SAFETY_CONFIGS["crack_segmentation"]
        
        self.input_size = config.get("input_size", (512, 512))
        self.num_classes = config.get("num_classes", 2)
        
        # Initialize model
        self.model = UNet(in_channels=3, num_classes=self.num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Load model if exists
        model_path = config.get("model_path")
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.is_trained = True
        else:
            self.is_trained = False
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed tensor
        """
        # Resize
        image_resized = cv2.resize(image, self.input_size)
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image_norm = image_rgb.astype(np.float32) / 255.0
        image_norm = (image_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor
    
    def segment(self, image: np.ndarray) -> Dict:
        """
        Segment cracks in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dictionary with segmentation results
        """
        if not self.is_trained:
            return {
                "cracks_detected": False,
                "coverage_ratio": 0.0,
                "severity": "none",
                "message": "Model not trained yet"
            }
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = F.softmax(output, dim=1)
            mask = torch.argmax(prediction, dim=1).cpu().numpy()[0]
        
        # Calculate metrics
        crack_pixels = np.sum(mask == 1)
        total_pixels = mask.size
        coverage_ratio = crack_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # Determine severity
        if coverage_ratio > 0.1:
            severity = "critical"
        elif coverage_ratio > 0.05:
            severity = "high"
        elif coverage_ratio > 0.02:
            severity = "medium"
        elif coverage_ratio > 0.01:
            severity = "low"
        else:
            severity = "none"
        
        # Resize mask back to original size
        original_size = (image.shape[1], image.shape[0])
        mask_resized = cv2.resize(mask.astype(np.uint8), original_size, 
                                  interpolation=cv2.INTER_NEAREST)
        
        return {
            "cracks_detected": coverage_ratio > 0.01,
            "coverage_ratio": float(coverage_ratio),
            "severity": severity,
            "mask": mask_resized,
            "crack_pixels": int(crack_pixels),
            "total_pixels": int(total_pixels)
        }
    
    def visualize(self, image: np.ndarray, segmentation_result: Dict) -> np.ndarray:
        """
        Visualize segmentation on image
        
        Args:
            image: Original image
            segmentation_result: Result from segment() method
            
        Returns:
            Image with segmentation overlay
        """
        vis_image = image.copy()
        
        if "mask" in segmentation_result:
            mask = segmentation_result["mask"]
            
            # Create colored overlay
            overlay = np.zeros_like(vis_image)
            overlay[mask == 1] = [0, 0, 255]  # Red for cracks
            
            # Blend overlay
            vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
        
        return vis_image
