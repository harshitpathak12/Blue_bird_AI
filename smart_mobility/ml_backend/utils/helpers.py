"""
Helper utility functions
"""
import numpy as np
import cv2
from typing import Tuple, Optional
from pathlib import Path


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array or None if failed
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                keep_aspect: bool = True) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Input image
        target_size: (width, height) target size
        keep_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if keep_aspect:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad if needed
        if new_w != target_w or new_h != target_h:
            pad_w = (target_w - new_w) // 2
            pad_h = (target_h - new_h) // 2
            resized = cv2.copyMakeBorder(
                resized, pad_h, target_h - new_h - pad_h,
                pad_w, target_w - new_w - pad_w,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        return resized
    else:
        return cv2.resize(image, target_size)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image


def denormalize_image(image: np.ndarray) -> np.ndarray:
    """
    Denormalize image from [0, 1] to [0, 255]
    
    Args:
        image: Normalized image
        
    Returns:
        Denormalized image
    """
    if image.max() <= 1.0:
        return (image * 255).astype(np.uint8)
    return image.astype(np.uint8)


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU score
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save image to file
    
    Args:
        image: Image to save
        output_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False
