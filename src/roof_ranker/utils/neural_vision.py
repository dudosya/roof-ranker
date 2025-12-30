"""Neural vision module for roof segmentation using Segment Anything Model (SAM)."""

import os
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry


class SAMRoofSegmenter:
    """Wrapper for SAM model for roof segmentation."""
    
    def __init__(self, model_type: str = "vit_b", device: Optional[str] = None):
        """
        Initialize SAM model.
        
        Args:
            model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_type = model_type
        self.model = None
        self.predictor = None
        
        # Model checkpoint URLs
        self.model_urls = {
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        }
        
        # Initialize model
        self._load_model()
    
    def _download_checkpoint(self, checkpoint_path: Path, model_type: str) -> None:
        """Download SAM checkpoint if not exists."""
        if checkpoint_path.exists():
            return
        
        print(f"Downloading SAM {model_type} checkpoint...")
        url = self.model_urls[model_type]
        
        # Create weights directory if it doesn't exist
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress bar
        try:
            urllib.request.urlretrieve(url, checkpoint_path)
            print(f"Downloaded checkpoint to {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download checkpoint: {e}")
    
    def _load_model(self) -> None:
        """Load SAM model and create predictor."""
        # Determine checkpoint path
        weights_dir = Path(__file__).parent.parent.parent / "weights"
        checkpoint_name = f"sam_{self.model_type}_01ec64.pth" if self.model_type == "vit_b" else f"sam_{self.model_type}_0b3195.pth"
        checkpoint_path = weights_dir / checkpoint_name
        
        # Download checkpoint if needed
        self._download_checkpoint(checkpoint_path, self.model_type)
        
        # Load model
        sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        
        self.model = sam
        self.predictor = SamPredictor(sam)
    
    def segment_roof(self, image: np.ndarray, input_point: Tuple[int, int]) -> np.ndarray:
        """
        Segment roof from image using click point.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
            input_point: Click coordinates (x, y) in image coordinates
        
        Returns:
            Boolean mask of roof area (True = roof, False = background)
        """
        if self.predictor is None:
            raise RuntimeError("SAM predictor not initialized")
        
        # Set image in predictor (computes image embedding)
        self.predictor.set_image(image)
        
        # Prepare input points and labels
        # SAM expects points in (x, y) format with foreground label=1
        input_points = np.array([input_point])
        input_labels = np.array([1])  # 1 = foreground point
        
        # Predict mask
        masks, scores, _ = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,  # Return multiple masks
        )
        
        # Select the best mask (highest score)
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx]
        
        return mask
    
    def segment_roof_with_box(self, image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Segment roof from image using bounding box.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
            box: Bounding box coordinates (x1, y1, x2, y2) where
                 (x1, y1) is top-left and (x2, y2) is bottom-right
        
        Returns:
            Boolean mask of roof area (True = roof, False = background)
        """
        if self.predictor is None:
            raise RuntimeError("SAM predictor not initialized")
        
        # Debug logging
        print(f"[DEBUG] segment_roof_with_box called with:")
        print(f"  Image shape: {image.shape}")
        print(f"  Box coordinates: {box}")
        print(f"  Box type: {type(box)}")
        
        # Validate box coordinates
        x1, y1, x2, y2 = box
        height, width = image.shape[:2]
        
        # Check box validity
        if not (0 <= x1 < width and 0 <= x2 < width and 0 <= y1 < height and 0 <= y2 < height):
            print(f"[WARNING] Box coordinates out of bounds: {box} vs image {width}x{height}")
        
        if x2 <= x1 or y2 <= y1:
            print(f"[WARNING] Invalid box dimensions: x2={x2} <= x1={x1} or y2={y2} <= y1={y1}")
        
        # Calculate box dimensions
        box_width = x2 - x1
        box_height = y2 - y1
        print(f"  Box dimensions: {box_width}x{box_height}")
        
        # Set image in predictor (computes image embedding)
        self.predictor.set_image(image)
        
        # Prepare input box
        # SAM expects box in (x1, y1, x2, y2) format
        input_box = np.array(box)
        print(f"  Input box to SAM: {input_box}")
        
        # Predict mask with box prompt
        masks, scores, _ = self.predictor.predict(
            box=input_box,
            multimask_output=True,  # Return multiple masks
        )
        
        print(f"  Number of masks returned: {len(masks)}")
        print(f"  Mask scores: {scores}")
        
        # Select the best mask (highest score)
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx]
        
        # Count pixels in mask
        mask_pixels = np.sum(mask)
        print(f"  Selected mask {best_mask_idx} with score {scores[best_mask_idx]:.3f}")
        print(f"  Mask pixels: {mask_pixels} ({(mask_pixels / (height * width) * 100):.1f}% of image)")
        
        return mask
    
    def segment_roof_from_pil(self, image: Image.Image, input_point: Tuple[int, int]) -> np.ndarray:
        """
        Segment roof from PIL Image.
        
        Args:
            image: PIL Image
            input_point: Click coordinates (x, y)
        
        Returns:
            Boolean mask of roof area
        """
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Ensure image is in RGB format
        if len(image_np.shape) == 2:  # Grayscale
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[2] == 4:  # RGBA
            image_np = image_np[:, :, :3]
        
        return self.segment_roof(image_np, input_point)


def create_mask_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """
    Create overlay of mask on image.
    
    Args:
        image: Original image (H, W, 3)
        mask: Boolean mask (H, W)
        color: RGB color for overlay (default: blue)
    
    Returns:
        Image with mask overlay
    """
    # Create colored mask
    colored_mask: np.ndarray = np.zeros_like(image)
    colored_mask[mask] = color
    
    # Blend with original image
    alpha = 0.3
    overlay: np.ndarray = image.copy()
    overlay[mask] = (overlay[mask] * (1 - alpha) + colored_mask[mask] * alpha).astype(np.uint8)
    
    return overlay


def count_mask_pixels(mask: np.ndarray) -> int:
    """Count number of True pixels in mask."""
    return int(np.sum(mask))
