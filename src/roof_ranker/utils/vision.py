"""Computer vision module for roof detection using OpenCV."""

from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from .constants import DEFAULT_HSV_LOWER, DEFAULT_HSV_UPPER


def detect_roof(
    image: np.ndarray,
    hsv_lower: Tuple[int, int, int] = DEFAULT_HSV_LOWER,
    hsv_upper: Tuple[int, int, int] = DEFAULT_HSV_UPPER,
    min_area: int = 500,
    max_area: int = 50000,
) -> Tuple[np.ndarray, int]:
    """
    Detect roof areas in an image using color segmentation.

    Args:
        image: Input image in BGR format (OpenCV default)
        hsv_lower: Lower bound for HSV thresholding
        hsv_upper: Upper bound for HSV thresholding
        min_area: Minimum contour area to consider as roof
        max_area: Maximum contour area to consider as roof

    Returns:
        Tuple of (processed_image, total_roof_pixels)
        processed_image has green semi-transparent overlay on detected roofs
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Convert tuple bounds to numpy arrays for OpenCV
    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)

    # Create mask for roof colors
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area
    roof_contours = []
    total_roof_pixels = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            roof_contours.append(contour)
            total_roof_pixels += int(area)

    # Create overlay
    overlay = image.copy()
    cv2.drawContours(overlay, roof_contours, -1, (0, 255, 0), 2)

    # Fill contours with semi-transparent green
    for contour in roof_contours:
        cv2.fillPoly(overlay, [contour], (0, 255, 0))

    # Blend overlay with original image (semi-transparent)
    alpha = 0.3
    processed = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw contour outlines on top
    cv2.drawContours(processed, roof_contours, -1, (0, 255, 0), 2)

    return processed, total_roof_pixels


def load_image(image_path: str) -> np.ndarray:
    """Load an image from file path and convert to OpenCV BGR format."""
    pil_image = Image.open(image_path)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR image to RGB for display."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(image: np.ndarray, max_width: int = 800) -> np.ndarray:
    """Resize image maintaining aspect ratio."""
    height, width = image.shape[:2]
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        return cv2.resize(image, (max_width, new_height))
    return image
