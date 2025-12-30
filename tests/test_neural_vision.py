"""Unit tests for neural_vision module."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from roof_ranker.utils.neural_vision import (
    SAMRoofSegmenter,
    create_mask_overlay,
    count_mask_pixels,
)


def test_create_mask_overlay() -> None:
    """Test mask overlay creation."""
    # Create a test image (3x3 RGB)
    image = np.zeros((3, 3, 3), dtype=np.uint8)
    image[:, :] = [100, 100, 100]  # Gray background
    
    # Create a mask (center pixel is True)
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    
    # Create overlay with blue color
    overlay = create_mask_overlay(image, mask, color=(0, 0, 255))
    
    # Check that overlay has correct shape
    assert overlay.shape == (3, 3, 3)
    assert overlay.dtype == np.uint8
    
    # Check that center pixel is blended with blue
    # Original: [100, 100, 100], Blue: [0, 0, 255], Alpha: 0.3
    # Expected: [100*0.7 + 0*0.3, 100*0.7 + 0*0.3, 100*0.7 + 255*0.3]
    # = [70, 70, 70 + 76.5 = 146.5] -> [70, 70, 147]
    expected_center = np.array([70, 70, 147], dtype=np.uint8)
    assert np.allclose(overlay[1, 1], expected_center, atol=1)
    
    # Check that other pixels remain unchanged
    expected_other = np.array([100, 100, 100], dtype=np.uint8)
    assert np.array_equal(overlay[0, 0], expected_other)
    assert np.array_equal(overlay[0, 1], expected_other)
    assert np.array_equal(overlay[0, 2], expected_other)


def test_create_mask_overlay_empty_mask() -> None:
    """Test mask overlay with empty mask."""
    image = np.ones((2, 2, 3), dtype=np.uint8) * 100
    mask = np.zeros((2, 2), dtype=bool)
    
    overlay = create_mask_overlay(image, mask)
    
    # With empty mask, overlay should be identical to original
    assert np.array_equal(overlay, image)


def test_create_mask_overlay_full_mask() -> None:
    """Test mask overlay with full mask."""
    image = np.ones((2, 2, 3), dtype=np.uint8) * 100
    mask = np.ones((2, 2), dtype=bool)
    
    overlay = create_mask_overlay(image, mask, color=(255, 0, 0))
    
    # All pixels should be blended with red
    assert overlay.shape == image.shape
    # Check that values changed (not equal to original)
    assert not np.array_equal(overlay, image)


def test_count_mask_pixels() -> None:
    """Test counting pixels in mask."""
    # Test with all False
    mask = np.zeros((3, 3), dtype=bool)
    assert count_mask_pixels(mask) == 0
    
    # Test with all True
    mask = np.ones((3, 3), dtype=bool)
    assert count_mask_pixels(mask) == 9
    
    # Test with some True
    mask = np.array([
        [True, False, True],
        [False, True, False],
        [True, False, True]
    ], dtype=bool)
    assert count_mask_pixels(mask) == 5


def test_count_mask_pixels_empty() -> None:
    """Test counting pixels in empty mask."""
    mask = np.zeros((0, 0), dtype=bool)
    assert count_mask_pixels(mask) == 0


@patch('roof_ranker.utils.neural_vision.sam_model_registry')
@patch('roof_ranker.utils.neural_vision.Path')
@patch('roof_ranker.utils.neural_vision.urllib.request')
def test_sam_roof_segmenter_init(mock_urllib, mock_path, mock_registry) -> None:
    """Test SAMRoofSegmenter initialization."""
    # Mock checkpoint path exists
    mock_checkpoint = Mock()
    mock_checkpoint.exists.return_value = True
    mock_path.return_value.__truediv__.return_value = mock_checkpoint
    
    # Mock model
    mock_model = Mock()
    mock_registry.return_value = mock_model
    
    # Mock predictor
    mock_predictor = Mock()
    
    # Create segmenter
    segmenter = SAMRoofSegmenter(model_type="vit_b", device="cpu")
    
    # Check initialization
    assert segmenter.device == "cpu"
    assert segmenter.model_type == "vit_b"
    assert segmenter.model is not None
    assert segmenter.predictor is not None


@patch('roof_ranker.utils.neural_vision.sam_model_registry')
@patch('roof_ranker.utils.neural_vision.Path')
@patch('roof_ranker.utils.neural_vision.urllib.request')
def test_sam_roof_segmenter_segment_roof(mock_urllib, mock_path, mock_registry) -> None:
    """Test SAMRoofSegmenter.segment_roof method."""
    # Mock checkpoint path exists
    mock_checkpoint = Mock()
    mock_checkpoint.exists.return_value = True
    mock_path.return_value.__truediv__.return_value = mock_checkpoint
    
    # Mock model and predictor
    mock_model = Mock()
    mock_registry.return_value = mock_model
    
    mock_predictor = Mock()
    mock_predictor.predict.return_value = (
        np.array([[[True, False], [False, True]]]),  # masks
        np.array([0.9]),  # scores
        None  # logits
    )
    
    # Create segmenter with mocked predictor
    segmenter = SAMRoofSegmenter(model_type="vit_b", device="cpu")
    segmenter.predictor = mock_predictor
    
    # Test image and input point
    image = np.ones((2, 2, 3), dtype=np.uint8) * 255
    input_point = (1, 1)
    
    # Call segment_roof
    mask = segmenter.segment_roof(image, input_point)
    
    # Verify calls
    mock_predictor.set_image.assert_called_once_with(image)
    
    # Check that predict was called with correct arguments (loose check)
    mock_predictor.predict.assert_called_once()
    call_args = mock_predictor.predict.call_args
    
    # Check keyword arguments
    assert 'multimask_output' in call_args[1]
    assert call_args[1]['multimask_output'] == True
    
    # Check returned mask
    assert mask.shape == (2, 2)
    assert mask.dtype == bool


def test_sam_roof_segmenter_segment_roof_no_predictor() -> None:
    """Test SAMRoofSegmenter.segment_roof with no predictor."""
    segmenter = SAMRoofSegmenter.__new__(SAMRoofSegmenter)
    segmenter.predictor = None
    
    image = np.ones((2, 2, 3), dtype=np.uint8)
    input_point = (1, 1)
    
    with pytest.raises(RuntimeError, match="SAM predictor not initialized"):
        segmenter.segment_roof(image, input_point)


@patch('roof_ranker.utils.neural_vision.np.array')
def test_sam_roof_segmenter_segment_roof_from_pil(mock_array) -> None:
    """Test SAMRoofSegmenter.segment_roof_from_pil method."""
    # Mock segmenter
    segmenter = SAMRoofSegmenter.__new__(SAMRoofSegmenter)
    mock_mask = np.array([[True, False], [False, True]])
    segmenter.segment_roof = Mock(return_value=mock_mask)
    
    # Mock PIL Image
    mock_image = Mock()
    mock_image_array = np.ones((2, 2, 3), dtype=np.uint8)
    mock_array.return_value = mock_image_array
    
    # Call segment_roof_from_pil
    input_point = (1, 1)
    mask = segmenter.segment_roof_from_pil(mock_image, input_point)
    
    # Verify call
    segmenter.segment_roof.assert_called_once()
    call_args = segmenter.segment_roof.call_args[0]
    assert call_args[0].shape == (2, 2, 3)
    assert call_args[1] == input_point
    
    # Check returned mask
    assert mask is mock_mask  # Should return the same mock mask


@patch('roof_ranker.utils.neural_vision.sam_model_registry')
@patch('roof_ranker.utils.neural_vision.Path')
@patch('roof_ranker.utils.neural_vision.urllib.request')
def test_sam_roof_segmenter_segment_roof_with_box(mock_urllib, mock_path, mock_registry) -> None:
    """Test SAMRoofSegmenter.segment_roof_with_box method."""
    # Mock checkpoint path exists
    mock_checkpoint = Mock()
    mock_checkpoint.exists.return_value = True
    mock_path.return_value.__truediv__.return_value = mock_checkpoint
    
    # Mock model and predictor
    mock_model = Mock()
    mock_registry.return_value = mock_model
    
    mock_predictor = Mock()
    mock_predictor.predict.return_value = (
        np.array([[[True, False], [False, True]]]),  # masks
        np.array([0.9]),  # scores
        None  # logits
    )
    
    # Create segmenter with mocked predictor
    segmenter = SAMRoofSegmenter(model_type="vit_b", device="cpu")
    segmenter.predictor = mock_predictor
    
    # Test image and input box
    image = np.ones((2, 2, 3), dtype=np.uint8) * 255
    box = (0, 0, 1, 1)  # x1, y1, x2, y2
    
    # Call segment_roof_with_box
    mask = segmenter.segment_roof_with_box(image, box)
    
    # Verify calls
    mock_predictor.set_image.assert_called_once_with(image)
    
    # Check that predict was called with correct arguments
    mock_predictor.predict.assert_called_once()
    call_args = mock_predictor.predict.call_args
    
    # Check keyword arguments - should have box parameter
    assert 'box' in call_args[1]
    assert np.array_equal(call_args[1]['box'], np.array(box))
    assert 'multimask_output' in call_args[1]
    assert call_args[1]['multimask_output'] == True
    
    # Check returned mask
    assert mask.shape == (2, 2)
    assert mask.dtype == bool


def test_sam_roof_segmenter_segment_roof_with_box_no_predictor() -> None:
    """Test SAMRoofSegmenter.segment_roof_with_box with no predictor."""
    segmenter = SAMRoofSegmenter.__new__(SAMRoofSegmenter)
    segmenter.predictor = None
    
    image = np.ones((2, 2, 3), dtype=np.uint8)
    box = (0, 0, 1, 1)
    
    with pytest.raises(RuntimeError, match="SAM predictor not initialized"):
        segmenter.segment_roof_with_box(image, box)


if __name__ == "__main__":
    pytest.main([__file__])
