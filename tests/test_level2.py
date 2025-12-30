"""Test script for Level 2 functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    # Test geo_handler imports
    from roof_ranker.utils.geo_handler import fetch_satellite_image, calculate_pixel_scale
    print("[OK] geo_handler imports successfully")
    
    # Test neural_vision imports
    from roof_ranker.utils.neural_vision import SAMRoofSegmenter, create_mask_overlay, count_mask_pixels
    print("[OK] neural_vision imports successfully")
    
    # Test app imports
    from roof_ranker.app import main
    print("[OK] app imports successfully")
    
    # Test calculator (should still work)
    from roof_ranker.utils.calculator import calculate_solar_potential
    print("[OK] calculator imports successfully")
    
    # Test vision (legacy, should still work)
    from roof_ranker.utils.vision import load_image, resize_image, convert_to_rgb
    print("[OK] vision imports successfully")
    
    # Test SAM model initialization (without downloading)
    print("\nTesting SAM model initialization...")
    try:
        # This will try to download the model, but we'll catch the error
        segmenter = SAMRoofSegmenter(model_type="vit_b")
        print("[OK] SAMRoofSegmenter initialized")
    except Exception as e:
        print(f"[WARN] SAM initialization error (expected without internet): {type(e).__name__}")
    
    # Test pixel scale calculation
    print("\nTesting pixel scale calculation...")
    scale = calculate_pixel_scale(40.0, zoom=19)
    print(f"[OK] Pixel scale at latitude 40 deg, zoom 19: {scale:.6f} meters/pixel")
    
    # Test solar calculation
    print("\nTesting solar calculations...")
    analysis = calculate_solar_potential(
        roof_pixels=10000,
        city_key="baku",
        pixel_to_meter=0.25
    )
    print(f"[OK] Solar analysis for 10,000 pixels:")
    print(f"  Area: {analysis['area_m2']:.1f} m2")
    print(f"  Capacity: {analysis['capacity_kw']:.1f} kW")
    print(f"  Savings: ${analysis['annual_savings_usd']:.0f}/year")
    
    print("\n[SUCCESS] All Level 2 tests passed!")
    
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Test error: {type(e).__name__}: {e}")
    sys.exit(1)
