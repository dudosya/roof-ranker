"""Quick test to verify core functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from roof_ranker.utils.constants import CITIES
from roof_ranker.utils.calculator import calculate_solar_potential


def test_constants() -> None:
    """Test that constants are loaded correctly."""
    print("Testing constants...")
    assert "baku" in CITIES
    assert "tashkent" in CITIES
    assert "astana" in CITIES
    assert "bishkek" in CITIES
    print("[OK] Constants loaded correctly")


def test_calculator() -> None:
    """Test calculator functionality."""
    print("\nTesting calculator...")
    result = calculate_solar_potential(roof_pixels=1000, city_key="baku")
    print(f"  Area: {result['area_m2']} m2")
    print(f"  Capacity: {result['capacity_kw']} kW")
    print(f"  Savings: ${result['annual_savings_usd']:,.2f}")
    print(f"  CO2: {result['co2_saved_tons']:.1f} tons/year")
    print("[OK] Calculator works")


def test_imports() -> None:
    """Test that all modules can be imported."""
    print("\nTesting imports...")
    try:
        from roof_ranker.utils import vision
        print("[OK] Vision module imports")
    except ImportError as e:
        print(f"[ERROR] Vision import failed: {e}")

    try:
        from roof_ranker import app
        print("[OK] App module imports")
    except ImportError as e:
        print(f"[ERROR] App import failed: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("Quick test of roof-ranker core functionality")
    print("=" * 50)

    test_constants()
    test_calculator()
    test_imports()

    print("\n" + "=" * 50)
    print("All quick tests completed!")
    print("=" * 50)
