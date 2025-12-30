"""Unit tests for calculator module."""

import pytest

from roof_ranker.utils.calculator import calculate_solar_potential


def test_calculate_solar_potential_baku() -> None:
    """Test solar potential calculation for Baku."""
    result = calculate_solar_potential(
        roof_pixels=1000,
        city_key="baku",
        pixel_to_meter=0.5,
    )

    assert result["city_name"] == "Baku, Azerbaijan"
    assert result["area_m2"] == 500.0  # 1000 pixels * 0.5
    assert result["capacity_kw"] > 0
    assert result["annual_savings_usd"] > 0
    assert result["co2_saved_tons"] > 0


def test_calculate_solar_potential_tashkent() -> None:
    """Test solar potential calculation for Tashkent."""
    result = calculate_solar_potential(
        roof_pixels=2000,
        city_key="tashkent",
        pixel_to_meter=0.5,
    )

    assert result["city_name"] == "Tashkent, Uzbekistan"
    assert result["area_m2"] == 1000.0
    assert result["capacity_kw"] > 0


def test_calculate_solar_potential_invalid_city() -> None:
    """Test that invalid city key defaults to Baku."""
    result = calculate_solar_potential(
        roof_pixels=1000,
        city_key="invalid_city",
        pixel_to_meter=0.5,
    )

    assert result["city_name"] == "Baku, Azerbaijan"


def test_calculate_solar_potential_custom_scale() -> None:
    """Test with custom pixel to meter scale."""
    result1 = calculate_solar_potential(roof_pixels=1000, city_key="baku", pixel_to_meter=0.5)
    result2 = calculate_solar_potential(roof_pixels=1000, city_key="baku", pixel_to_meter=1.0)

    # Area should double with double scale
    assert result2["area_m2"] == 2 * result1["area_m2"]
    # Capacity and savings should also increase proportionally
    assert result2["capacity_kw"] > result1["capacity_kw"]
    assert result2["annual_savings_usd"] > result1["annual_savings_usd"]


def test_calculate_solar_potential_zero_pixels() -> None:
    """Test with zero roof pixels."""
    result = calculate_solar_potential(
        roof_pixels=0,
        city_key="baku",
        pixel_to_meter=0.5,
    )

    assert result["area_m2"] == 0.0
    assert result["capacity_kw"] == 0.0
    assert result["annual_savings_usd"] == 0.0
    assert result["co2_saved_tons"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
