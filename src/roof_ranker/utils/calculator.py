"""Solar potential and financial calculations."""

from typing import TypedDict

from .constants import CITIES, CO2_PER_TON, DAYS_PER_YEAR, DEFAULT_PIXEL_TO_METER


class SolarAnalysis(TypedDict):
    """Results of solar potential analysis."""

    area_m2: float
    capacity_kw: float
    annual_savings_usd: float
    co2_saved_tons: float
    city_name: str


def calculate_solar_potential(
    roof_pixels: int,
    city_key: str,
    pixel_to_meter: float = DEFAULT_PIXEL_TO_METER,
) -> SolarAnalysis:
    """
    Calculate solar potential and financial benefits.

    Args:
        roof_pixels: Total roof area in pixels
        city_key: Key from CITIES dictionary (e.g., 'baku', 'tashkent')
        pixel_to_meter: Conversion factor from pixels to square meters

    Returns:
        Dictionary containing analysis results
    """
    # Get city data
    city_data = CITIES.get(city_key.lower(), CITIES["baku"])

    # Calculate roof area in square meters
    area_m2 = roof_pixels * pixel_to_meter

    # Calculate solar capacity (kW)
    # Formula: Area × Solar Irradiance × Panel Efficiency
    daily_kwh = area_m2 * city_data["solar_irradiance"] * city_data["panel_efficiency"]
    capacity_kw = daily_kwh / city_data["sun_hours"]  # Convert to kW

    # Calculate annual energy production (kWh/year)
    annual_kwh = daily_kwh * DAYS_PER_YEAR

    # Calculate annual savings (USD/year)
    annual_savings_usd = annual_kwh * city_data["energy_cost"]

    # Calculate CO2 savings (tons/year)
    co2_saved_kg = annual_kwh * city_data["co2_per_kwh"]
    co2_saved_tons = co2_saved_kg / CO2_PER_TON

    return {
        "area_m2": round(area_m2, 2),
        "capacity_kw": round(capacity_kw, 2),
        "annual_savings_usd": round(annual_savings_usd, 2),
        "co2_saved_tons": round(co2_saved_tons, 2),
        "city_name": city_data["name"],
    }


def format_currency(amount: float) -> str:
    """Format currency with USD symbol and commas."""
    return f"${amount:,.2f}"


def format_area(area_m2: float) -> str:
    """Format area with units."""
    return f"{area_m2:,.1f} m²"


def format_power(capacity_kw: float) -> str:
    """Format power capacity."""
    return f"{capacity_kw:,.1f} kW"
