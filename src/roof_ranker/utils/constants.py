"""City-specific solar data for CECECO capitals (legacy module)."""

from typing import TypedDict
import warnings

from roof_ranker.config import get_config


class CityData(TypedDict):
    """Solar and economic data for a city (legacy)."""

    name: str
    solar_irradiance: float  # kWh/m²/day
    energy_cost: float  # USD/kWh
    sun_hours: float  # hours/day
    panel_efficiency: float  # decimal (e.g., 0.2 for 20%)
    co2_per_kwh: float  # kg CO2/kWh saved


# Data for CECECO capitals (legacy - use config module instead)
CITIES: dict[str, CityData] = {
    "baku": {
        "name": "Baku, Azerbaijan",
        "solar_irradiance": 4.2,  # Moderate sun, high wind
        "energy_cost": 0.08,
        "sun_hours": 5.2,
        "panel_efficiency": 0.18,
        "co2_per_kwh": 0.45,
    },
    "tashkent": {
        "name": "Tashkent, Uzbekistan",
        "solar_irradiance": 5.1,  # High sun
        "energy_cost": 0.06,
        "sun_hours": 6.1,
        "panel_efficiency": 0.19,
        "co2_per_kwh": 0.42,
    },
    "astana": {
        "name": "Astana, Kazakhstan",
        "solar_irradiance": 3.5,  # Low sun, high heating need
        "energy_cost": 0.05,
        "sun_hours": 4.3,
        "panel_efficiency": 0.17,
        "co2_per_kwh": 0.48,
    },
    "bishkek": {
        "name": "Bishkek, Kyrgyzstan",
        "solar_irradiance": 4.5,  # Moderate sun
        "energy_cost": 0.04,
        "sun_hours": 5.5,
        "panel_efficiency": 0.18,
        "co2_per_kwh": 0.44,
    },
}

# Default values (legacy - use config module instead)
DEFAULT_PIXEL_TO_METER = 0.5  # 1 pixel = 0.5 square meters
DEFAULT_HSV_LOWER = (0, 0, 100)  # Light gray/white lower bound (H,S,V)
DEFAULT_HSV_UPPER = (180, 50, 255)  # Light gray/white upper bound

# Solar constants (legacy - use config module instead)
DAYS_PER_YEAR = 365
CO2_PER_TON = 1000  # kg per ton


def get_city_data(city_name: str) -> CityData:
    """
    Get city data by name (legacy function).
    
    Args:
        city_name: City name (case-insensitive)
    
    Returns:
        CityData dictionary
        
    Deprecated: Use config.get_config().get_city() instead
    """
    warnings.warn(
        "get_city_data is deprecated. Use config.get_config().get_city() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    city_key = city_name.lower()
    if city_key not in CITIES:
        raise ValueError(f"Unknown city: {city_name}")
    
    return CITIES[city_key]


def get_city_names() -> list[str]:
    """
    Get list of available city names (legacy function).
    
    Returns:
        List of city names
        
    Deprecated: Use config.get_config().get_city_names() instead
    """
    warnings.warn(
        "get_city_names is deprecated. Use config.get_config().get_city_names() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return [data["name"] for data in CITIES.values()]


# Compatibility imports for backward compatibility
__all__ = [
    "CityData",
    "CITIES",
    "DEFAULT_PIXEL_TO_METER",
    "DEFAULT_HSV_LOWER",
    "DEFAULT_HSV_UPPER",
    "DAYS_PER_YEAR",
    "CO2_PER_TON",
    "get_city_data",
    "get_city_names",
]
