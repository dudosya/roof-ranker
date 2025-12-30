"""Configuration module for centralized application settings."""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SAMConfig:
    """Configuration for Segment Anything Model."""
    
    model_type: str = "vit_b"
    device: Optional[str] = None
    checkpoint_urls: Dict[str, str] = field(default_factory=lambda: {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    })
    weights_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "weights")
    
    def get_checkpoint_name(self, model_type: Optional[str] = None) -> str:
        """Get checkpoint filename for model type."""
        model_type = model_type or self.model_type
        if model_type == "vit_b":
            return f"sam_{model_type}_01ec64.pth"
        else:
            return f"sam_{model_type}_0b3195.pth"
    
    def get_checkpoint_path(self, model_type: Optional[str] = None) -> Path:
        """Get full checkpoint path for model type."""
        model_type = model_type or self.model_type
        checkpoint_name = self.get_checkpoint_name(model_type)
        return self.weights_dir / checkpoint_name


@dataclass
class GoogleMapsConfig:
    """Configuration for Google Maps API."""
    
    static_maps_url: str = "https://maps.googleapis.com/maps/api/staticmap"
    geocoding_url: str = "https://maps.googleapis.com/maps/api/geocode/json"
    default_zoom: int = 19
    default_size: str = "640x640"
    default_scale: int = 2
    default_maptype: str = "satellite"
    request_timeout: int = 30
    geocoding_timeout: int = 10


@dataclass
class SolarConfig:
    """Configuration for solar calculations."""
    
    # Solar panel efficiency (typical commercial panels)
    panel_efficiency: float = 0.18  # 18%
    
    # Performance ratio (accounts for losses: dirt, wiring, inverter, etc.)
    performance_ratio: float = 0.75  # 75%
    
    # Hours of sunlight per day (average)
    sunlight_hours_per_day: float = 4.5
    
    # Days per year
    days_per_year: int = 365
    
    # Default cost per kWh (USD)
    default_cost_per_kwh: float = 0.12
    
    # Default CO2 emissions per kWh (kg)
    default_co2_per_kwh: float = 0.5
    
    # Panel cost per kW (USD)
    panel_cost_per_kw: float = 1000.0
    
    # Installation cost multiplier
    installation_multiplier: float = 1.5  # 50% installation cost


@dataclass
class CitySolarData:
    """Solar data for a specific city."""
    
    name: str
    country: str
    latitude: float
    longitude: float
    solar_irradiance: float  # kWh/m²/day
    electricity_cost: float  # USD/kWh
    co2_emissions: float  # kg CO2/kWh
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "country": self.country,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "solar_irradiance": self.solar_irradiance,
            "electricity_cost": self.electricity_cost,
            "co2_emissions": self.co2_emissions,
        }


@dataclass
class AppConfig:
    """Main application configuration."""
    
    # Application metadata
    app_name: str = "Sol-Scout Pro"
    app_version: str = "1.0.0"
    app_description: str = "AI-Powered Solar Potential Analysis for Institutional Buildings"
    
    # Component configurations
    sam: SAMConfig = field(default_factory=SAMConfig)
    google_maps: GoogleMapsConfig = field(default_factory=GoogleMapsConfig)
    solar: SolarConfig = field(default_factory=SolarConfig)
    
    # City data
    cities: Dict[str, CitySolarData] = field(default_factory=dict)
    
    # Image processing
    default_image_size: tuple[int, int] = (640, 640)
    mask_overlay_alpha: float = 0.3
    mask_overlay_color: tuple[int, int, int] = (0, 0, 255)  # Blue
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    def __post_init__(self) -> None:
        """Initialize default city data."""
        if not self.cities:
            self._init_default_cities()
    
    def _init_default_cities(self) -> None:
        """Initialize default city data for CECECO capitals."""
        self.cities = {
            "baku": CitySolarData(
                name="Baku",
                country="Azerbaijan",
                latitude=40.4093,
                longitude=49.8671,
                solar_irradiance=4.2,  # kWh/m²/day
                electricity_cost=0.08,  # USD/kWh
                co2_emissions=0.45,  # kg CO2/kWh
            ),
            "tashkent": CitySolarData(
                name="Tashkent",
                country="Uzbekistan",
                latitude=41.2995,
                longitude=69.2401,
                solar_irradiance=4.5,  # kWh/m²/day
                electricity_cost=0.06,  # USD/kWh
                co2_emissions=0.55,  # kg CO2/kWh
            ),
            "astana": CitySolarData(
                name="Astana",
                country="Kazakhstan",
                latitude=51.1694,
                longitude=71.4491,
                solar_irradiance=3.8,  # kWh/m²/day
                electricity_cost=0.07,  # USD/kWh
                co2_emissions=0.60,  # kg CO2/kWh
            ),
            "bishkek": CitySolarData(
                name="Bishkek",
                country="Kyrgyzstan",
                latitude=42.8746,
                longitude=74.5698,
                solar_irradiance=4.0,  # kWh/m²/day
                electricity_cost=0.05,  # USD/kWh
                co2_emissions=0.40,  # kg CO2/kWh
            ),
        }
    
    def get_city(self, city_name: str) -> Optional[CitySolarData]:
        """Get city data by name (case-insensitive)."""
        city_key = city_name.lower()
        return self.cities.get(city_key)
    
    def get_city_names(self) -> list[str]:
        """Get list of available city names."""
        return [city.name for city in self.cities.values()]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "app_description": self.app_description,
            "sam": {
                "model_type": self.sam.model_type,
                "device": self.sam.device,
                "weights_dir": str(self.sam.weights_dir),
            },
            "google_maps": {
                "static_maps_url": self.google_maps.static_maps_url,
                "default_zoom": self.google_maps.default_zoom,
                "default_size": self.google_maps.default_size,
                "request_timeout": self.google_maps.request_timeout,
            },
            "solar": {
                "panel_efficiency": self.solar.panel_efficiency,
                "performance_ratio": self.solar.performance_ratio,
                "sunlight_hours_per_day": self.solar.sunlight_hours_per_day,
                "default_cost_per_kwh": self.solar.default_cost_per_kwh,
            },
            "cities": {key: city.to_dict() for key, city in self.cities.items()},
            "image_processing": {
                "default_image_size": self.default_image_size,
                "mask_overlay_alpha": self.mask_overlay_alpha,
                "mask_overlay_color": self.mask_overlay_color,
            },
            "logging": {
                "log_level": self.log_level,
                "log_file": str(self.log_file) if self.log_file else None,
            },
        }


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs: Any) -> None:
    """Update configuration with new values."""
    global config
    
    # Create a new config with updated values
    config_dict = config.to_dict()
    config_dict.update(kwargs)
    
    # Recreate config (simplified - in practice would need proper merging)
    # For now, just update the existing config object
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.sam, key):
            setattr(config.sam, key, value)
        elif hasattr(config.google_maps, key):
            setattr(config.google_maps, key, value)
        elif hasattr(config.solar, key):
            setattr(config.solar, key, value)
