"""Geospatial handler for address-to-image conversion using Google Maps Static API."""

import math
import logging
from typing import Tuple, Optional
from io import BytesIO

import requests
from geopy.geocoders import Nominatim
from PIL import Image

# Set up logging
logger = logging.getLogger(__name__)


def geocode_address(address: str, api_key: Optional[str] = None) -> Tuple[float, float]:
    """
    Geocode address using Google Maps Geocoding API with Nominatim fallback.
    
    Args:
        address: String address to geocode
        api_key: Optional Google Maps API key
        
    Returns:
        Tuple of (latitude, longitude)
        
    Raises:
        ValueError: If address cannot be geocoded by any service
    """
    # Validate address
    if not address or not address.strip():
        raise ValueError("Address cannot be empty")
    
    # Try Google Maps Geocoding API first if API key is provided
    if api_key and api_key.strip():
        try:
            logger.info(f"Attempting Google Maps geocoding for: {address}")
            return _geocode_with_google(address, api_key)
        except Exception as google_error:
            logger.warning(f"Google Maps geocoding failed: {google_error}. Falling back to Nominatim.")
    
    # Fall back to Nominatim
    try:
        logger.info(f"Attempting Nominatim geocoding for: {address}")
        return _geocode_with_nominatim(address)
    except Exception as nominatim_error:
        logger.error(f"All geocoding attempts failed for: {address}")
        raise ValueError(
            f"Could not geocode address: '{address}'. "
            f"Please check the address format and try again.\n"
            f"Tip: Use format 'Street, City, Country' (e.g., 'Baku State University, Baku, Azerbaijan')"
        ) from nominatim_error


def _geocode_with_google(address: str, api_key: str) -> Tuple[float, float]:
    """
    Geocode address using Google Maps Geocoding API.
    
    Args:
        address: String address to geocode
        api_key: Google Maps API key
        
    Returns:
        Tuple of (latitude, longitude)
        
    Raises:
        ValueError: If geocoding fails
    """
    try:
        import googlemaps
        
        # Initialize Google Maps client
        gmaps = googlemaps.Client(key=api_key)
        
        # Geocode the address
        geocode_result = gmaps.geocode(address)
        
        if not geocode_result:
            raise ValueError(f"No results found for address: '{address}'")
        
        # Extract latitude and longitude
        location = geocode_result[0]['geometry']['location']
        lat = location['lat']
        lng = location['lng']
        
        logger.info(f"Google Maps geocoding successful: {address} -> ({lat}, {lng})")
        return lat, lng
        
    except ImportError:
        raise ValueError("Google Maps client not available. Please install 'googlemaps' package.")
    except Exception as e:
        raise ValueError(f"Google Maps geocoding failed: {e}")


def _geocode_with_nominatim(address: str) -> Tuple[float, float]:
    """
    Geocode address using Nominatim (OpenStreetMap).
    
    Args:
        address: String address to geocode
        
    Returns:
        Tuple of (latitude, longitude)
        
    Raises:
        ValueError: If geocoding fails
    """
    try:
        geolocator = Nominatim(user_agent="roof-ranker", timeout=10)
        location = geolocator.geocode(address)
        
        if location is None:
            raise ValueError(f"No results found for address: '{address}'")
        
        lat, lon = location.latitude, location.longitude
        logger.info(f"Nominatim geocoding successful: {address} -> ({lat}, {lon})")
        return lat, lon
        
    except Exception as e:
        if "timed out" in str(e).lower():
            raise ValueError(f"Geocoding timed out for address: '{address}'. Please try again.") from e
        else:
            raise ValueError(f"Geocoding failed: {e}") from e


def fetch_satellite_image(address: str, api_key: str) -> Tuple[Image.Image, float]:
    """
    Fetch satellite image for a given address using Google Maps Static API.

    Args:
        address: String address (e.g., "Baku State University")
        api_key: Google Maps Static API key

    Returns:
        Tuple of (PIL Image, area_per_pixel in m²)
    
    Raises:
        ValueError: If address is empty or API key is empty
        GeocoderTimedOut: If geocoding times out
        GeocoderServiceError: If geocoding service fails
        requests.exceptions.RequestException: If API request fails
        IOError: If image cannot be loaded from response
    """
    # Validate inputs
    if not address or not address.strip():
        raise ValueError("Address cannot be empty")
    
    if not api_key or not api_key.strip():
        raise ValueError("Google Maps API key cannot be empty")
    
    # Step 1: Geocoding - convert address to lat/lon
    try:
        lat, lon = geocode_address(address, api_key)
        
    except Exception as e:
        # Re-raise the error with improved message
        if "Could not geocode address" in str(e):
            raise e
        elif "timed out" in str(e).lower():
            raise ValueError(f"Geocoding timed out for address: '{address}'. Please try again.") from e
        else:
            raise ValueError(f"Geocoding failed for address: '{address}'. Error: {e}") from e
    
    # Step 2: Calculate meters_per_pixel using Mercator projection
    # Formula: meters_per_pixel = 156543.03392 * cos(lat * pi / 180) / (2 ^ zoom)
    zoom = 19  # As specified in spec
    meters_per_pixel = 156543.03392 * math.cos(lat * math.pi / 180) / (2 ** zoom)
    
    # Step 3: Request image from Google Static Maps API
    # Parameters: zoom=19, scale=2 (for higher resolution), maptype='satellite'
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": "640x640",  # Standard size
        "scale": 2,  # Double resolution
        "maptype": "satellite",
        "key": api_key,
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        
    except requests.exceptions.Timeout:
        raise ValueError("Google Maps API request timed out. Please try again.")
    except requests.exceptions.HTTPError as e:
        if response.status_code == 403:
            raise ValueError("Google Maps API key is invalid or doesn't have Static Maps API enabled.")
        elif response.status_code == 400:
            raise ValueError(f"Invalid request parameters. Please check the address: '{address}'.")
        else:
            raise ValueError(f"Google Maps API request failed with status {response.status_code}: {e}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to connect to Google Maps API: {e}")
    
    # Step 4: Convert response to PIL Image
    try:
        image = Image.open(BytesIO(response.content))
    except IOError as e:
        raise ValueError(f"Failed to load image from Google Maps API response: {e}")
    
    # Step 5: Calculate area_per_pixel (m² per pixel)
    # Since scale=2, the actual pixel density is doubled
    # The image size is 640x640 but scale=2 means 1280x1280 effective pixels
    # meters_per_pixel is for zoom level 19 at scale=1
    # At scale=2, each pixel represents half the distance
    effective_meters_per_pixel = meters_per_pixel / 2
    area_per_pixel = effective_meters_per_pixel ** 2
    
    return image, area_per_pixel


def calculate_pixel_scale(latitude: float, zoom: int = 19) -> float:
    """
    Calculate meters per pixel for a given latitude and zoom level.
    
    Args:
        latitude: Latitude in degrees
        zoom: Google Maps zoom level (default 19)
    
    Returns:
        Meters per pixel at scale=1
    """
    result: float = 156543.03392 * math.cos(latitude * math.pi / 180) / (2 ** zoom)
    return result


def demo_fetch() -> Tuple[Image.Image, float]:
    """
    Demo function that returns a sample image and pixel scale.
    Used for testing when API key is not available.
    
    Returns:
        Tuple of (PIL Image, area_per_pixel in m²)
    """
    # Load a sample image from assets
    from pathlib import Path
    
    demo_path = Path(__file__).parent.parent / "assets" / "sample_roof.jpg"
    
    if not demo_path.exists():
        # Create a simple placeholder image if demo image doesn't exist
        image: Image.Image = Image.new('RGB', (640, 640), color='gray')
    else:
        image = Image.open(demo_path)  # type: ignore[assignment]
    
    # Use a reasonable default pixel scale for demo
    area_per_pixel: float = 0.25  # 0.5m x 0.5m = 0.25 m² per pixel
    
    return image, area_per_pixel  # type: ignore[no-any-return]
