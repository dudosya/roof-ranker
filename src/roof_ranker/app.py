"""Main Streamlit dashboard for Sol-Scout Pro with clickable image coordinates."""

import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import streamlit as st
from PIL import Image
import cv2
from streamlit_image_coordinates import streamlit_image_coordinates

from roof_ranker.utils.calculator import calculate_solar_potential, format_area, format_currency, format_power
from roof_ranker.utils.constants import CITIES
from roof_ranker.utils.geo_handler import fetch_satellite_image
from roof_ranker.utils.neural_vision import SAMRoofSegmenter, create_mask_overlay, count_mask_pixels
from roof_ranker.utils.vision import convert_to_rgb, resize_image

# Page configuration
st.set_page_config(
    page_title="Sol-Scout Pro: AI-Powered Solar Audit",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_sam() -> SAMRoofSegmenter:
    """Initialize SAM model with caching."""
    if "sam_model" not in st.session_state:
        with st.spinner("Loading AI model (first time may take a minute)..."):
            st.session_state.sam_model = SAMRoofSegmenter(model_type="vit_b")
    return st.session_state.sam_model  # type: ignore[no-any-return]


def draw_polygon_on_image(
    image_np: np.ndarray,
    points: List[Tuple[int, int]],
    closed: bool = False
) -> np.ndarray:
    """
    Draw polygon on image.
    
    Args:
        image_np: Image as numpy array (H, W, 3)
        points: List of (x, y) coordinates
        closed: Whether to close the polygon
    
    Returns:
        Image with polygon drawn
    """
    if not points:
        return image_np
    
    # Create a copy to avoid modifying original
    marked_image: np.ndarray = image_np.copy()
    
    try:
        # Draw points
        for i, (x, y) in enumerate(points):
            color = (0, 0, 255) if i == 0 else (255, 0, 0)  # Red for first point, blue for others
            cv2.circle(marked_image, (x, y), 8, color, -1)  # Filled circle
            cv2.circle(marked_image, (x, y), 8, (255, 255, 255), 2)  # White border
            # Add point number
            cv2.putText(marked_image, str(i+1), (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw lines between points
        if len(points) > 1:
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                cv2.line(marked_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines
            
            # Close polygon if requested
            if closed and len(points) > 2:
                x1, y1 = points[-1]
                x2, y2 = points[0]
                cv2.line(marked_image, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.line(marked_image, (x1, y1), (x2, y2), (0, 0, 0), 1)
        
        # Draw bounding box if we have enough points
        if len(points) >= 3 and closed:
            # Create bounding box from points
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Draw bounding box
            cv2.rectangle(marked_image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            
    except Exception as e:
        st.error(f"Error drawing polygon: {e}")
    
    return marked_image


def create_bounding_box_from_points(points: List[Tuple[int, int]]) -> Optional[Tuple[int, int, int, int]]:
    """
    Create bounding box from polygon points.
    
    Args:
        points: List of (x, y) coordinates
    
    Returns:
        (x_min, y_min, x_max, y_max) or None
    """
    if not points or len(points) < 3:
        return None
    
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    return (x_min, y_min, x_max, y_max)


def main() -> None:
    """Main application function."""
    # Header
    st.markdown('<div class="main-header">Sol-Scout Pro: AI-Powered Solar Audit</div>', unsafe_allow_html=True)
    st.markdown("Advanced solar potential analysis with AI roof segmentation")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        # City selection
        city_options = {data["name"]: key for key, data in CITIES.items()}
        selected_city_name = st.selectbox(
            "Select City",
            options=list(city_options.keys()),
            index=0,
            help="Choose the city for solar irradiance and energy cost data",
        )
        selected_city_key = city_options[selected_city_name]

        # Google Maps API Key
        st.subheader("Google Maps API")
        api_key = st.text_input(
            "Google Maps Static API Key",
            type="password",
            help="Required to fetch satellite images",
        )

        # Address input
        st.subheader("Building Address")
        
        address = st.text_input(
            "Enter Building Address",
            value="Baku State University, Baku, Azerbaijan",
            help="Use format: 'Street, City, Country'",
        )

        # Fetch button
        fetch_button = st.button("Fetch Satellite Image", width='stretch')

        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("""
        1. Enter Google Maps API key
        2. Enter building address
        3. Click "Fetch Satellite Image"
        4. Click on image to add polygon points (minimum 3)
        5. Click "Complete Polygon" when done
        6. Click "Process Polygon" to create bounding box
        7. Click "Confirm Segmentation"
        8. View AI analysis results
        """)

    # Main content area
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Satellite Image")

        # Initialize session state
        if "satellite_image" not in st.session_state:
            st.session_state.satellite_image = None
            st.session_state.area_per_pixel = None
            st.session_state.mask = None
            st.session_state.box = None
            st.session_state.polygon_points = []
            st.session_state.polygon_closed = False

        # Fetch image when button is clicked
        if fetch_button:
            if not api_key:
                st.error("Please enter a Google Maps API key")
            elif not address:
                st.error("Please enter a building address")
            else:
                with st.spinner("Fetching satellite image..."):
                    try:
                        image, area_per_pixel = fetch_satellite_image(address, api_key)
                        # Resize for display
                        image_np = np.array(image)
                        image_np = convert_to_rgb(image_np)
                        resized_image = resize_image(image_np, max_width=600)
                        
                        # Store in session state
                        st.session_state.satellite_image = resized_image
                        st.session_state.area_per_pixel = area_per_pixel
                        st.session_state.mask = None
                        st.session_state.box = None
                        st.session_state.polygon_points = []
                        st.session_state.polygon_closed = False
                        
                        st.success(f"Image fetched! Pixel scale: {area_per_pixel:.4f} m² per pixel")
                    except Exception as e:
                        st.error(f"Error fetching image: {e}")
                        st.info("Make sure your API key is valid and has Static Maps API enabled")

        # Display image or placeholder
        if st.session_state.satellite_image is not None:
            image_display = st.session_state.satellite_image
            height, width = image_display.shape[:2]
            
            # Click-to-Draw Polygon Tool
            st.subheader("Polygon Tool")
            st.info("Click on image to add points. Minimum 3 points required.")
            
            # Prepare image with polygon drawing
            if st.session_state.polygon_points:
                display_image = draw_polygon_on_image(
                    image_display,
                    st.session_state.polygon_points,
                    closed=st.session_state.polygon_closed
                )
            else:
                display_image = image_display
            
            # Display clickable image with a unique key based on points count
            points_key = f"image_coords_{len(st.session_state.polygon_points)}"
            click_data = streamlit_image_coordinates(
                Image.fromarray(display_image),
                key=points_key,
                width=width,
                cursor="crosshair"
            )
            
            # Handle click event
            if click_data:
                x = click_data["x"]
                y = click_data["y"]
                
                # Add point to polygon
                st.session_state.polygon_points.append((x, y))
                st.success(f"Point {len(st.session_state.polygon_points)} added at ({x}, {y})")
                # Use rerun to refresh the UI
                st.rerun()
            
            # Polygon status
            num_points = len(st.session_state.polygon_points)
            if num_points > 0:
                st.write(f"Points: {num_points} (minimum 3)")
                if num_points >= 3:
                    st.success("Minimum points reached. Click 'Complete Polygon' when done.")
                else:
                    st.warning(f"Need {3 - num_points} more points.")
            
            # Action buttons
            col_complete, col_clear, col_process = st.columns(3)
            with col_complete:
                complete_disabled = num_points < 3
                complete_button = st.button(
                    "Complete Polygon", 
                    disabled=complete_disabled, 
                    width='stretch',
                    help="Complete polygon (requires at least 3 points)"
                )
            
            with col_clear:
                clear_button = st.button("Clear Points", width='stretch', help="Remove all polygon points")
            
            with col_process:
                process_disabled = num_points < 3 or not st.session_state.polygon_closed
                process_button = st.button(
                    "Process Polygon", 
                    disabled=process_disabled, 
                    width='stretch',
                    help="Process polygon into bounding box"
                )
            
            if complete_button and num_points >= 3:
                # Close the polygon
                st.session_state.polygon_closed = True
                st.success(f"Polygon completed with {num_points} points!")
                st.rerun()
            
            if clear_button:
                # Clear points
                st.session_state.polygon_points = []
                st.session_state.polygon_closed = False
                st.session_state.box = None
                st.rerun()
            
            if process_button and num_points >= 3 and st.session_state.polygon_closed:
                # Create bounding box from polygon
                box = create_bounding_box_from_points(st.session_state.polygon_points)
                if box:
                    st.session_state.box = box
                    st.success(f"Polygon processed! Bounding box: {box}")
                else:
                    st.error("Failed to create bounding box from polygon")
                st.rerun()
            
            # Show current box status
            if st.session_state.box:
                x1, y1, x2, y2 = st.session_state.box
                box_width = x2 - x1
                box_height = y2 - y1
                st.success(f"Bounding Box Selected: ({x1}, {y1}) to ({x2}, {y2}) - Size: {box_width}×{box_height} pixels")
                
                # Segmentation button
                confirm_button = st.button("Confirm Segmentation", type="primary", width='stretch')
                
                if confirm_button:
                    # Segment roof with SAM
                    with st.spinner("AI segmenting roof..."):
                        try:
                            sam_model = initialize_sam()
                            
                            # Validate box
                            x1, y1, x2, y2 = st.session_state.box
                            
                            # Check box validity
                            errors = []
                            if not (0 <= x1 < width and 0 <= x2 < width and 0 <= y1 < height and 0 <= y2 < height):
                                errors.append(f"Box coordinates out of bounds: {st.session_state.box} vs image {width}x{height}")
                            
                            if x2 <= x1:
                                errors.append(f"Invalid box: x2={x2} <= x1={x1}")
                            
                            if y2 <= y1:
                                errors.append(f"Invalid box: y2={y2} <= y1={y1}")
                            
                            # Check minimum size
                            box_width = x2 - x1
                            box_height = y2 - y1
                            if box_width < 10 or box_height < 10:
                                errors.append(f"Box too small: {box_width}x{box_height} pixels (minimum 10x10)")
                            
                            if errors:
                                for error in errors:
                                    st.error(error)
                                mask = None
                            else:
                                mask = sam_model.segment_roof_with_box(image_display, st.session_state.box)
                                st.success(f"Roof segmented with box {st.session_state.box}")
                                
                            if mask is not None:
                                st.session_state.mask = mask
                                
                        except Exception as e:
                            st.error(f"Error segmenting roof: {e}")
                            import traceback
                            st.code(traceback.format_exc(), language="python")
                            st.rerun()
            else:
                if st.session_state.polygon_closed:
                    st.info("Polygon closed. Click 'Process Polygon' to create bounding box.")
                elif num_points >= 3:
                    st.info("Polygon ready. Click 'Complete Polygon' to close shape.")
                elif num_points > 0:
                    st.info(f"Add more points to complete polygon ({num_points}/3).")
                else:
                    st.info("Click on the image to add polygon points around the roof.")
                
        else:
            # Display placeholder
            st.info("Enter API key and address, then click 'Fetch Satellite Image'")
            placeholder = np.zeros((400, 600, 3), dtype=np.uint8)
            placeholder[:, :] = [240, 240, 240]  # Light gray
            st.image(placeholder, caption="Satellite image will appear here", width='stretch')

    with col2:
        st.subheader("AI Analysis")

        if st.session_state.satellite_image is not None and st.session_state.mask is not None:
            # Create mask overlay
            image_display = st.session_state.satellite_image
            mask = st.session_state.mask
            
            # Create overlay (blue mask)
            overlay = create_mask_overlay(image_display, mask, color=(0, 0, 255))
            
            # Display overlay
            st.image(
                overlay,
                caption=f"AI-Segmented Roof - {count_mask_pixels(mask):,} pixels",
                width='stretch',
            )
            
            # Calculate solar potential
            if st.session_state.area_per_pixel is not None:
                roof_pixels = count_mask_pixels(mask)
                area_per_pixel = st.session_state.area_per_pixel
                
                analysis = calculate_solar_potential(
                    roof_pixels=roof_pixels,
                    city_key=selected_city_key,
                    pixel_to_meter=area_per_pixel,
                )
                
                # Display metrics
                st.subheader("Solar Potential Analysis")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Roof Area", format_area(analysis["area_m2"]))
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Power Capacity", format_power(analysis["capacity_kw"]))
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with metric_col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Annual Savings", format_currency(analysis["annual_savings_usd"]))
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Additional info
                with st.expander("Detailed Analysis"):
                    st.write(f"**City:** {analysis['city_name']}")
                    st.write(f"**Address:** {address}")
                    st.write(f"**CO₂ Reduction:** {analysis['co2_saved_tons']:.1f} tons/year")
                    st.write(f"**Roof Pixels:** {roof_pixels:,}")
                    st.write(f"**Pixel Area:** {area_per_pixel:.4f} m² per pixel")
                    
                    # Show selection info
                    if st.session_state.box:
                        st.write(f"**Bounding Box:** {st.session_state.box}")
                        st.write(f"**Polygon Points:** {len(st.session_state.polygon_points)} points")
        
        elif st.session_state.satellite_image is not None:
            st.info("Draw a polygon around the roof and click 'Confirm Segmentation'")
            placeholder = np.zeros((400, 600, 3), dtype=np.uint8)
            placeholder[:, :] = [240, 240, 240]
            st.image(placeholder, caption="AI analysis will appear here", width='stretch')
        
        else:
            st.info("Fetch a satellite image first to see AI analysis here")
            placeholder = np.zeros((400, 600, 3), dtype=np.uint8)
            placeholder[:, :] = [240, 240, 240]
            st.image(placeholder, caption="AI analysis will appear here", width='stretch')

    # Export button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Generate AI Analysis Report", width='stretch'):
            if st.session_state.mask is not None:
                st.toast("AI report generated successfully!")
                st.success("PDF report with AI segmentation results would be downloaded here.")
            else:
                st.warning("Please segment a roof first to generate a report")


if __name__ == "__main__":
    main()
