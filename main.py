import streamlit as st
import fitz  # PyMuPDF
import cv2
import numpy as np
import io
from datetime import datetime
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any, Set
import base64
import json 
from collections import defaultdict

from utils.detect import *
from utils.transform import *
import argparse
import imutils
import cv2
import numpy as np

stroke_color = (0, 0, 1)
line_width = 2.0
PRINT_FLAG = 4

@dataclass
class Wall:
    start: tuple
    end: tuple
    wall_type: str = "standard"

def detect_walls(pix, kernel=3, opening_iter=3, dilate_iter=3, approx_accuracy=0.001):
    # Convert input image
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    gray = img
    wall_img = wall_filter(gray, kernel, opening_iter, dilate_iter)
    boxes, img = detectPreciseBoxes(wall_img, wall_img, approx_accuracy)
    
    height, width = img.shape[0], img.shape[1]
    walls_plan = []

    # Create three separate views of identical size
    original_view = cv2.cvtColor(wall_img, cv2.COLOR_GRAY2BGR)
    analysis_view = np.zeros((height, width, 3), dtype=np.uint8)
    clean_wall_view = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Convert vertices to wall segments
    verts = []
    for box in boxes:
        verts.extend([scale_point_to_vector(box, 100, 0)])
        
    for room in verts:
        for i in range(len(room) - 1):
            line = (int(room[i][0] * width / 100), int(room[i][1] * height / 100)), \
                   (int(room[i + 1][0] * width / 100), int(room[i + 1][1] * height / 100))
            walls_plan.append(line)
            # Draw original endpoints with thicker points
            cv2.circle(analysis_view, line[0], 5, (255, 0, 0), -1)  # Blue dot
            cv2.circle(analysis_view, line[1], 5, (0, 255, 0), -1)  # Green dot
            # Draw original wall segment in white
            cv2.line(analysis_view, line[0], line[1], (255, 255, 255), 2)
    
    # Get merged walls
    merged_walls = merge_nearby_walls(walls_plan, 
                                    distance_threshold=3, 
                                    angle_threshold=5)
    
    # Draw merged walls with thicker lines
    for wall in merged_walls:
        p1 = (int(wall[0][0]), int(wall[0][1]))
        p2 = (int(wall[1][0]), int(wall[1][1]))
        cv2.line(analysis_view, p1, p2, (0, 0, 255), 3)  # Red lines
        cv2.circle(analysis_view, p1, 6, (255, 0, 255), -1)  # Magenta dot
        cv2.circle(analysis_view, p2, 6, (255, 255, 0), -1)  # Yellow dot
        # Draw on clean view (just the walls)
        cv2.line(clean_wall_view, p1, p2, (0, 0, 0), 2)  # Black lines
    
    # Create padding (black bars between images)
    padding = np.zeros((height, 20, 3), dtype=np.uint8)
    
    # Calculate the target width for each image (1/3 of the final width, accounting for padding)
    target_width = width
    target_height = height
    
    # Resize all images to the same dimensions
    original_view = cv2.resize(original_view, (target_width, target_height))
    analysis_view = cv2.resize(analysis_view, (target_width, target_height))
    clean_wall_view = cv2.resize(clean_wall_view, (target_width, target_height))
    
    # Stack all views horizontally with padding
    combined_view = np.hstack((original_view, padding, analysis_view, padding, clean_wall_view))
    
    return combined_view, merged_walls, len(walls_plan)

def process_page(pdf_document, page_number: int, kernel, opening_iter, dilate_iter, accuracy):
    """Process a single page and add wall annotations."""
    page = pdf_document[page_number]
    page_rect = page.rect
    pix = page.get_pixmap()
    
    # Get wall detection results with merged walls
    _, wall_coords = detect_walls(pix, kernel, opening_iter, dilate_iter, accuracy)
    
    # Convert coordinates to Wall objects
    walls = []
    for line in wall_coords:
        wall = Wall(
            start=line[0],
            end=line[1],
            wall_type="standard"
        )
        walls.append(wall)
    
    # Process each wall
    for wall in walls:
        pdf_x1 = wall.start[0] * page_rect.width / pix.width
        pdf_y1 = wall.start[1] * page_rect.height / pix.height
        pdf_x2 = wall.end[0] * page_rect.width / pix.width
        pdf_y2 = wall.end[1] * page_rect.height / pix.height
        
        add_wall_annotation(
            page,
            (pdf_x1, pdf_y1),
            (pdf_x2, pdf_y2),
            wall.wall_type
        )
    
    return len(walls)

def add_wall_annotation(page, start_point, end_point, wall_type):
        """Add a wall annotation to the PDF page."""
        annot = page.add_line_annot(start_point, end_point)
        annot.set_border(width=line_width)
        annot.set_colors(stroke=stroke_color)
        
        annot.info.update({
            "Title": "Wall Markup",
            "Subject": wall_type,
            "Creator": "Wall Detector",
            "ModDate": datetime.now().strftime("%Y%m%d%H%M%S"),
            "Type": "Wall"
        })
        
        annot.set_flags(PRINT_FLAG)
        annot.update()
def create_clean_wall_view(walls, height, width):
    """
    Create a clean visualization with just the merged walls, no dots
    """
    # Create a white background
    clean_view = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw merged walls with black lines
    for wall in walls:
        p1 = (int(wall[0][0]), int(wall[0][1]))
        p2 = (int(wall[1][0]), int(wall[1][1]))
        cv2.line(clean_view, p1, p2, (0, 0, 0), 2)  # Black lines with thickness 2
    
    return clean_view

def merge_nearby_walls(wall_coords, distance_threshold=5, angle_threshold=10):
    """
    Merge walls that are likely duplicates based on proximity, angle, and overlap
    """
    def point_distance(p1, p2):
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def line_length(line):
        return point_distance(line[0], line[1])
    
    def get_line_angle(line):
        dx = line[1][0] - line[0][0]
        dy = line[1][1] - line[0][1]
        angle = np.degrees(np.arctan2(dy, dx)) % 180
        return angle
    
    def are_parallel(angle1, angle2, threshold):
        diff = abs(angle1 - angle2) % 180
        return min(diff, 180 - diff) < threshold
    
    def segments_overlap(line1, line2):
        """Check if two line segments overlap or are very close to overlapping"""
        # Convert lines to directional vectors
        v1 = np.array([line1[1][0] - line1[0][0], line1[1][1] - line1[0][1]])
        v2 = np.array([line2[1][0] - line2[0][0], line2[1][1] - line2[0][1]])
        
        # Project endpoints of line2 onto line1
        p1 = np.array(line1[0])
        proj_start = np.dot(np.array(line2[0]) - p1, v1) / np.dot(v1, v1)
        proj_end = np.dot(np.array(line2[1]) - p1, v1) / np.dot(v1, v1)
        
        # Check if projections overlap with line1 segment
        overlap = (min(proj_start, proj_end) <= 1 and max(proj_start, proj_end) >= 0) or \
                 (min(proj_start, proj_end) >= 0 and max(proj_start, proj_end) <= 1)
        
        if overlap:
            # Calculate perpendicular distance
            n1 = np.array([-v1[1], v1[0]]) / np.linalg.norm(v1)
            d1 = abs(np.dot(np.array(line2[0]) - p1, n1))
            d2 = abs(np.dot(np.array(line2[1]) - p1, n1))
            return max(d1, d2) < distance_threshold
        return False
    
    def merge_overlapping_lines(lines):
        if not lines:
            return None
        
        # Get all endpoints
        points = []
        for line in lines:
            points.extend([np.array(line[0]), np.array(line[1])])
        
        # Find principal direction using PCA
        points_array = np.array(points)
        mean = np.mean(points_array, axis=0)
        centered = points_array - mean
        u, s, vh = np.linalg.svd(centered)
        direction = vh[0]
        
        # Project all points onto principal direction
        projections = np.dot(centered, direction)
        
        # Get extreme points in projection direction
        min_idx = np.argmin(projections)
        max_idx = np.argmax(projections)
        
        # Return new line from extreme points
        start = tuple(map(int, points_array[min_idx]))
        end = tuple(map(int, points_array[max_idx]))
        return (start, end)
    
    merged_walls = []
    used_walls = set()
    
    # Filter out very short walls (likely noise)
    min_length = distance_threshold * 2
    wall_coords = [wall for wall in wall_coords if line_length(wall) > min_length]
    
    # Sort walls by length, process longer walls first
    wall_coords = sorted(wall_coords, key=line_length, reverse=True)
    
    for i, wall1 in enumerate(wall_coords):
        if i in used_walls:
            continue
            
        current_group = [wall1]
        used_walls.add(i)
        angle1 = get_line_angle(wall1)
        
        for j, wall2 in enumerate(wall_coords):
            if j in used_walls:
                continue
                
            angle2 = get_line_angle(wall2)
            if are_parallel(angle1, angle2, angle_threshold):
                if segments_overlap(wall1, wall2):
                    current_group.append(wall2)
                    used_walls.add(j)
        
        if current_group:
            merged_wall = merge_overlapping_lines(current_group)
            if merged_wall:
                merged_walls.append(merged_wall)
    
    return merged_walls

def create_page_preview(pdf_document):
    """Create preview thumbnails for all pages in the PDF with checkboxes"""
    # Initialize selected pages in session state if not exists
    if 'selected_pages' not in st.session_state:
        st.session_state.selected_pages = [True] * len(pdf_document)
    
    cols = st.columns(4)  # Show 4 thumbnails per row
    for idx, page in enumerate(pdf_document):
        pix = page.get_pixmap(matrix=fitz.Matrix(0.2, 0.2))  # Scale down for thumbnail
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        with cols[idx % 4]:
            st.checkbox(f"Page {idx + 1}", key=f"page_{idx}", value=st.session_state.selected_pages[idx])
            st.image(img, use_column_width=True)
    
    # Update selected pages based on checkboxes
    st.session_state.selected_pages = [st.session_state[f"page_{i}"] for i in range(len(pdf_document))]

def create_preview_carousel(pdf_document, kernel, opening_iter, dilate_iter, accuracy):
    """Create a horizontal scrollable preview with three states for each page"""
    st.write("### Page Controls")
    
    # Initialize page states if needed
    if 'page_states' not in st.session_state or len(st.session_state.page_states) != len(pdf_document):
        st.session_state.page_states = {
            i: {'status': 'selected', 'show_preview': True} 
            for i in range(len(pdf_document))
        }
    
    # Page controls
    cols_control = st.columns(6)
    for idx in range(len(pdf_document)):
        with cols_control[idx % 6]:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.session_state.page_states[idx]['show_preview']:
                    selected = st.checkbox(f"Page {idx + 1}", 
                                        key=f"select_{idx}",
                                        value=st.session_state.page_states[idx]['status'] == 'selected')
                    st.session_state.page_states[idx]['status'] = 'selected' if selected else 'deselected'
            with col2:
                if st.button('✕', key=f"remove_{idx}", help="Remove from preview"):
                    st.session_state.page_states[idx]['show_preview'] = False
                    if str(idx) in st.session_state.preview_cache:
                        del st.session_state.preview_cache[str(idx)]
                    st.rerun()
    
    # Show preview section
    st.write("### Wall Detection Preview")
    
    # Generate previews for visible pages
    visible_pages = [idx for idx in range(len(pdf_document)) 
                    if st.session_state.page_states[idx]['show_preview']]
    
    if not visible_pages:
        st.warning("No pages to preview. All pages have been removed.")
        if st.button("Reset all pages"):
            st.session_state.page_states = {
                i: {'status': 'selected', 'show_preview': True} 
                for i in range(len(pdf_document))
            }
            st.session_state.preview_cache = {}
            st.rerun()
        return

    # Process pages and display full-width previews
    for idx in visible_pages:
        page = pdf_document[idx]
        # Increase resolution of the pixmap
        pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))  # Higher resolution for better quality
        
        # Get visualization and wall counts
        combined_view, merged_walls, original_count = detect_walls(
            pix, kernel, opening_iter, dilate_iter, accuracy
        )
        
        # Calculate scaling to maintain aspect ratio while fitting screen width
        max_display_width = 1200  # Maximum width for display
        scale_factor = max_display_width / combined_view.shape[1]
        display_height = int(combined_view.shape[0] * scale_factor)
        
        # Resize the combined view while maintaining aspect ratio
        combined_view_resized = cv2.resize(combined_view, 
                                         (max_display_width, display_height), 
                                         interpolation=cv2.INTER_AREA)
        
        status = st.session_state.page_states[idx]['status']
        status_color = "🟢" if status == 'selected' else "🔵"
        
        # Create a container for the preview
        with st.container():
            st.write(f"#### Page {idx + 1} Wall Analysis")
            st.write(f"Found {len(merged_walls)} merged walls from {original_count} detected segments")
            
            # Convert BGR to RGB for Streamlit display
            combined_view_rgb = cv2.cvtColor(combined_view_resized, cv2.COLOR_BGR2RGB)
            
            # Display the image with fixed width and proper aspect ratio
            st.image(combined_view_rgb, 
                    caption=f"{status_color} Original | Analysis | Clean Walls", 
                    use_column_width=True)
            
            st.write("---")

def process_selected_pages(pdf_document, kernel, opening_iter, dilate_iter, accuracy):
    """Process only the pages marked as 'selected'"""
    total_walls = 0
    selected_pages = [idx for idx, state in st.session_state.page_states.items() 
                     if state['status'] == 'selected']
    
    for page_idx in selected_pages:
        wall_count = process_page(
            pdf_document,
            page_idx,
            kernel, opening_iter, dilate_iter, accuracy
        )
        st.session_state.processed_pages.add(page_idx)
        total_walls += wall_count
    
    return total_walls, len(selected_pages)

def main():
    st.title("Enhanced Wall Detector")
    st.write("Detects black walls and rooms in floor plans using advanced computer vision")

    # Initialize session state variables
    if 'processed_pages' not in st.session_state:
        st.session_state.processed_pages = set()
    if 'pdf_document' not in st.session_state:
        st.session_state.pdf_document = None
    if 'current_file_name' not in st.session_state:
        st.session_state.current_file_name = None

    uploaded_file = st.file_uploader("Choose a PDF floor plan", type=["pdf"])

    if uploaded_file is not None:
        current_file_name = uploaded_file.name

        # Check if a new file was uploaded by comparing names
        if st.session_state.current_file_name != current_file_name:
            # Reset states for new file
            st.session_state.current_file_name = current_file_name
            st.session_state.processed_pages = set()
            if 'preview_cache' in st.session_state:
                del st.session_state.preview_cache
            if 'page_states' in st.session_state:
                del st.session_state.page_states
            if 'param_hash' in st.session_state:
                del st.session_state.param_hash
            
            # Read new PDF document
            pdf_bytes = uploaded_file.read()
            st.session_state.pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Parameter controls
        st.sidebar.write("### Detection Parameters")
        kernel = st.sidebar.slider("Kernel (noise)", 1, 5, 3, 
                                 help="Controls noise reduction")
        opening_iter = st.sidebar.slider("Opening iterations", 1, 5, 3,
                                       help="Number of morphological opening iterations")
        dilate_iter = st.sidebar.slider("Dilation iterations", 1, 5, 3,
                                      help="Number of dilation iterations")
        accuracy = st.sidebar.selectbox('Accuracy', 
                                      (0.001, 0.01, 0.1, 1),
                                      help="Controls detection precision")

        # Create preview carousel with wall detection
        create_preview_carousel(st.session_state.pdf_document, 
                              kernel, opening_iter, dilate_iter, accuracy)

        # Control buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Apply Wall Detection", type="primary"):
                with st.spinner("Processing selected pages..."):
                    total_walls, pages_processed = process_selected_pages(
                        st.session_state.pdf_document,
                        kernel, opening_iter, dilate_iter, accuracy
                    )
                    st.success(f"Successfully detected {total_walls} walls across {pages_processed} pages")

        with col2:
            if len(st.session_state.processed_pages) > 0:
                if st.button("Download Annotated PDF"):
                    with st.spinner("Preparing PDF..."):
                        output_bytes = io.BytesIO()
                        st.session_state.pdf_document.save(output_bytes)
                        st.download_button(
                            label="Click to Download",
                            data=output_bytes.getvalue(),
                            file_name=f"annotated_{current_file_name}",
                            mime="application/pdf"
                        )
            else:
                st.button("Download Annotated PDF", 
                         disabled=True, 
                         help="Process at least one page first")
                
if __name__ == "__main__":
    main() 