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
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    gray = img
    # create wall image (filter out small objects from image)
    wall_img = wall_filter(gray, kernel, opening_iter, dilate_iter)
    # detect walls
    boxes, img = detectPreciseBoxes(wall_img, wall_img, approx_accuracy)
    
    # Scale pixel value to 3d pos
    scale = 100

    # Create top walls verts
    verts = []
    for box in boxes:
        verts.extend([scale_point_to_vector(box, scale, 0)])

    height, width = img.shape[0], img.shape[1]
    result_image = np.zeros((height, width, 3), np.uint8)
    walls_plan_for_xml_generator = []

    for room in verts:
        for i in range(len(room) - 1):
            line = (int(room[i][0] * width / 100), int(room[i][1] * height / 100)), \
                   (int(room[i + 1][0] * width / 100), int(room[i + 1][1] * height / 100))
            walls_plan_for_xml_generator.append(line)
            cv2.line(result_image, line[0], line[1], (255, 255, 0), 2)  # Increased line thickness for visibility

    # Convert to RGB for better visualization
    wall_img_rgb = cv2.cvtColor(wall_img, cv2.COLOR_GRAY2RGB)
    
    # Combine the original filtered image with detected walls
    combined_img = cv2.addWeighted(wall_img_rgb, 0.7, result_image, 0.3, 0)
    
    return combined_img, walls_plan_for_xml_generator

def process_page(pdf_document, page_number: int, kernel, opening_iter, dilate_iter, accuracy):
    """Process a single page and add wall annotations."""
    page = pdf_document[page_number]
    page_rect = page.rect
    pix = page.get_pixmap()
    
    # Get wall detection results
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
    
    # Check if parameters have changed
    current_param_hash = f"{kernel}{opening_iter}{dilate_iter}{accuracy}"
    params_changed = ('param_hash' not in st.session_state or 
                     st.session_state.param_hash != current_param_hash)
    
    # Initialize or update preview cache
    if 'preview_cache' not in st.session_state:
        st.session_state.preview_cache = {}
    
    if params_changed:
        # Only clear cache, don't regenerate yet
        st.session_state.preview_cache = {}
        st.session_state.param_hash = current_param_hash
    
    # Page controls
    cols_control = st.columns(6)
    for idx in range(len(pdf_document)):
        with cols_control[idx % 6]:
            col1, col2 = st.columns([3, 1])
            with col1:
                # Only show selection checkbox if page isn't removed
                if st.session_state.page_states[idx]['show_preview']:
                    selected = st.checkbox(f"Page {idx + 1}", 
                                        key=f"select_{idx}",
                                        value=st.session_state.page_states[idx]['status'] == 'selected')
                    st.session_state.page_states[idx]['status'] = 'selected' if selected else 'deselected'
            with col2:
                if st.button('✕', key=f"remove_{idx}", help="Remove from preview"):
                    st.session_state.page_states[idx]['show_preview'] = False
                    if str(idx) in st.session_state.preview_cache:  # Convert to string for key
                        del st.session_state.preview_cache[str(idx)]
                    st.rerun()
    
    # Show preview section
    st.write("### Wall Detection Preview")
    preview_cols = st.columns(3)
    
    # Get list of pages that should be shown
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
    
    # Generate previews only for visible pages that aren't in cache
    pages_to_process = [idx for idx in visible_pages 
                       if str(idx) not in st.session_state.preview_cache]  # Convert to string for comparison
    
    if pages_to_process:
        progress_bar = st.progress(0)
        for i, idx in enumerate(pages_to_process):
            page = pdf_document[idx]
            pix = page.get_pixmap()
            preview_img, wall_coords = detect_walls(pix, kernel, opening_iter, dilate_iter, accuracy)
            st.session_state.preview_cache[str(idx)] = preview_img  # Store with string key
            progress_bar.progress((i + 1) / len(pages_to_process))
        progress_bar.empty()
    
    # Display previews
    for i, idx in enumerate(visible_pages):
        col_idx = i % 3
        with preview_cols[col_idx]:
            preview_img = st.session_state.preview_cache[str(idx)]  # Use string key
            status = st.session_state.page_states[idx]['status']
            status_color = "🟢" if status == 'selected' else "🔵"
            st.image(preview_img, 
                    caption=f"{status_color} Page {idx + 1} ({status})", 
                    use_column_width=True)
        
        if col_idx == 2 and i < len(visible_pages) - 1:
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