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
        
        detected_walls = detect_walls(pix, kernel, opening_iter, dilate_iter, accuracy)
        
        for wall in detected_walls:
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
        
        return len(detected_walls)

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

def main():
    st.title("Enhanced Wall Detector")
    st.write("Detects black walls and rooms in floor plans using advanced computer vision")

    # Initialize session state
    if 'processed_pages' not in st.session_state:
        st.session_state.processed_pages = set()
    if 'pdf_document' not in st.session_state:
        st.session_state.pdf_document = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'wall_count' not in st.session_state:
        st.session_state.wall_count = None

    ## Initialize detector
    #detector = EnhancedWallDetector()

    uploaded_file = st.file_uploader("Choose a PDF floor plan", type=["pdf"])

    if uploaded_file is not None:
        # Check if a new file was uploaded
        if st.session_state.current_file != uploaded_file:
            st.session_state.current_file = uploaded_file
            st.session_state.processed_pages = set()
            # Open new PDF document
            pdf_bytes = uploaded_file.read()
            st.session_state.pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Page selection
        page_count = len(st.session_state.pdf_document)
        page_options = [f"Page {i+1}" for i in range(page_count)]
        selected_page = st.selectbox("Select a page to analyze", page_options)
        selected_index = int(selected_page.split()[1]) - 1

        # Get page for preview
        page = st.session_state.pdf_document[selected_index]
        pix = page.get_pixmap()

        kernel = st.slider("Kernel (noise)", 1, 5, 3, 
                                 help="x")
    
        opening_iter= st.slider("Opening_iter", 1, 5, 3)

        dilate_iter= st.slider("Dilate_iter", 1, 5, 3)

        accuracy= st.selectbox('Accuracy', (0.001, 0.01, 0.1, 1))

        # Real-time preview with wall thickness parameter
        preview_img, wall_coords = detect_walls(pix, kernel, opening_iter, dilate_iter, accuracy)
        st.image(preview_img, caption=f"Preview of detected walls", 
                use_column_width=True)

        # Show processed pages
        if st.session_state.processed_pages:
            st.write("Processed pages:", ", ".join(f"Page {p+1}" for p in sorted(st.session_state.processed_pages)))

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Apply Wall Detection"):
                with st.spinner("Processing..."):
                    wall_count = process_page(  # Use hybrid detector instead of local_detector
                        st.session_state.pdf_document,
                        selected_index,
                        kernel, opening_iter, dilate_iter, accuracy
                    )
                    st.session_state.processed_pages.add(selected_index)
                    st.session_state.wall_count = wall_count
                    

        with col2:
            if len(st.session_state.processed_pages) > 0:
                if st.button("Download Annotated PDF"):
                    with st.spinner("Preparing PDF..."):
                        output_bytes = io.BytesIO()
                        st.session_state.pdf_document.save(output_bytes)
                        st.download_button(
                            label="Click to Download",
                            data=output_bytes.getvalue(),
                            file_name="annotated_floor_plan.pdf",
                            mime="application/pdf"
                        )
            else:
                st.button("Download Annotated PDF", disabled=True, help="Process at least one page first")

if __name__ == "__main__":
    main() 