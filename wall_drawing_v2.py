import streamlit as st
import fitz  # PyMuPDF
import cv2
import numpy as np
import io
from datetime import datetime
from PIL import Image
from dataclasses import dataclass 
from typing import Tuple, List, Optional 

@dataclass
class WallLine:
    start: Tuple[float, float]
    end: Tuple[float, float]
    is_horizontal: bool
    is_outer: bool
    length: float

class WallDetectorPDF:
    def __init__(self):
        self.stroke_color = (0, 0, 1)
        self.line_width = 2.0
        self.PRINT_FLAG = 4
        self.OUTER_WALL_COLOR = (255, 0, 0)    # Red for outer walls
        self.INNER_WALL_COLOR = (0, 255, 0)    # Green for inner walls

    def calculate_parameters(self, sensitivity: float) -> dict:
        """Parameters optimized specifically for architectural floor plans."""
        normalized = sensitivity / 100
        
        # Stricter base parameters
        base_params = {
            'canny_low': 75,         # Higher to reduce noise
            'canny_high': 200,       # Higher to get stronger edges
            'hough_threshold': 70,   # Higher to get fewer false positives
            'min_line_length': 80,   # Longer minimum length to avoid annotations
            'max_line_gap': 15,      # Slightly larger to connect wall segments
            'angle_tolerance': 1.5,  # Stricter angle for true walls
            'duplicate_tolerance': 8, # Tighter duplicate detection
            'min_wall_thickness': 5, # Minimum thickness for wall lines
            'outer_wall_threshold': 0.85
        }
        
        # More lenient max parameters
        max_params = {
            'canny_low': 50,
            'canny_high': 150,
            'hough_threshold': 50,
            'min_line_length': 60,
            'max_line_gap': 25,
            'angle_tolerance': 3.0,
            'duplicate_tolerance': 12,
            'min_wall_thickness': 3,
            'outer_wall_threshold': 0.75
        }
        
        params = {
            'canny_low': int(base_params['canny_low'] + (max_params['canny_low'] - base_params['canny_low']) * normalized),
            'canny_high': int(base_params['canny_high'] + (max_params['canny_high'] - base_params['canny_high']) * normalized),
            'hough_threshold': int(base_params['hough_threshold'] + (max_params['hough_threshold'] - base_params['hough_threshold']) * normalized),
            'min_line_length': int(base_params['min_line_length'] + (max_params['min_line_length'] - base_params['min_line_length']) * normalized),
            'max_line_gap': int(base_params['max_line_gap'] + (max_params['max_line_gap'] - base_params['max_line_gap']) * normalized),
            'angle_tolerance': base_params['angle_tolerance'] + (max_params['angle_tolerance'] - base_params['angle_tolerance']) * normalized,
            'duplicate_tolerance': base_params['duplicate_tolerance'] + (max_params['duplicate_tolerance'] - base_params['duplicate_tolerance']) * normalized,
            'min_wall_thickness': base_params['min_wall_thickness'] + (max_params['min_wall_thickness'] - base_params['min_wall_thickness']) * normalized,
            'outer_wall_threshold': base_params['outer_wall_threshold'] + (max_params['outer_wall_threshold'] - base_params['outer_wall_threshold']) * normalized
        }
        
        return params
    def classify_walls(self, walls: List[WallLine], img_shape) -> List[WallLine]:
        """Classify walls as outer or inner based on position and length."""
        height, width = img_shape[:2]
        max_horizontal_length = width * 0.8
        max_vertical_length = height * 0.8
        
        # Find the bounding box of all walls
        all_x = [coord for wall in walls for coord in (wall.start[0], wall.end[0])]
        all_y = [coord for wall in walls for coord in (wall.start[1], wall.end[1])]
        
        if not all_x or not all_y:
            return walls
            
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        bounding_width = max_x - min_x
        bounding_height = max_y - min_y
        
        # Classify each wall
        for wall in walls:
            if wall.is_horizontal:
                # Horizontal wall classification
                is_at_edge = (abs(wall.start[1] - min_y) < 20 or 
                            abs(wall.start[1] - max_y) < 20)
                is_long = wall.length > bounding_width * 0.7
                wall.is_outer = is_at_edge and is_long
            else:
                # Vertical wall classification
                is_at_edge = (abs(wall.start[0] - min_x) < 20 or 
                            abs(wall.start[0] - max_x) < 20)
                is_long = wall.length > bounding_height * 0.7
                wall.is_outer = is_at_edge and is_long
        
        return walls

    def check_wall_thickness(self, img, x1, y1, x2, y2, params):
        """Check if line represents a wall by verifying its thickness."""
        thickness = params['min_wall_thickness']
        mask = np.zeros_like(img)
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, int(thickness * 2))
        
        # Check the average intensity along the line
        intersection = cv2.bitwise_and(img, mask)
        non_zero = cv2.countNonZero(intersection)
        total = cv2.countNonZero(mask)
        
        if total == 0:
            return False
            
        ratio = non_zero / total
        return ratio > 0.3  # At least 30% of the line should be dark

    def detect_walls(self, page_pixmap, sensitivity: float = 50) -> List[WallLine]:
        """Detect walls with enhanced filtering for architectural drawings."""
        img = np.frombuffer(page_pixmap.samples, dtype=np.uint8).reshape(
            page_pixmap.height, page_pixmap.width, page_pixmap.n
        )
        
        if page_pixmap.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        elif page_pixmap.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        params = self.calculate_parameters(sensitivity)
        
        # Enhanced preprocessing
        # Use bilateral filter to preserve edges while removing noise
        blurred = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Additional contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        # Adaptive thresholding with smaller window
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # Inverted to detect dark walls
            11, 2
        )
        
        # Edge detection
        edges = cv2.Canny(
            thresh,
            params['canny_low'],
            params['canny_high'],
            apertureSize=3,
            L2gradient=True
        )
        
        # Separate dilation for vertical and horizontal components
        kernel_v = np.ones((3,1), np.uint8)
        kernel_h = np.ones((1,3), np.uint8)
        edges_v = cv2.dilate(edges, kernel_v, iterations=1)
        edges_h = cv2.dilate(edges, kernel_h, iterations=1)
        edges = cv2.bitwise_or(edges_v, edges_h)
        
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=params['hough_threshold'],
            minLineLength=params['min_line_length'],
            maxLineGap=params['max_line_gap']
        )
        
        walls = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
                
                # Check if line is horizontal or vertical
                is_horizontal = abs(angle) < params['angle_tolerance'] or abs(angle - 180) < params['angle_tolerance']
                is_vertical = abs(angle - 90) < params['angle_tolerance']
                
                if (is_horizontal or is_vertical) and self.check_wall_thickness(thresh, x1, y1, x2, y2, params):
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    wall = WallLine(
                        start=(float(x1), float(y1)),
                        end=(float(x2), float(y2)),
                        is_horizontal=is_horizontal,
                        is_outer=False,
                        length=length
                    )
                    walls.append(wall)
            
            walls = self.merge_walls(walls, params)
            walls = self.classify_walls(walls, img.shape)
        
        return walls

    def merge_walls(self, walls: List[WallLine], params) -> List[WallLine]:
        """Merge similar walls."""
        if not walls:
            return []
        
        tolerance = params['duplicate_tolerance']
        merged = []
        used = set()
        
        for i, wall1 in enumerate(walls):
            if i in used:
                continue
                
            current_group = [wall1]
            used.add(i)
            
            for j, wall2 in enumerate(walls[i+1:], i+1):
                if j in used:
                    continue
                    
                if wall1.is_horizontal == wall2.is_horizontal:
                    if wall1.is_horizontal:
                        if abs(wall1.start[1] - wall2.start[1]) <= tolerance:
                            # Check for x-overlap
                            x_min = min(wall1.start[0], wall1.end[0], wall2.start[0], wall2.end[0])
                            x_max = max(wall1.start[0], wall1.end[0], wall2.start[0], wall2.end[0])
                            if x_max - x_min <= (wall1.length + wall2.length + tolerance):
                                current_group.append(wall2)
                                used.add(j)
                    else:
                        if abs(wall1.start[0] - wall2.start[0]) <= tolerance:
                            # Check for y-overlap
                            y_min = min(wall1.start[1], wall1.end[1], wall2.start[1], wall2.end[1])
                            y_max = max(wall1.start[1], wall1.end[1], wall2.start[1], wall2.end[1])
                            if y_max - y_min <= (wall1.length + wall2.length + tolerance):
                                current_group.append(wall2)
                                used.add(j)
            
            # Merge the group
            if current_group:
                is_horizontal = current_group[0].is_horizontal
                if is_horizontal:
                    x_coords = [x for wall in current_group for x in (wall.start[0], wall.end[0])]
                    y_coords = [wall.start[1] for wall in current_group]
                    merged_wall = WallLine(
                        start=(min(x_coords), np.mean(y_coords)),
                        end=(max(x_coords), np.mean(y_coords)),
                        is_horizontal=True,
                        is_outer=False,
                        length=max(x_coords) - min(x_coords)
                    )
                else:
                    x_coords = [wall.start[0] for wall in current_group]
                    y_coords = [y for wall in current_group for y in (wall.start[1], wall.end[1])]
                    merged_wall = WallLine(
                        start=(np.mean(x_coords), min(y_coords)),
                        end=(np.mean(x_coords), max(y_coords)),
                        is_horizontal=False,
                        is_outer=False,
                        length=max(y_coords) - min(y_coords)
                    )
                merged.append(merged_wall)
        
        return merged

    def preview_detection(self, page_pixmap, sensitivity: float = 50):
        """Generate a preview image with detected walls."""
        img = np.frombuffer(page_pixmap.samples, dtype=np.uint8).reshape(
            page_pixmap.height, page_pixmap.width, page_pixmap.n
        )
        
        if page_pixmap.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        preview_img = img.copy()
        walls = self.detect_walls(page_pixmap, sensitivity)
        
        for wall in walls:
            color = self.OUTER_WALL_COLOR if wall.is_outer else self.INNER_WALL_COLOR
            start_point = (int(wall.start[0]), int(wall.start[1]))
            end_point = (int(wall.end[0]), int(wall.end[1]))
            cv2.line(preview_img, start_point, end_point, color, 2)
        
        return preview_img, len(walls)

    def add_wall_annotation(self, page, start_point, end_point, wall_type="Wall"):
        """Add a wall annotation to the PDF page."""
        annot = page.add_line_annot(start_point, end_point)
        annot.set_border(width=self.line_width)
        annot.set_colors(stroke=self.stroke_color)
        
        annot.info.update({
            "Title": "Wall Markup",
            "Subject": wall_type,
            "Creator": "Wall Detector",
            "ModDate": datetime.now().strftime("%Y%m%d%H%M%S"),
            "Type": "Wall"
        })
        
        annot.set_flags(self.PRINT_FLAG)
        annot.update()

    def process_page(self, pdf_document, page_number: int, sensitivity: float = 50):
        """Process a single page of the PDF and add wall annotations."""
        page = pdf_document[page_number]
        page_rect = page.rect
        pix = page.get_pixmap()
        
        detected_walls = self.detect_walls(pix, sensitivity)
        
        for wall in detected_walls:
            # Convert coordinates to PDF space
            pdf_x1 = wall.start[0] * page_rect.width / pix.width
            pdf_y1 = wall.start[1] * page_rect.height / pix.height
            pdf_x2 = wall.end[0] * page_rect.width / pix.width
            pdf_y2 = wall.end[1] * page_rect.height / pix.height
            
            # Determine wall type based on classification
            wall_type = "Outer Wall" if wall.is_outer else "Inner Wall"
            
            self.add_wall_annotation(
                page,
                (pdf_x1, pdf_y1),
                (pdf_x2, pdf_y2),
                wall_type
            )
        
        return len(detected_walls)

def main():
    st.title("PDF Wall Detector")
    
    # Initialize session state
    if 'processed_pages' not in st.session_state:
        st.session_state.processed_pages = set()
    if 'pdf_document' not in st.session_state:
        st.session_state.pdf_document = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'wall_count' not in st.session_state:
        st.session_state.wall_count = None
        
    uploaded_file = st.file_uploader("Choose a PDF floor plan", type=["pdf"])
    
    if uploaded_file is not None:
        # Check if a new file was uploaded
        if st.session_state.current_file != uploaded_file:
            st.session_state.current_file = uploaded_file
            st.session_state.processed_pages = set()
            # Open new PDF document
            pdf_bytes = uploaded_file.read()
            st.session_state.pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Initialize PDF processor
        pdf_processor = WallDetectorPDF()
        
        # Page selection
        page_count = len(st.session_state.pdf_document)
        page_options = [f"Page {i+1}" for i in range(page_count)]
        selected_page = st.selectbox("Select a page to analyze", page_options)
        selected_index = int(selected_page.split()[1]) - 1
        
        # Get page for preview
        page = st.session_state.pdf_document[selected_index]
        pix = page.get_pixmap()
        
        # Sensitivity slider
        sensitivity = st.slider("Wall Detection Sensitivity", 0, 100, 50)
        
        # Real-time preview
        preview_img, wall_count = pdf_processor.preview_detection(pix, sensitivity)
        st.image(preview_img, caption=f"Preview of detected walls - {wall_count} walls found", use_column_width=True)
        
        # Show processed pages
        if st.session_state.processed_pages:
            st.write("Processed pages:", ", ".join(f"Page {p+1}" for p in sorted(st.session_state.processed_pages)))
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Apply Wall Detection"):
                with st.spinner("Processing..."):
                    wall_count = pdf_processor.process_page(
                        st.session_state.pdf_document,
                        selected_index,
                        sensitivity
                    )
                    st.session_state.processed_pages.add(selected_index)
                    st.session_state.wall_count = wall_count
                    st.rerun()
        
        with col2:
            if len(st.session_state.processed_pages) > 0:
                if st.button("Download Complete PDF"):
                    with st.spinner("Preparing PDF..."):
                        output_bytes = io.BytesIO()
                        st.session_state.pdf_document.save(output_bytes)
                        
                        st.download_button(
                            label="Click to Download",
                            data=output_bytes.getvalue(),
                            file_name="processed_floor_plan.pdf",
                            mime="application/pdf"
                        )
            else:
                st.button("Download Complete PDF", disabled=True, help="Process at least one page first")

if __name__ == "__main__":
    main()