import streamlit as st
import fitz  # PyMuPDF
import cv2
import numpy as np
import io
import anthropic
from datetime import datetime
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any, Set
import base64
import json 
from collections import defaultdict
from pydantic import BaseModel, Field
import re

class Coordinate(BaseModel):
    x: float
    y: float

class WallDetection(BaseModel):
    type: str
    coordinates: dict

class FloorPlanAnalysis(BaseModel):
    walls: List[WallDetection]

@dataclass(frozen=True)  # Make the class immutable
class WallLine:
    start: Tuple[float, float]
    end: Tuple[float, float]
    is_horizontal: bool
    is_outer: bool
    length: float
    wall_type: str = "INTERIOR_WALL"

    def __hash__(self):
        """Make WallLine hashable for use as dictionary key."""
        return hash((self.start, self.end, self.is_horizontal))

    def __eq__(self, other):
        """Define equality for WallLine objects."""
        if not isinstance(other, WallLine):
            return False
        return (self.start == other.start and 
                self.end == other.end and 
                self.is_horizontal == other.is_horizontal)

class EnhancedWallDetector:
    def __init__(self):
        self.stroke_color = (0, 0, 1)
        self.line_width = 2.0
        self.PRINT_FLAG = 4
        self.OUTER_WALL_COLOR = (255, 0, 0)    # Red for outer walls
        self.INNER_WALL_COLOR = (0, 255, 0)    # Green for inner walls
        self.PARTITION_WALL_COLOR = (0, 0, 255) # Blue for partition walls
        self.junction_points = []
        self.room_boundaries = []

    def calculate_parameters(self, sensitivity: float) -> dict:
        """Parameters optimized for black line detection in architectural floor plans."""
        normalized = sensitivity / 100
        
        base_params = {
            'threshold_value': 200,
            'canny_low': 50,
            'canny_high': 150,
            'hough_threshold': 70,
            'min_line_length': 80,
            'max_line_gap': 15,
            'angle_tolerance': 1.5,
            'duplicate_tolerance': 8,
            'min_wall_thickness': 5,
            'outer_wall_threshold': 0.85,
            'junction_tolerance': 10,
            'min_room_area': 1000,
            'wall_darkness_threshold': 100
        }
        
        max_params = {
            'threshold_value': 180,
            'canny_low': 30,
            'canny_high': 120,
            'hough_threshold': 50,
            'min_line_length': 60,
            'max_line_gap': 25,
            'angle_tolerance': 3.0,
            'duplicate_tolerance': 12,
            'min_wall_thickness': 3,
            'outer_wall_threshold': 0.75,
            'junction_tolerance': 15,
            'min_room_area': 500,
            'wall_darkness_threshold': 150
        }
        
        params = {}
        for key in base_params:
            params[key] = base_params[key] + (max_params[key] - base_params[key]) * normalized
            if key in ['canny_low', 'canny_high', 'hough_threshold', 'min_line_length', 
                      'max_line_gap', 'threshold_value', 'wall_darkness_threshold']:
                params[key] = int(params[key])
        
        return params
    
    def check_wall_thickness(self, img: np.ndarray, x1: float, y1: float, x2: float, y2: float, 
                           params: dict, wall_thickness: int) -> bool:
        """Check if line represents a wall by verifying its thickness and darkness."""
        mask = np.zeros_like(img)
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, int(wall_thickness * 2))
        
        intersection = cv2.bitwise_and(img, mask)
        line_pixels = img[mask > 0]
        
        if len(line_pixels) == 0:
            return False
        
        avg_darkness = np.mean(line_pixels)
        is_dark_enough = avg_darkness < params['wall_darkness_threshold']
        
        non_zero = cv2.countNonZero(intersection)
        total = cv2.countNonZero(mask)
        
        if total == 0:
            return False
        
        thickness_ratio = non_zero / total
        return thickness_ratio > 0.3 and is_dark_enough

    def detect_walls(self, page_pixmap, sensitivity: float = 50, wall_thickness: int = 5) -> List[WallLine]:
        """Enhanced wall detection focusing on black lines with configurable thickness."""
        img = np.frombuffer(page_pixmap.samples, dtype=np.uint8).reshape(
            page_pixmap.height, page_pixmap.width, page_pixmap.n
        )
        
        if page_pixmap.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        elif page_pixmap.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        params = self.calculate_parameters(sensitivity)
        
        # Threshold to focus on dark lines
        _, thresh = cv2.threshold(
            img,
            params['threshold_value'],
            255,
            cv2.THRESH_BINARY_INV
        )
        
        # Edge detection focused on strong edges (black lines)
        edges = cv2.Canny(
            thresh,
            params['canny_low'],
            params['canny_high'],
            apertureSize=3,
            L2gradient=True
        )
        
        # Morphological operations to enhance line detection
        kernel_v = np.ones((wall_thickness, 1), np.uint8)
        kernel_h = np.ones((1, wall_thickness), np.uint8)
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
                
                is_horizontal = abs(angle) < params['angle_tolerance'] or abs(angle - 180) < params['angle_tolerance']
                is_vertical = abs(angle - 90) < params['angle_tolerance']
                
                if (is_horizontal or is_vertical) and self.check_wall_thickness(img, x1, y1, x2, y2, params, wall_thickness):
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
            walls = self.enhance_wall_classification(walls)
        
        return walls

    def preview_detection(self, page_pixmap, sensitivity: float = 50, wall_thickness: int = 5):
        """Enhanced preview with room detection visualization."""
        img = np.frombuffer(page_pixmap.samples, dtype=np.uint8).reshape(
            page_pixmap.height, page_pixmap.width, page_pixmap.n
        )
        
        if page_pixmap.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        preview_img = img.copy()
        walls = self.detect_walls(page_pixmap, sensitivity, wall_thickness)
        
        # Draw walls
        for wall in walls:
            if wall.wall_type == "EXTERIOR_WALL":
                color = self.OUTER_WALL_COLOR
            elif wall.wall_type == "PARTITION_WALL":
                color = self.PARTITION_WALL_COLOR
            else:
                color = self.INNER_WALL_COLOR
                
            start_point = (int(wall.start[0]), int(wall.start[1]))
            end_point = (int(wall.end[0]), int(wall.end[1]))
            cv2.line(preview_img, start_point, end_point, color, wall_thickness)
        
        # Draw junction points
        for junction in self.junction_points:
            cv2.circle(preview_img, (int(junction[0]), int(junction[1])), 5, (0, 0, 255), -1)
        
        # Draw room boundaries
        for room in self.room_boundaries:
            points = np.array(room, dtype=np.int32)
            cv2.polylines(preview_img, [points], True, (255, 165, 0), 2)
        
        return preview_img, len(walls)
    

    def detect_junctions(self, walls: List[WallLine], tolerance: float = 10) -> List[Tuple[float, float]]:
        """Detect wall junction points where walls intersect."""
        junctions = []
        
        for i, wall1 in enumerate(walls):
            for wall2 in walls[i+1:]:
                if wall1.is_horizontal == wall2.is_horizontal:
                    continue
                    
                h_wall = wall1 if wall1.is_horizontal else wall2
                v_wall = wall2 if wall1.is_horizontal else wall1
                
                h_y = h_wall.start[1]
                v_x = v_wall.start[0]
                
                h_x_min = min(h_wall.start[0], h_wall.end[0])
                h_x_max = max(h_wall.start[0], h_wall.end[0])
                v_y_min = min(v_wall.start[1], v_wall.end[1])
                v_y_max = max(v_wall.start[1], v_wall.end[1])
                
                if (h_x_min - tolerance <= v_x <= h_x_max + tolerance and
                    v_y_min - tolerance <= h_y <= v_y_max + tolerance):
                    junctions.append((v_x, h_y))
        
        self.junction_points = junctions
        return junctions

    def detect_rooms(self, walls: List[WallLine], min_room_area: float) -> List[List[Tuple[float, float]]]:
        """Detect closed rooms formed by walls."""
        junctions = self.detect_junctions(walls)
        rooms = []
        
        def get_connected_walls(point: Tuple[float, float]) -> List[WallLine]:
            connected = []
            for wall in walls:
                if (abs(wall.start[0] - point[0]) < 10 and abs(wall.start[1] - point[1]) < 10) or \
                   (abs(wall.end[0] - point[0]) < 10 and abs(wall.end[1] - point[1]) < 10):
                    connected.append(wall)
            return connected
        
        def find_room(start_point: Tuple[float, float], visited: Set[Tuple[float, float]] = None) -> List[Tuple[float, float]]:
            if visited is None:
                visited = set()
            
            room = [start_point]
            visited.add(start_point)
            
            connected_walls = get_connected_walls(start_point)
            for wall in connected_walls:
                next_point = wall.end if wall.start == start_point else wall.start
                if next_point not in visited:
                    room.extend(find_room(next_point, visited)[1:])
            
            return room
        
        for junction in junctions:
            if any(junction in room for room in rooms):
                continue
                
            room = find_room(junction)
            if len(room) >= 4:
                area = 0
                for i in range(len(room)):
                    j = (i + 1) % len(room)
                    area += room[i][0] * room[j][1]
                    area -= room[j][0] * room[i][1]
                area = abs(area) / 2
                
                if area >= min_room_area:
                    rooms.append(room)
        
        self.room_boundaries = rooms
        return rooms

    def enhance_wall_classification(self, walls: List[WallLine]) -> List[WallLine]:
        """Enhanced wall classification using junction analysis and room detection."""
        # Create new mutable walls for modification
        mutable_walls = []
        for wall in walls:
            new_wall = WallLine(
                start=wall.start,
                end=wall.end,
                is_horizontal=wall.is_horizontal,
                is_outer=wall.is_outer,
                length=wall.length,
                wall_type=wall.wall_type
            )
            mutable_walls.append(new_wall)

        # Detect rooms
        params = self.calculate_parameters(50)  # Default sensitivity
        self.detect_rooms(mutable_walls, params['min_room_area'])
        
        # Count how many rooms each wall belongs to
        wall_room_count = defaultdict(int)
        for room in self.room_boundaries:
            room_lines = list(zip(room, room[1:] + [room[0]]))
            for wall in mutable_walls:
                wall_line = (wall.start, wall.end)
                for room_line in room_lines:
                    if self.lines_match(wall_line, room_line):
                        wall_room_count[wall] += 1
        
        # Update classifications
        classified_walls = []
        for wall in mutable_walls:
            new_wall = WallLine(
                start=wall.start,
                end=wall.end,
                is_horizontal=wall.is_horizontal,
                is_outer=(wall_room_count[wall] == 1),
                length=wall.length,
                wall_type=("EXTERIOR_WALL" if wall_room_count[wall] == 1 else
                          "INTERIOR_WALL" if wall_room_count[wall] > 1 else
                          "PARTITION_WALL")
            )
            classified_walls.append(new_wall)
        
        return classified_walls

    
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
                            x_min = min(wall1.start[0], wall1.end[0], wall2.start[0], wall2.end[0])
                            x_max = max(wall1.start[0], wall1.end[0], wall2.start[0], wall2.end[0])
                            if x_max - x_min <= (wall1.length + wall2.length + tolerance):
                                current_group.append(wall2)
                                used.add(j)
                    else:
                        if abs(wall1.start[0] - wall2.start[0]) <= tolerance:
                            y_min = min(wall1.start[1], wall1.end[1], wall2.start[1], wall2.end[1])
                            y_max = max(wall1.start[1], wall1.end[1], wall2.start[1], wall2.end[1])
                            if y_max - y_min <= (wall1.length + wall2.length + tolerance):
                                current_group.append(wall2)
                                used.add(j)
            
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

    def lines_match(self, line1: Tuple[Tuple[float, float], Tuple[float, float]], 
                   line2: Tuple[Tuple[float, float], Tuple[float, float]], 
                   tolerance: float = 10) -> bool:
        """Check if two lines match within tolerance."""
        def point_match(p1, p2):
            return abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance
        
        return (point_match(line1[0], line2[0]) and point_match(line1[1], line2[1])) or \
               (point_match(line1[0], line2[1]) and point_match(line1[1], line2[0]))


    def process_page(self, pdf_document, page_number: int, sensitivity: float = 50):
        """Process a single page and add wall annotations."""
        page = pdf_document[page_number]
        page_rect = page.rect
        pix = page.get_pixmap()
        
        detected_walls = self.detect_walls(pix, sensitivity)
        
        for wall in detected_walls:
            pdf_x1 = wall.start[0] * page_rect.width / pix.width
            pdf_y1 = wall.start[1] * page_rect.height / pix.height
            pdf_x2 = wall.end[0] * page_rect.width / pix.width
            pdf_y2 = wall.end[1] * page_rect.height / pix.height
            
            self.add_wall_annotation(
                page,
                (pdf_x1, pdf_y1),
                (pdf_x2, pdf_y2),
                wall.wall_type
            )
        
        return len(detected_walls)

    def add_wall_annotation(self, page, start_point, end_point, wall_type):
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

    # Initialize detector
    detector = EnhancedWallDetector()

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

        # Add wall thickness slider
        wall_thickness = st.slider("Wall Thickness (pixels)", 1, 20, 5, 
                                 help="Adjust based on the thickness of walls in your floor plan")
        
        # Sensitivity slider
        sensitivity = st.slider("Wall Detection Sensitivity", 0, 100, 50)

        # Real-time preview with wall thickness parameter
        preview_img, wall_count = detector.preview_detection(pix, sensitivity, wall_thickness)
        st.image(preview_img, caption=f"Preview of detected walls - {wall_count} walls found", 
                use_column_width=True)


        # Show processed pages
        if st.session_state.processed_pages:
            st.write("Processed pages:", ", ".join(f"Page {p+1}" for p in sorted(st.session_state.processed_pages)))

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Apply Wall Detection"):
                with st.spinner("Processing..."):
                    wall_count = detector.process_page(  # Use hybrid detector instead of local_detector
                        st.session_state.pdf_document,
                        selected_index,
                        sensitivity
                    )
                    st.session_state.processed_pages.add(selected_index)
                    st.session_state.wall_count = wall_count
                    if api_key:
                        st.success(f"Detected {wall_count} walls using hybrid detection")
                    else:
                        st.success(f"Detected {wall_count} walls using local detection")

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