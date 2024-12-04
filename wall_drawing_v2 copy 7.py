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
        """Parameters optimized specifically for architectural floor plans."""
        normalized = sensitivity / 100
        
        # Enhanced base parameters for better detection
        base_params = {
            'canny_low': 75,
            'canny_high': 200,
            'hough_threshold': 70,
            'min_line_length': 80,
            'max_line_gap': 15,
            'angle_tolerance': 1.5,
            'duplicate_tolerance': 8,
            'min_wall_thickness': 5,
            'outer_wall_threshold': 0.85,
            'junction_tolerance': 10,
            'min_room_area': 1000
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
            'outer_wall_threshold': 0.75,
            'junction_tolerance': 15,
            'min_room_area': 500
        }
        
        params = {}
        for key in base_params:
            params[key] = base_params[key] + (max_params[key] - base_params[key]) * normalized
            if key in ['canny_low', 'canny_high', 'hough_threshold', 'min_line_length', 'max_line_gap']:
                params[key] = int(params[key])
        
        return params
    
    def _filter_text(self, img: np.ndarray, params: dict) -> Tuple[List[np.ndarray], np.ndarray]:
        """Text filtering with configurable parameters."""
        # Create MSER detector
        mser = cv2.MSER_create(
            delta=params.get('mser_delta', 5),
            min_area=params.get('min_text_area', 30)
        )
        
        # Detect text regions
        regions, _ = mser.detectRegions(img)
        
        # Create text mask
        mask = np.ones_like(img)
        text_regions = []
        
        for region in regions:
            # Calculate region properties
            x, y, w, h = cv2.boundingRect(region)
            aspect_ratio = w / h if h != 0 else 0
            
            # Filter out non-text regions based on shape
            if 0.1 < aspect_ratio < 10 and w < img.shape[1] / 3:
                hull = cv2.convexHull(region.reshape(-1, 1, 2))
                text_regions.append(hull)
                cv2.fillPoly(mask, [hull], 0)
        
        # Apply cleanup
        cleanup_strength = params.get('text_cleanup', 5)
        kernel_size = 2 * cleanup_strength + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask and enhance contrast
        cleaned = cv2.bitwise_and(img, img, mask=mask)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cleaned = clahe.apply(cleaned)
        
        return text_regions, cleaned

    def _detect_edges(self, img: np.ndarray, params: dict) -> np.ndarray:
        """Enhanced edge detection with noise control."""
        # Apply noise reduction
        noise_reduction = params.get('noise_reduction', 50)
        if noise_reduction > 0:
            sigma = noise_reduction / 25.0
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img)
        
        # Apply adaptive thresholding
        edge_sensitivity = params.get('edge_sensitivity', 50)
        block_size = 11 + 2 * (edge_sensitivity // 20)
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            2
        )
        
        # Edge detection
        edges = cv2.Canny(
            thresh,
            edge_sensitivity,
            edge_sensitivity * 2,
            apertureSize=3,
            L2gradient=True
        )
        
        # Multi-directional enhancement
        kernels = [
            np.ones((3,1), np.uint8),  # Vertical
            np.ones((1,3), np.uint8),  # Horizontal
            np.array([[1,0,0],[0,1,0],[0,0,1]], np.uint8),  # Diagonal 45°
            np.array([[0,0,1],[0,1,0],[1,0,0]], np.uint8)   # Diagonal 135°
        ]
        
        enhanced_edges = np.zeros_like(edges)
        for kernel in kernels:
            dilated = cv2.dilate(edges, kernel, iterations=1)
            enhanced_edges = cv2.bitwise_or(enhanced_edges, dilated)
        
        return enhanced_edges

    def _is_diagonal(self, wall: WallLine, params: dict) -> bool:
        """Check if wall is diagonal."""
        angle = np.degrees(np.arctan2(
            wall.end[1] - wall.start[1],
            wall.end[0] - wall.start[0]
        )) % 180
        
        angle_tolerance = params.get('angle_tolerance', 3)
        return (
            abs(angle - 45) < angle_tolerance or
            abs(angle - 135) < angle_tolerance
        )

    def get_default_params(self) -> dict:
        """Get default parameters for wall detection."""
        return {
            # Text filtering params
            'mser_delta': 5,
            'min_text_area': 30,
            'text_cleanup': 5,
            
            # Wall detection params
            'min_wall_thickness': 5,
            'max_line_gap': 15,
            'edge_sensitivity': 50,
            'min_wall_length': 80,
            
            # Diagonal detection params
            'enable_diagonals': True,
            'angle_tolerance': 3,
            'diagonal_thickness_factor': 1.4,
            
            # Advanced processing params
            'noise_reduction': 50,
            'junction_tolerance': 10,
            'show_text_regions': False,
            
            # Edge detection params
            'canny_low': 75,
            'canny_high': 200,
            'hough_threshold': 70,
            'duplicate_tolerance': 8,
            'outer_wall_threshold': 0.85,
            'min_room_area': 1000
        }

        
    def check_wall_thickness(self, img, x1, y1, x2, y2, params):
        """Improved wall thickness verification with orientation awareness."""
        thickness = params['min_wall_thickness']  # Changed from thickness_min
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        
        mask = np.zeros_like(img)
        
        if abs(angle - 45) < 20 or abs(angle - 135) < 20:
            thickness = int(thickness * params['diagonal_thickness_factor'])
        
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, int(thickness * 2))
        
        if abs(angle - 45) < 20 or abs(angle - 135) < 20:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        intersection = cv2.bitwise_and(img, mask)
        non_zero = cv2.countNonZero(intersection)
        total = cv2.countNonZero(mask)
        
        if total == 0:
            return False
        
        ratio = non_zero / total
        min_ratio = 0.25 if (abs(angle - 45) < 20 or abs(angle - 135) < 20) else 0.3
        
        return ratio > min_ratio

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

    def detect_walls(self, page_pixmap, sensitivity: float = 50) -> List[WallLine]:
        """Enhanced wall detection with diagonal line support and text filtering."""
        img = np.frombuffer(page_pixmap.samples, dtype=np.uint8).reshape(
            page_pixmap.height, page_pixmap.width, page_pixmap.n
        )
        
        if page_pixmap.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        elif page_pixmap.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        params = self.calculate_parameters(sensitivity)
        
        # Enhanced preprocessing for text removal
        blurred = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Text removal using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        
        # MSER to detect and remove text regions
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(morph)
        mask = np.ones_like(morph)
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.fillPoly(mask, [hull], (0))
        
        # Apply mask to remove text
        cleaned = cv2.bitwise_and(morph, morph, mask=mask)
        
        # Enhanced edge detection
        edges = cv2.Canny(
            cleaned,
            params['canny_low'],
            params['canny_high'],
            apertureSize=3,
            L2gradient=True
        )
        
        # Multi-directional dilation for better line connectivity
        kernels = [
            np.ones((3,1), np.uint8),  # Vertical
            np.ones((1,3), np.uint8),  # Horizontal
            np.array([[1,0,0],[0,1,0],[0,0,1]], np.uint8),  # Diagonal 45°
            np.array([[0,0,1],[0,1,0],[1,0,0]], np.uint8)   # Diagonal 135°
        ]
        
        enhanced_edges = np.zeros_like(edges)
        for kernel in kernels:
            dilated = cv2.dilate(edges, kernel, iterations=1)
            enhanced_edges = cv2.bitwise_or(enhanced_edges, dilated)
        
        # Modified Hough transform parameters for diagonal lines
        lines = cv2.HoughLinesP(
            enhanced_edges,
            rho=1,
            theta=np.pi/360,  # Increased angle resolution
            threshold=int(params['hough_threshold'] * 0.8),  # Slightly lower threshold
            minLineLength=params['min_line_length'],
            maxLineGap=params['max_line_gap'] * 1.5  # Increased gap tolerance
        )
        
        walls = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
                
                # Allow diagonal lines within specific angle ranges
                is_horizontal = abs(angle) < params['angle_tolerance'] or abs(angle - 180) < params['angle_tolerance']
                is_vertical = abs(angle - 90) < params['angle_tolerance']
                is_diagonal = (
                    abs(angle - 45) < params['angle_tolerance'] * 2 or 
                    abs(angle - 135) < params['angle_tolerance'] * 2
                )
                
                if (is_horizontal or is_vertical or is_diagonal) and self.check_wall_thickness(cleaned, x1, y1, x2, y2, params):
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

    def preview_detection(self, page_pixmap, params: dict):
        """Enhanced preview with detailed statistics and visualization."""
        # Ensure we have all required parameters by merging with defaults
        default_params = self.get_default_params()
        params = {**default_params, **params}  # Override defaults with provided params
        
        # Convert pixmap to OpenCV image
        img = np.frombuffer(page_pixmap.samples, dtype=np.uint8).reshape(
            page_pixmap.height, page_pixmap.width, page_pixmap.n
        )
        
        if page_pixmap.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Make a copy for visualization
        preview_img = img.copy()
        
        # Convert to grayscale for processing
        if page_pixmap.n == 3 or page_pixmap.n == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Text filtering stage
        text_regions, cleaned_img = self._filter_text(gray, params)
        
        # Edge detection stage
        edges = self._detect_edges(cleaned_img, params)
        
        # Wall detection stage
        walls, stats = self._detect_walls_with_stats(edges, params)
        
        # Prepare statistics
        detection_stats = {
            'total_walls': len(walls),
            'diagonal_walls': sum(1 for w in walls if self._is_diagonal(w, params)),  # Pass params here
            'text_regions': len(text_regions),
            'horizontal_walls': sum(1 for w in walls if w.is_horizontal),
            'vertical_walls': sum(1 for w in walls if not w.is_horizontal and not self._is_diagonal(w, params)),  # And here
            'merged_walls': stats['merged_count'],
            'filtered_lines': stats['filtered_count']
        }
        
        # Visualization
        # First draw original detected edges faintly
        edge_overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        preview_img = cv2.addWeighted(preview_img, 0.7, edge_overlay, 0.3, 0)
        
        # Draw walls with different colors based on type
        for wall in walls:
            if self._is_diagonal(wall, params):  # And here
                color = (255, 165, 0)  # Orange for diagonal walls
            elif wall.is_horizontal:
                color = (0, 255, 0)    # Green for horizontal walls
            else:
                color = (0, 0, 255)    # Red for vertical walls
                
            start_point = (int(wall.start[0]), int(wall.start[1]))
            end_point = (int(wall.end[0]), int(wall.end[1]))
            cv2.line(preview_img, start_point, end_point, color, 2)
        
        # Draw junction points
        junctions = self.detect_junctions(walls, params['junction_tolerance'])
        for junction in junctions:
            cv2.circle(preview_img, (int(junction[0]), int(junction[1])), 4, (255, 0, 255), -1)
        
        # Highlight text regions if requested
        if params.get('show_text_regions', False):
            text_overlay = preview_img.copy()
            for region in text_regions:
                cv2.fillPoly(text_overlay, [region], (0, 255, 255))
            preview_img = cv2.addWeighted(preview_img, 0.7, text_overlay, 0.3, 0)
        
        return preview_img, detection_stats

    def _detect_walls_with_stats(self, edges: np.ndarray, params: dict) -> Tuple[List[WallLine], dict]:
        """Detect walls and collect statistics about the detection process."""
        stats = {
            'merged_count': 0,
            'filtered_count': 0,
            'initial_lines': 0
        }
        
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/360 if params['enable_diagonals'] else np.pi/2,
            threshold=params['edge_sensitivity'],
            minLineLength=params['min_wall_length'],
            maxLineGap=params['max_line_gap']
        )
        
        walls = []
        if lines is not None:
            stats['initial_lines'] = len(lines)
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
                
                is_horizontal = abs(angle) < params['angle_tolerance'] or abs(angle - 180) < params['angle_tolerance']
                is_vertical = abs(angle - 90) < params['angle_tolerance']
                is_diagonal = params['enable_diagonals'] and (
                    abs(angle - 45) < params['angle_tolerance'] * 2 or 
                    abs(angle - 135) < params['angle_tolerance'] * 2
                )
                
                if (is_horizontal or is_vertical or is_diagonal):
                    thickness_factor = params['diagonal_thickness_factor'] if is_diagonal else 1.0
                    if self.check_wall_thickness(edges, x1, y1, x2, y2, params):  # Passing entire params dict
                        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        if length >= params['min_wall_length']:
                            wall = WallLine(
                                start=(float(x1), float(y1)),
                                end=(float(x2), float(y2)),
                                is_horizontal=is_horizontal,
                                is_outer=False,
                                length=length
                            )
                            walls.append(wall)
                        else:
                            stats['filtered_count'] += 1
                    else:
                        stats['filtered_count'] += 1
        
        merged_walls = self.merge_walls(walls, params)
        stats['merged_count'] = len(walls) - len(merged_walls)
        
        classified_walls = self.enhance_wall_classification(merged_walls)
        
        return classified_walls, stats

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
    st.set_page_config(layout="wide")
    st.title("Enhanced Wall Detector")
    st.write("Advanced wall detection for architectural floor plans with detailed controls")

    # Initialize session state
    if 'processed_pages' not in st.session_state:
        st.session_state.processed_pages = set()
    if 'pdf_document' not in st.session_state:
        st.session_state.pdf_document = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'wall_count' not in st.session_state:
        st.session_state.wall_count = None
    if 'last_params' not in st.session_state:
        st.session_state.last_params = None

    # Initialize detector
    detector = EnhancedWallDetector()

    # File upload section
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

        # Create two columns for controls and preview
        col1, col2 = st.columns([1, 2])

        with col1:
            # Quick settings
            st.subheader("Quick Settings")
            quick_mode = st.radio(
                "Detection Mode",
                ["Basic", "Advanced"],
                help="Basic mode uses preset parameters, Advanced mode allows detailed control"
            )

            if quick_mode == "Basic":
                sensitivity = st.slider("Overall Sensitivity", 0, 100, 50,
                    help="Higher values detect more potential walls")
                quality = st.select_slider(
                    "Drawing Quality",
                    options=["Draft", "Normal", "High Quality"],
                    value="Normal",
                    help="Adjust based on your drawing quality"
                )
                wall_style = st.multiselect(
                    "Wall Types to Detect",
                    ["Straight", "Diagonal"],
                    default=["Straight"],
                    help="Select which types of walls to detect"
                )
                
                # Convert basic settings to detailed parameters
                params = detector.get_default_params()
                params.update({
                    'edge_sensitivity': sensitivity,
                    'enable_diagonals': "Diagonal" in wall_style,
                    'quality_preset': quality
                })

            else:  # Advanced mode
                with st.expander("Text Filtering", expanded=False):
                    text_params = {
                        'mser_delta': st.slider("MSER Sensitivity", 1, 10, 5, 
                            help="Lower values detect more text regions"),
                        'min_text_area': st.slider("Minimum Text Size", 10, 100, 30, 
                            help="Minimum area for text detection"),
                        'text_cleanup': st.slider("Text Cleanup Strength", 1, 10, 5,
                            help="Strength of morphological operations for text removal")
                    }

                with st.expander("Wall Detection", expanded=False):
                    wall_params = {
                        'min_wall_thickness': st.slider("Minimum Wall Thickness", 1, 20, 5,
                            help="Minimum thickness to consider a line as wall"),
                        'max_line_gap': st.slider("Maximum Line Gap", 5, 50, 15,
                            help="Maximum gap between line segments to merge"),
                        'edge_sensitivity': st.slider("Edge Detection Sensitivity", 0, 100, 50,
                            help="Sensitivity of edge detection")
                    }

                # For basic mode conversion:
                if quick_mode == "Basic":
                    params = detector.get_default_params()
                    quality_mappings = {
                        "Draft": {'min_wall_thickness': 3, 'max_line_gap': 25, 'edge_sensitivity': 30},
                        "Normal": {'min_wall_thickness': 5, 'max_line_gap': 15, 'edge_sensitivity': 50},
                        "High Quality": {'min_wall_thickness': 7, 'max_line_gap': 10, 'edge_sensitivity': 70}
                    }
                    params.update({
                        'edge_sensitivity': sensitivity,
                        'enable_diagonals': "Diagonal" in wall_style,
                        **quality_mappings[quality]
                    })

                with st.expander("Diagonal Lines", expanded=False):
                    diagonal_params = {
                        'enable_diagonals': st.checkbox("Enable Diagonal Detection", True),
                        'angle_tolerance': st.slider("Angle Tolerance (degrees)", 1, 10, 3,
                            help="Tolerance for diagonal angle detection"),
                        'diagonal_thickness_factor': st.slider("Diagonal Thickness Factor", 1.0, 2.0, 1.4, 0.1,
                            help="Multiplier for diagonal wall thickness checking")
                    }

                with st.expander("Advanced Processing", expanded=False):
                    advanced_params = {
                        'noise_reduction': st.slider("Noise Reduction", 0, 100, 50,
                            help="Strength of noise reduction preprocessing"),
                        'junction_tolerance': st.slider("Junction Detection Tolerance", 1, 20, 10,
                            help="Tolerance for detecting wall intersections"),
                        'min_wall_length': st.slider("Minimum Wall Length", 10, 200, 80,  # Changed from min_wall_length
                            help="Minimum length to consider a line as wall")
                    }

                # Combine all parameters
                params = {
                    **text_params,
                    **wall_params,
                    **diagonal_params,
                    **advanced_params
                }

            # Save parameters button
            if st.button("Save as Default Parameters"):
                st.session_state.last_params = params
                st.success("Parameters saved!")

            # Load saved parameters button
            if st.session_state.last_params is not None:
                if st.button("Load Saved Parameters"):
                    params = st.session_state.last_params
                    st.experimental_rerun()

        with col2:
            # Preview section
            st.subheader("Preview")
            preview_img, stats = detector.preview_detection(pix, params)

            # Display statistics in a grid
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            with stat_col1:
                st.metric("Total Walls", stats['total_walls'])
                st.metric("Horizontal Walls", stats['horizontal_walls'])
            with stat_col2:
                st.metric("Vertical Walls", stats['vertical_walls'])
                st.metric("Diagonal Walls", stats['diagonal_walls'])
            with stat_col3:
                st.metric("Text Regions Filtered", stats['text_regions'])
                st.metric("Merged Wall Segments", stats['merged_walls'])

            # Display preview image
            st.image(preview_img, caption="Wall Detection Preview", use_column_width=True)

        # Processing buttons
        st.subheader("Process and Export")
        process_col1, process_col2 = st.columns(2)

        with process_col1:
            if st.button("Apply Wall Detection", key="apply_detection"):
                with st.spinner("Processing..."):
                    wall_count = detector.process_page(
                        st.session_state.pdf_document,
                        selected_index,
                        params
                    )
                    st.session_state.processed_pages.add(selected_index)
                    st.session_state.wall_count = wall_count
                    st.success(f"Successfully processed page {selected_index + 1}")

        with process_col2:
            if len(st.session_state.processed_pages) > 0:
                if st.button("Download Annotated PDF", key="download_pdf"):
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
                st.button("Download Annotated PDF", disabled=True, 
                         help="Process at least one page first")

        # Show processed pages
        if st.session_state.processed_pages:
            st.write("Processed pages:", ", ".join(f"Page {p+1}" 
                    for p in sorted(st.session_state.processed_pages)))

if __name__ == "__main__":
    main()
 