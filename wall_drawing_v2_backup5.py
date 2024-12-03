import streamlit as st
import fitz  # PyMuPDF
import cv2
import numpy as np
import io
import anthropic
from datetime import datetime
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import base64
import json 
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


@dataclass
class WallLine:
    start: Tuple[float, float]
    end: Tuple[float, float]
    is_horizontal: bool
    is_outer: bool
    length: float
    wall_type: str = "INTERIOR_WALL"

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
            'canny_low': 75,
            'canny_high': 200,
            'hough_threshold': 70,
            'min_line_length': 80,
            'max_line_gap': 15,
            'angle_tolerance': 1.5,
            'duplicate_tolerance': 8,
            'min_wall_thickness': 5,
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
        
        # Find the bounding box of all walls
        all_x = [coord for wall in walls for coord in (wall.start[0], wall.end[0])]
        all_y = [coord for wall in walls for coord in (wall.start[1], wall.end[1])]
        
        if not all_x or not all_y:
            return walls
            
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        bounding_width = max_x - min_x
        bounding_height = max_y - min_y
        
        for wall in walls:
            if wall.is_horizontal:
                is_at_edge = (abs(wall.start[1] - min_y) < 20 or abs(wall.start[1] - max_y) < 20)
                is_long = wall.length > bounding_width * 0.7
                wall.is_outer = is_at_edge and is_long
            else:
                is_at_edge = (abs(wall.start[0] - min_x) < 20 or abs(wall.start[0] - max_x) < 20)
                is_long = wall.length > bounding_height * 0.7
                wall.is_outer = is_at_edge and is_long
            
            wall.wall_type = "EXTERIOR_WALL" if wall.is_outer else "INTERIOR_WALL"
        
        return walls

    def check_wall_thickness(self, img, x1, y1, x2, y2, params):
        """Check if line represents a wall by verifying its thickness."""
        thickness = params['min_wall_thickness']
        mask = np.zeros_like(img)
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, int(thickness * 2))
        
        intersection = cv2.bitwise_and(img, mask)
        non_zero = cv2.countNonZero(intersection)
        total = cv2.countNonZero(mask)
        
        if total == 0:
            return False
        
        ratio = non_zero / total
        return ratio > 0.3

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
        blurred = cv2.bilateralFilter(img, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        
        edges = cv2.Canny(
            thresh,
            params['canny_low'],
            params['canny_high'],
            apertureSize=3,
            L2gradient=True
        )
        
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
            pdf_x1 = wall.start[0] * page_rect.width / pix.width
            pdf_y1 = wall.start[1] * page_rect.height / pix.height
            pdf_x2 = wall.end[0] * page_rect.width / pix.width
            pdf_y2 = wall.end[1] * page_rect.height / pix.height
            
            wall_type = "Outer Wall" if wall.is_outer else "Inner Wall"
            
            self.add_wall_annotation(
                page,
                (pdf_x1, pdf_y1),
                (pdf_x2, pdf_y2),
                wall_type
            )
        
        return len(detected_walls)

class HybridWallDetector:
    def __init__(self, api_key: Optional[str] = None):
        self.local_detector = WallDetectorPDF()
        self.client = anthropic.Client(api_key=api_key) if api_key else None

    def get_claude_prompt(self) -> str:
        return """Analyze this architectural floor plan and:
        1. Identify all visible walls and lines
        2. For each wall/line, provide:
           - Start coordinates (x1, y1)
           - End coordinates (x2, y2)
           - Wall type classification:
             * EXTERIOR_WALL: outer perimeter walls
             * INTERIOR_WALL: main internal walls
             * PARTITION_WALL: lighter internal divisions
        3. Return data in this JSON format:
        {
          "walls": [
            {
              "type": "EXTERIOR_WALL",
              "coordinates": {
                "start": {"x": 0, "y": 0},
                "end": {"x": 100, "y": 0}
              }
            }
          ]
        }
        Focus only on structural walls. Ignore annotations, text, and dimensions.
        Use normalized coordinates (0-100 scale).
        """

    def extract_json_from_claude_response(self, response) -> dict:
        """Extract JSON content from Claude's text response."""
        # Get the text content from Claude's response
        text_content = response.content[0].text
        
        # Find JSON content using regex
        json_match = re.search(r'{\s*"walls":\s*\[.*?\]\s*}', text_content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                raise ValueError("Found JSON-like structure but couldn't parse it")
        raise ValueError("No valid JSON structure found in response")

    def detect_with_claude(self, image: np.ndarray) -> List[WallLine]:
        """Detect walls using Claude Vision API with updated parsing"""
        if not self.client:
            return []

        try:
            # Convert numpy array to base64
            _, buffer = cv2.imencode('.png', image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # Get response from Claude
            message = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": self.get_claude_prompt()
                        }
                    ]
                }]
            )

            # Parse response
            try:
                json_data = extract_json_from_claude_response(message)
                analysis = FloorPlanAnalysis.parse_obj(json_data)
                
                # Convert to WallLine objects
                claude_walls = []
                for wall in analysis.walls:
                    start = (
                        wall.coordinates["start"]["x"],
                        wall.coordinates["start"]["y"]
                    )
                    end = (
                        wall.coordinates["end"]["x"],
                        wall.coordinates["end"]["y"]
                    )
                    
                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    is_horizontal = abs(dy) < abs(dx)
                    length = np.sqrt(dx**2 + dy**2)
                    
                    wall_line = WallLine(
                        start=start,
                        end=end,
                        is_horizontal=is_horizontal,
                        is_outer=(wall.type == 'EXTERIOR_WALL'),
                        length=length,
                        wall_type=wall.type
                    )
                    claude_walls.append(wall_line)
                
                return claude_walls

            except ValueError as e:
                st.warning(f"Could not parse Claude's response: {str(e)}")
                return []
            except Exception as e:
                st.error(f"Error processing Claude's response: {str(e)}")
                return []

        except Exception as e:
            st.error(f"Claude API error: {str(e)}")
            return []


    def combine_detections(self, image: np.ndarray, sensitivity: float = 50) -> List[WallLine]:
        """Combine wall detections from both methods"""
        # Get local detections
        pix = fitz.Pixmap(image.tobytes(), image.shape[1], image.shape[0], 3)
        local_walls = self.local_detector.detect_walls(pix, sensitivity)

        # Get Claude detections if available
        claude_walls = self.detect_with_claude(image) if self.client else []

        # Combine and de-duplicate walls
        all_walls = local_walls + claude_walls
        return self.merge_similar_walls(all_walls)
    
    def preview_detection(self, page_pixmap, sensitivity: float = 50):
        """Generate a preview image with detected walls from both methods."""
        img = np.frombuffer(page_pixmap.samples, dtype=np.uint8).reshape(
            page_pixmap.height, page_pixmap.width, page_pixmap.n
        )
        
        if page_pixmap.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        preview_img = img.copy()
        
        # Get walls from both detectors
        local_walls = self.local_detector.detect_walls(page_pixmap, sensitivity)
        claude_walls = self.detect_with_claude(img) if self.client else []
        
        # Combine walls
        all_walls = self.merge_similar_walls(local_walls + claude_walls)
        
        # Draw walls with different colors
        for wall in all_walls:
            if wall.wall_type == "EXTERIOR_WALL":
                color = (255, 0, 0)  # Red for exterior
            elif wall.wall_type == "PARTITION_WALL":
                color = (0, 0, 255)  # Blue for partitions
            else:
                color = (0, 255, 0)  # Green for interior
                
            start_point = (int(wall.start[0]), int(wall.start[1]))
            end_point = (int(wall.end[0]), int(wall.end[1]))
            cv2.line(preview_img, start_point, end_point, color, 2)
        
        return preview_img, len(all_walls)

    def process_page(self, pdf_document, page_number: int, sensitivity: float = 50):
        """Process a single page using both detection methods."""
        page = pdf_document[page_number]
        page_rect = page.rect
        pix = page.get_pixmap()
        
        # Get image for Claude
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Get walls from both detectors
        local_walls = self.local_detector.detect_walls(pix, sensitivity)
        claude_walls = self.detect_with_claude(img) if self.client else []
        
        # Combine walls
        all_walls = self.merge_similar_walls(local_walls + claude_walls)
        
        # Add annotations
        for wall in all_walls:
            pdf_x1 = wall.start[0] * page_rect.width / pix.width
            pdf_y1 = wall.start[1] * page_rect.height / pix.height
            pdf_x2 = wall.end[0] * page_rect.width / pix.width
            pdf_y2 = wall.end[1] * page_rect.height / pix.height
            
            self.local_detector.add_wall_annotation(
                page,
                (pdf_x1, pdf_y1),
                (pdf_x2, pdf_y2),
                wall.wall_type
            )
        
        return len(all_walls)

    def detect_with_claude(self, image: np.ndarray) -> List[WallLine]:
        """Detect walls using Claude Vision API"""
        if not self.client:
            return []

        try:
            # Convert numpy array to base64
            _, buffer = cv2.imencode('.png', image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            message = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": self.get_claude_prompt()
                        }
                    ]
                }]
            )
            print(message.content[0].text)
            # Parse response and convert to WallLine objects
            try:
                response_data = json.loads(message.content[0].text)
                
            except json.JSONDecodeError:
                st.warning("Could not parse Claude's response as JSON")
                return []

            claude_walls = []
            for wall_data in response_data.get('walls', []):
                start = (
                    wall_data['coordinates']['start']['x'],
                    wall_data['coordinates']['start']['y']
                )
                end = (
                    wall_data['coordinates']['end']['x'],
                    wall_data['coordinates']['end']['y']
                )

                dx = end[0] - start[0]
                dy = end[1] - start[1]
                is_horizontal = abs(dy) < abs(dx)
                length = np.sqrt(dx**2 + dy**2)

                wall = WallLine(
                    start=start,
                    end=end,
                    is_horizontal=is_horizontal,
                    is_outer=(wall_data['type'] == 'EXTERIOR_WALL'),
                    length=length,
                    wall_type=wall_data['type']
                )
                claude_walls.append(wall)

            return claude_walls

        except Exception as e:
            st.error(f"Claude API error: {str(e)}")
            return []

    def merge_similar_walls(self, walls: List[WallLine], tolerance: float = 10) -> List[WallLine]:
        """Merge similar walls from both detection methods"""
        if not walls:
            return []
            
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
                    
                if self.are_walls_similar(wall1, wall2, tolerance):
                    current_group.append(wall2)
                    used.add(j)
            
            merged_wall = self.merge_wall_group(current_group)
            if merged_wall:
                merged.append(merged_wall)
            
        return merged

    def are_walls_similar(self, wall1: WallLine, wall2: WallLine, tolerance: float) -> bool:
        """Check if two walls are similar enough to merge"""
        if wall1.is_horizontal != wall2.is_horizontal:
            return False
            
        if wall1.is_horizontal:
            return (abs(wall1.start[1] - wall2.start[1]) <= tolerance and
                   self.have_x_overlap(wall1, wall2, tolerance))
        else:
            return (abs(wall1.start[0] - wall2.start[0]) <= tolerance and
                   self.have_y_overlap(wall1, wall2, tolerance))

    def have_x_overlap(self, wall1: WallLine, wall2: WallLine, tolerance: float) -> bool:
        x1_min, x1_max = min(wall1.start[0], wall1.end[0]), max(wall1.start[0], wall1.end[0])
        x2_min, x2_max = min(wall2.start[0], wall2.end[0]), max(wall2.start[0], wall2.end[0])
        return (x1_min <= x2_max + tolerance and x2_min <= x1_max + tolerance)

    def have_y_overlap(self, wall1: WallLine, wall2: WallLine, tolerance: float) -> bool:
        y1_min, y1_max = min(wall1.start[1], wall1.end[1]), max(wall1.start[1], wall1.end[1])
        y2_min, y2_max = min(wall2.start[1], wall2.end[1]), max(wall2.start[1], wall2.end[1])
        return (y1_min <= y2_max + tolerance and y2_min <= y1_max + tolerance)

    def merge_wall_group(self, walls: List[WallLine]) -> Optional[WallLine]:
        """Merge a group of similar walls into one"""
        if not walls:
            return None
            
        # Use the most common wall type and outer/inner classification
        wall_types = [w.wall_type for w in walls]
        is_outer = sum(1 for w in walls if w.is_outer) > len(walls) / 2
        
        if walls[0].is_horizontal:
            x_coords = [x for w in walls for x in (w.start[0], w.end[0])]
            y_coords = [w.start[1] for w in walls]
            merged = WallLine(
                start=(min(x_coords), np.mean(y_coords)),
                end=(max(x_coords), np.mean(y_coords)),
                is_horizontal=True,
                is_outer=is_outer,
                length=max(x_coords) - min(x_coords),
                wall_type=max(set(wall_types), key=wall_types.count)
            )
        else:
            x_coords = [w.start[0] for w in walls]
            y_coords = [y for w in walls for y in (w.start[1], w.end[1])]
            merged = WallLine(
                start=(np.mean(x_coords), min(y_coords)),
                end=(np.mean(x_coords), max(y_coords)),
                is_horizontal=False,
                is_outer=is_outer,
                length=max(y_coords) - min(y_coords),
                wall_type=max(set(wall_types), key=wall_types.count)
            )
            
        return merged

def main():
    st.title("Hybrid Wall Detector")
    st.write("Detects walls in floor plans using computer vision and Claude AI")

    # Initialize session state
    if 'processed_pages' not in st.session_state:
        st.session_state.processed_pages = set()
    if 'pdf_document' not in st.session_state:
        st.session_state.pdf_document = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'wall_count' not in st.session_state:
        st.session_state.wall_count = None

    # API key input in sidebar
    with st.sidebar:
        api_key = st.text_input("Claude API Key (optional)", type="password")
        st.write("Leave blank to use only local detection")

    # Initialize detector
    detector = HybridWallDetector(api_key if api_key else None)

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

        # Sensitivity slider
        sensitivity = st.slider("Wall Detection Sensitivity", 0, 100, 50)

        # Real-time preview
        preview_img, wall_count = detector.preview_detection(pix, sensitivity)
    
        if api_key:
            st.image(preview_img, caption=f"Preview of detected walls (Hybrid detection) - {wall_count} walls found", use_column_width=True)
        else:
            st.image(preview_img, caption=f"Preview of detected walls (Local detection only) - {wall_count} walls found", use_column_width=True)


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