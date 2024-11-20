import streamlit as st
import fitz  # PyMuPDF
import cv2
import numpy as np
import io
from datetime import datetime
from PIL import Image

class WallDetectorPDF:
    def __init__(self):
        self.stroke_color = (0, 0, 1)  # Default blue color
        self.line_width = 2.0

    def calculate_parameters(self, sensitivity: float) -> dict:
        """Convert sensitivity value (0-100) to appropriate parameter ranges."""
        normalized = sensitivity / 100
        return {
            'canny_low': int(80 + (1 - normalized) * 80),
            'canny_high': int(100 + (1 - normalized) * 100),
            'hough_threshold': int(20 + (1 - normalized) * 80),
            'min_line_length': int(10 + (1 - normalized) * 50),
            'max_line_gap': int(2 + normalized * 10)
        }

    def detect_walls(self, page_pixmap, sensitivity: float = 50):
        """Detect walls in the page pixmap using computer vision."""
        # Convert pixmap to numpy array
        img = np.frombuffer(page_pixmap.samples, dtype=np.uint8).reshape(
            page_pixmap.height, page_pixmap.width, page_pixmap.n
        )
        
        # Convert to grayscale if necessary
        if page_pixmap.n == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        elif page_pixmap.n == 3:  # RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        params = self.calculate_parameters(sensitivity)
        
        # Apply image processing
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        edges = cv2.Canny(thresh, params['canny_low'], params['canny_high'], 
                         apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=params['hough_threshold'],
            minLineLength=params['min_line_length'],
            maxLineGap=params['max_line_gap']
        )

        return lines if lines is not None else []

    def add_wall_annotation(self, page, start_point, end_point, wall_type="Wall"):
        """Add a wall annotation to the PDF page."""
        # Create line annotation
        annot = page.add_line_annot(start_point, end_point)
        
        # Set properties
        annot.set_border(width=self.line_width)
        annot.set_colors(stroke=self.stroke_color)
        
        # Set Bluebeam-compatible metadata
        annot.info.update({
            "Title": "Wall Markup",
            "Subject": wall_type,
            "Creator": "Wall Detector",
            "ModDate": datetime.now().strftime("%Y%m%d%H%M%S"),
        })
        
        # Set flags
        annot.set_flags(fitz.PDF_ANNOT_IS_PRINTED)
        annot.update()

    def process_pdf(self, pdf_file, page_number: int, sensitivity: float = 50):
        """Process a single page of the PDF and add wall annotations."""
        # Open PDF
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        page = pdf_document[page_number]
        
        # Get page dimensions
        page_rect = page.rect
        
        # Create pixmap for analysis
        pix = page.get_pixmap()
        
        # Detect walls
        detected_lines = self.detect_walls(pix, sensitivity)
        
        # Add wall annotations
        for line in detected_lines:
            x1, y1, x2, y2 = line[0]
            
            # Convert coordinates to PDF space
            pdf_x1 = x1 * page_rect.width / pix.width
            pdf_y1 = y1 * page_rect.height / pix.height
            pdf_x2 = x2 * page_rect.width / pix.width
            pdf_y2 = y2 * page_rect.height / pix.height
            
            # Add wall annotation
            self.add_wall_annotation(
                page,
                (pdf_x1, pdf_y1),
                (pdf_x2, pdf_y2),
                "Detected Wall"
            )
        
        # Save to memory
        output_bytes = io.BytesIO()
        pdf_document.save(output_bytes)
        pdf_document.close()
        
        return output_bytes.getvalue(), len(detected_lines)

def main():
    st.title("PDF Wall Detector")
    
    uploaded_file = st.file_uploader("Choose a PDF floor plan", type=["pdf"])
    
    if uploaded_file is not None:
        # Initialize PDF processor
        pdf_processor = WallDetectorPDF()
        
        # Load PDF for preview
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        
        # Page selection
        page_count = len(pdf_document)
        page_options = [f"Page {i+1}" for i in range(page_count)]
        selected_page = st.selectbox("Select a page to analyze", page_options)
        selected_index = int(selected_page.split()[1]) - 1
        
        # Display preview
        page = pdf_document[selected_index]
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        st.image(img, caption=f"Preview of {selected_page}", use_column_width=True)
        
        # Sensitivity slider
        sensitivity = st.slider("Wall Detection Sensitivity", 0, 100, 50)
        
        if st.button("Detect Walls"):
            with st.spinner("Processing..."):
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Process PDF
                processed_pdf, wall_count = pdf_processor.process_pdf(
                    uploaded_file,
                    selected_index,
                    sensitivity
                )
                
                # Show results
                st.write(f"Detected {wall_count} potential walls")
                
                # Provide download button
                st.download_button(
                    label="Download Processed PDF",
                    data=processed_pdf,
                    file_name="processed_floor_plan.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()