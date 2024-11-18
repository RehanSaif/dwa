import os
from pathlib import Path
import fitz  # PyMuPDF
import cv2
import numpy as np
import streamlit as st
import io
from PIL import Image

def analyze_architectural_drawing(uploaded_file, edge_detection_params):
    # Read the image
    file_bytes = uploaded_file.getvalue()
    file_bytes = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    wall_coordinates = []  # Initialize empty list for wall coordinates
    
    # Apply blur for edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Enhance edges using adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Detect edges with user-adjusted parameters
    edges = cv2.Canny(thresh, edge_detection_params['canny_low'], edge_detection_params['canny_high'], apertureSize=3)
    
    # Dilate edges to connect nearby lines, but with smaller kernel
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Detect lines with user-adjusted parameters
    lines = cv2.HoughLinesP(
        dilated,
        rho=1,
        theta=np.pi/180,
        threshold=edge_detection_params['hough_threshold'],
        minLineLength=edge_detection_params['min_line_length'],
        maxLineGap=edge_detection_params['max_line_gap']
    )

    result_image = image.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            wall_coordinates.append([x1, y1, x2, y2])
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return "Analysis completed", wall_coordinates, result_image, edge_detection_params

def pdf_to_images(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    images = []
    for page in pdf_document:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

def main():
    st.title("Architectural Drawing Analyzer")

    uploaded_file = st.file_uploader("Choose an architectural drawing image or PDF", type=["png", "jpg", "jpeg", "pdf"])
    
    if uploaded_file is not None:
        print("File uploaded successfully")
        if uploaded_file.type == "application/pdf":
            print("Uploaded file is a PDF")
            # Check if PDF has already been parsed
            if 'pdf_images' not in st.session_state or st.session_state.uploaded_pdf != uploaded_file:
                print("Parsing PDF images")
                # Parse images from PDF only once or if a new PDF is uploaded
                st.session_state.pdf_images = pdf_to_images(uploaded_file)
                st.session_state.uploaded_pdf = uploaded_file
                st.session_state.page_confirmed = False
                print(f"Parsed {len(st.session_state.pdf_images)} images from PDF")
                        
            # Handle page selection
            if 'pdf_images' in st.session_state:
                st.write(f"PDF uploaded with {len(st.session_state.pdf_images)} pages")
                
                # Create a selectbox for page selection
                page_options = [f"Page {i+1}" for i in range(len(st.session_state.pdf_images))]
                selected_page = st.selectbox("Select a page to analyze", page_options, key='page_selector')
                
                # Display the selected page
                selected_index = int(selected_page.split()[1]) - 1
                st.image(st.session_state.pdf_images[selected_index], caption=f"Selected {selected_page}", use_column_width=True)
            
            if not st.session_state.page_confirmed:
                if st.button("Confirm Page Selection"):
                    print(f"Page {selected_index + 1} selected and confirmed")
                    selected_image = st.session_state.pdf_images[selected_index]
                    img_byte_arr = io.BytesIO()
                    selected_image.save(img_byte_arr, format='PNG')
                    st.session_state.img_byte_arr = img_byte_arr.getvalue()
                    print("Image converted to bytes and stored in session state")
                    st.session_state.page_confirmed = True
                    st.session_state.edge_detection_params = None  # Reset edge detection params for new image
                else:
                    st.info("Please confirm your page selection before proceeding.")
                    print("Waiting for page selection confirmation")
                    return
        else:
            print("Uploaded file is an image")
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            if 'img_byte_arr' not in st.session_state or st.session_state.img_byte_arr != uploaded_file.getvalue():
                st.session_state.img_byte_arr = uploaded_file.getvalue()
                st.session_state.edge_detection_params = None  # Reset edge detection params for new image
            print("Image stored in session state")
            st.session_state.page_confirmed = True
        
        # Initialize or retrieve edge_detection_params
        if 'edge_detection_params' not in st.session_state or st.session_state.edge_detection_params is None:
            st.session_state.edge_detection_params = {
                'canny_low': 50,
                'canny_high': 150,
                'hough_threshold': 100,
                'min_line_length': 100,
                'max_line_gap': 10
            }
        
        st.subheader("Edge Detection Parameters")
        st.write("Adjust the parameters before analyzing the drawing.")
        
        # Create sliders for each parameter
        canny_low = st.slider("Canny Low", 0, 255, st.session_state.edge_detection_params['canny_low'])
        canny_high = st.slider("Canny High", 0, 255, st.session_state.edge_detection_params['canny_high'])
        hough_threshold = st.slider("Hough Threshold", 0, 200, st.session_state.edge_detection_params['hough_threshold'])
        min_line_length = st.slider("Min Line Length", 0, 200, st.session_state.edge_detection_params['min_line_length'])
        max_line_gap = st.slider("Max Line Gap", 0, 50, st.session_state.edge_detection_params['max_line_gap'])
        
        # Update parameters with user adjustments
        st.session_state.edge_detection_params = {
            'canny_low': canny_low,
            'canny_high': canny_high,
            'hough_threshold': hough_threshold,
            'min_line_length': min_line_length,
            'max_line_gap': max_line_gap
        }
        
        if st.button("Analyze Drawing") and st.session_state.page_confirmed:
            with st.spinner("Analyzing..."):
                analysis, walls, analyzed_image, _ = analyze_architectural_drawing(io.BytesIO(st.session_state.img_byte_arr), st.session_state.edge_detection_params)
            
            st.subheader("General Analysis")
            st.write(analysis)
            
            st.subheader("Detected Walls")
            st.write(f"Number of potential walls detected: {len(walls)}")
            
            st.subheader("Analyzed Drawing")
            st.image(analyzed_image, caption="Analyzed Drawing", use_column_width=True)

if __name__ == "__main__":
    main()
