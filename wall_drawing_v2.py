import os
from pathlib import Path
import fitz  # PyMuPDF
import base64
from openai import OpenAI
import cv2
import numpy as np
import streamlit as st
import json
import io
from PIL import Image

def analyze_architectural_drawing(uploaded_file, edge_detection_params):
    # Initialize OpenAI client
    client = OpenAI(api_key="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")

    # Read the image
    file_bytes = uploaded_file.getvalue()
    file_bytes = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use OpenAI Vision API to identify and analyze walls
    base64_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

    # First, get general analysis and edge detection parameters from Vision API
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this architectural drawing and suggest optimal parameters for edge detection. Return a JSON object with the following structure: {'general_analysis': 'Your analysis here', 'edge_detection_params': {'canny_low': int, 'canny_high': int, 'hough_threshold': int, 'min_line_length': int, 'max_line_gap': int}}"
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ],
        max_tokens=1000
    )

    try:
        response_data = json.loads(response.choices[0].message.content)
        general_analysis = response_data['general_analysis']
        suggested_params = response_data['edge_detection_params']
    except json.JSONDecodeError:
        st.warning("Could not parse JSON response from Vision API")
        general_analysis = response.choices[0].message.content
        suggested_params = {
            'canny_low': 50,
            'canny_high': 150,
            'hough_threshold': 100,
            'min_line_length': 100,
            'max_line_gap': 10
        }

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

    return general_analysis, wall_coordinates, result_image, suggested_params

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
        if uploaded_file.type == "application/pdf":
            # Store the uploaded file
            st.session_state.uploaded_pdf = uploaded_file
            
            # Parse images from PDF
            images = pdf_to_images(st.session_state.uploaded_pdf)
            st.session_state.pdf_images = images
            st.write(f"PDF uploaded with {len(st.session_state.pdf_images)} pages")
            
            # Handle page selection
            selected_pages = []
            for i, img in enumerate(st.session_state.pdf_images):
                col1, col2 = st.columns([1, 4])
                with col1:
                    selected = st.checkbox(f"Page {i+1}", key=f"checkbox_{i}")
                    if selected:
                        selected_pages.append(i)
                with col2:
                    st.image(img, caption=f"Page {i+1}", width=200)
            
            # Store selected pages
            st.session_state.selected_pages = selected_pages
            
            if st.session_state.selected_pages:
                st.write(f"Selected pages: {', '.join(map(str, [p+1 for p in st.session_state.selected_pages]))}")
                if st.button("Confirm Page Selection"):
                    selected_image = st.session_state.pdf_images[st.session_state.selected_pages[0]]
                    img_byte_arr = io.BytesIO()
                    selected_image.save(img_byte_arr, format='PNG')
                    st.session_state.img_byte_arr = img_byte_arr.getvalue()
                else:
                    st.info("Please confirm your page selection before proceeding.")
                    return
            else:
                st.warning("Please select at least one page to analyze.")
                return
        else:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            img_byte_arr = uploaded_file.getvalue()
        
        # Initialize edge_detection_params with default values
        edge_detection_params = {
            'canny_low': 50,
            'canny_high': 150,
            'hough_threshold': 100,
            'min_line_length': 100,
            'max_line_gap': 10
        }
        
        if st.button("Analyze Drawing"):
            with st.spinner("Analyzing..."):
                analysis, walls, analyzed_image, suggested_params = analyze_architectural_drawing(io.BytesIO(img_byte_arr), edge_detection_params)
            
            st.subheader("General Analysis")
            st.write(analysis)
            
            st.subheader("Edge Detection Parameters")
            st.write("Adjust the parameters to see real-time changes in the analyzed drawing.")
            
            # Create sliders for each parameter
            canny_low = st.slider("Canny Low", 0, 255, suggested_params['canny_low'])
            canny_high = st.slider("Canny High", 0, 255, suggested_params['canny_high'])
            hough_threshold = st.slider("Hough Threshold", 0, 200, suggested_params['hough_threshold'])
            min_line_length = st.slider("Min Line Length", 0, 200, suggested_params['min_line_length'])
            max_line_gap = st.slider("Max Line Gap", 0, 50, suggested_params['max_line_gap'])
            
            # Update parameters with user adjustments
            edge_detection_params = {
                'canny_low': canny_low,
                'canny_high': canny_high,
                'hough_threshold': hough_threshold,
                'min_line_length': min_line_length,
                'max_line_gap': max_line_gap
            }
            
            # Reanalyze in real-time as sliders are adjusted
            _, walls, analyzed_image, _ = analyze_architectural_drawing(io.BytesIO(img_byte_arr), edge_detection_params)
            
            st.subheader("Detected Walls")
            st.write(f"Number of potential walls detected: {len(walls)}")
            
            st.subheader("Analyzed Drawing")
            st.image(analyzed_image, caption="Analyzed Drawing", use_column_width=True)

if __name__ == "__main__":
    main()
