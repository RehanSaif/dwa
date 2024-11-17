import os
from pathlib import Path
import base64
from openai import OpenAI
import cv2
import numpy as np

def analyze_architectural_drawing(png_path):
    # Initialize OpenAI client
    client = OpenAI(api_key="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")

    # Read the image
    image = cv2.imread(png_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use OpenAI Vision API for general analysis
    with open(png_path, 'rb') as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this architectural drawing and identify the locations of walls."
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

    general_analysis = response.choices[0].message.content

    # Use OpenCV for wall detection
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Enhance edges using adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Detect edges with less sensitive parameters
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    
    # Dilate edges to connect nearby lines, but with smaller kernel
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Detect lines with stricter parameters
    lines = cv2.HoughLinesP(
        dilated,
        rho=1,
        theta=np.pi/180,
        threshold=100,  # Higher threshold to detect fewer lines
        minLineLength=100,  # Longer minimum line length
        maxLineGap=10  # Smaller max gap to avoid connecting non-wall lines
    )

    result_image = image.copy()
    wall_coordinates = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            wall_coordinates.append([x1, y1, x2, y2])
            cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save the result
    cv2.imwrite('analyzed_drawing.jpg', result_image)

    return general_analysis, wall_coordinates, 'analyzed_drawing.jpg'

# Example usage
if __name__ == "__main__":
    png_path = "C:/Users/20176677/Documents/DWA/bouwtekening.png"
    analysis, walls, analyzed_image = analyze_architectural_drawing(png_path)
    print(f"General analysis: {analysis}")
    print(f"Detected {len(walls)} potential walls")
    print(f"Analysis complete. Result saved as {analyzed_image}")
