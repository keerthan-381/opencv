import cv2
import pytesseract
from PIL import Image
import streamlit as st
import numpy as np
import os
from io import BytesIO

# Configure pytesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\e430388\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Load the face detection cascade
face_cascade = cv2.CascadeClassifier(r'C:\Users\e430388\Downloads\Vs Code\cv\cascade_frontface_default.xml')

def extract_text_from_image(image):
    """Extract text from image using pytesseract."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(Image.fromarray(threshold), config='--psm 11')
    return text.strip()

def detect_faces_in_image(image):
    """Detect faces in an image."""
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
    face_count = len(faces)
    for x, y, w, h in faces:
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, f"Faces: {face_count}", (10, 30), font, 0.5, (255, 0, 0), 1)
    return image, face_count

def detect_faces_in_video(video_file):
    """Detect faces in video and save the output."""
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        st.error("Error: Could not open video stream.")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_video.avi', fourcc, fps, (frame_width, frame_height))
    
    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
        face_count = len(faces)
        
        for x, y, w, h in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"Faces: {face_count}", (10, 30), font, 0.5, (255, 0, 0), 1)
        out.write(img)
    
    cap.release()
    out.release()
    return 'output_video.avi', face_count

# Helper function to convert image to byte format for download
def get_image_download_link(img, filename="output_image.png"):
    is_success, buffer = cv2.imencode(".png", img)
    byte_image = buffer.tobytes()
    return st.download_button(
        label="Download Image",
        data=byte_image,
        file_name=filename,
        mime="image/png"
    )

def get_video_download_link(video_path, filename="output_video.avi"):
    """Provides a download link for the video."""
    with open(video_path, "rb") as file:
        video_bytes = file.read()
    return st.download_button(
        label="Download Video",
        data=video_bytes,
        file_name=filename,
        mime="video/avi"
    )

st.title("Face Detection & Text Extraction App")

option = st.radio("Choose an option", ["Extract Text from Image", "Detect Faces in Image", "Detect Faces in Video"])

if option == "Extract Text from Image":
    st.header("Extract Text from Image")
    uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_image is not None:
        image = np.array(Image.open(uploaded_image))
        text = extract_text_from_image(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.subheader("Extracted Text:")
        st.write(text)
        # Add download option for the text
        st.download_button(
            label="Download Text",
            data=text,
            file_name="extracted_text.txt",
            mime="text/plain"
        )

# Handle Detect Faces in Image
elif option == "Detect Faces in Image":
    st.header("Detect Faces in Image")
    uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_image is not None:
        image = np.array(Image.open(uploaded_image))
        image_with_faces, face_count = detect_faces_in_image(image)
        st.image(image_with_faces, caption="Faces Detected", use_column_width=True)
        st.write(f"Number of faces detected: {face_count}")
        # Add download option for the image with faces
        get_image_download_link(image_with_faces)

# Handle Detect Faces in Video
elif option == "Detect Faces in Video":
    st.header("Detect Faces in Video")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        video_path = uploaded_video.name
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        output_video, face_count = detect_faces_in_video(video_path)
        
        # Show video after detection
        st.video(output_video)
        
        st.write(f"Number of faces detected: {face_count}")
        
        # Add download option for the video
        get_video_download_link(output_video)

# Display the app
if __name__ == "__main__":
    st.write("Select an option and upload your file.")
