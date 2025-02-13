import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
from datetime import datetime
import tempfile
import imageio
from pathlib import Path
import subprocess

class ObjectDetectionApp:
    def __init__(self):
        # Get the directory where yolo_app.py is located
        current_dir = Path(__file__).parent.absolute()
        
        # Construct absolute path to the weights file
        weights_path = current_dir / "runs" / "detect" / "train13" / "weights" / "epoch15.pt"
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {weights_path}")
            
        self.model = YOLO(str(weights_path))
        
        # Create processed_videos directory relative to the script location
        self.output_dir = current_dir / "processed_videos"
        os.makedirs(str(self.output_dir), exist_ok=True)
    
    def process_image(self, image):
        results = self.model.predict(image)
        return Image.fromarray(results[0].plot())
    
    def process_video(self, video_path, progress_placeholder, status_placeholder):
        # Get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Generate unique filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        input_filename = f'input_{timestamp}.mp4'
        input_path = str(self.output_dir / input_filename)
        
        # Copy input video
        with open(video_path, 'rb') as source_file:
            with open(input_path, 'wb') as target_file:
                target_file.write(source_file.read())
        
        # Create a generator for prediction to track progress
        results = self.model.predict(input_path, save=True, 
                                   project=str(self.output_dir),
                                   name=f'output_{timestamp}', 
                                   stream=True)
        
        # Track progress
        frame_count = 0
        for r in results:
            frame_count += 1
            progress = float(frame_count) / total_frames
            progress_placeholder.progress(progress)
            status_placeholder.text(f"Processing: video 1/1 (frame {frame_count}/{total_frames})")
        
        output_path = str(self.output_dir / f'output_{timestamp}.mp4')
        return output_path

def get_absolute_path(relative_path):
    """Convert relative path to absolute path based on current file location"""
    current_dir = Path(__file__).parent.parent  # Go up one level to main project directory
    return str(current_dir / relative_path)

def main():
    # Configuration de la page
    st.set_page_config(
        page_title="Object Detection App",
        page_icon="🔍",
        layout="wide"
    )
    
    # Barre de navigation
    st.sidebar.title("Navigation")
    
    # Bouton pour retourner à la page d'accueil
    if st.sidebar.button("🏠 Return to Home"):
        home_path = get_absolute_path("home.py")
        if os.path.exists(home_path):
            current_dir = os.getcwd()
            os.chdir(os.path.dirname(home_path))
            subprocess.Popen(["streamlit", "run", home_path])
            os.chdir(current_dir)
        else:
            st.sidebar.error(f"Path not found: {home_path}")
    
    # Bouton pour aller à la classification
    if st.sidebar.button("🌱 Classification & Nutrition Guide"):
        app_path = get_absolute_path("agricultural_classification/app/app.py")
        if os.path.exists(app_path):
            current_dir = os.getcwd()
            os.chdir(os.path.dirname(app_path))
            subprocess.Popen(["streamlit", "run", app_path])
            os.chdir(current_dir)
        else:
            st.sidebar.error(f"Path not found: {app_path}")
    
    # Séparateur dans la barre latérale
    st.sidebar.markdown("---")
    
    # Contenu principal
    st.title("🔍 Object Detection App")
    st.write("Detect Apples, Oranges, and Strawberries in images and videos")
    
    app = ObjectDetectionApp()
    
    tab1, tab2 = st.tabs(["Image Detection", "Video Detection"])
    
    with tab1:
        st.header("Image Detection")
        uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.subheader("Original Image")
            st.image(image)
            
            if st.button("Detect Objects in Image"):
                st.subheader("Processed Image")
                processed_image = app.process_image(image)
                st.image(processed_image)
    
    with tab2:
        st.header("Video Detection")
        uploaded_video = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])
        
        if uploaded_video is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.getvalue())
                video_path = tmp_file.name
            
            st.video(uploaded_video)
            
            if st.button("Detect Objects in Video"):
                # Create placeholders for progress bar and status
                progress_placeholder = st.progress(0)
                status_placeholder = st.empty()
                
                with st.spinner("Processing video..."):
                    output_path = app.process_video(video_path, progress_placeholder, status_placeholder)
                    progress_placeholder.progress(1.0)  # Ensure progress bar shows completion
                    status_placeholder.text("Processing complete!")
                    st.success(f"Video processed! You can find it at: {output_path}")
                
                os.unlink(video_path)

if __name__ == "__main__":
    main()