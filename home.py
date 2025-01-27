import streamlit as st
import os
import subprocess
from pathlib import Path

def get_absolute_path(relative_path):
    current_dir = Path(__file__).parent.absolute()
    return str(current_dir / relative_path)

def styled_title(text, color="#2E86C1"):
    st.markdown(f"""
        <h1 style='color: {color};'>{text}</h1>
    """, unsafe_allow_html=True)

def styled_header(text, color="#2874A6"):
    st.markdown(f"""
        <h2 style='color: {color};'>{text}</h2>
    """, unsafe_allow_html=True)

def main():
    # Page config
    st.set_page_config(page_title="Agricultural Products Analysis", page_icon="🏠", layout="wide")
    
    # Set yellow background for entire page
    st.markdown("""
        <style>
        .stApp {
            background-color: #FFF9C4;
        }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("Navigation")

    if st.sidebar.button("🔄 Classification & Nutrition Guide"):
        app_path = get_absolute_path("agricultural_classification/app/app.py")
        if os.path.exists(app_path):
            subprocess.Popen(["streamlit", "run", app_path])
        else:
            st.sidebar.error(f"Chemin non trouvé: {app_path}")

    if st.sidebar.button("🔍 Object Detection (YOLO)"):
        yolo_path = get_absolute_path("Object-Detection-Yolo/yolo_app.py")
        if os.path.exists(yolo_path):
            yolo_dir = os.path.dirname(yolo_path)
            current_dir = os.getcwd()
            os.chdir(yolo_dir)
            subprocess.Popen(["streamlit", "run", yolo_path])
            os.chdir(current_dir)
        else:
            st.sidebar.error(f"Chemin non trouvé: {yolo_path}")

    styled_title("🌱 Agricultural Products Analysis Platform")

    col1, col2 = st.columns([2, 1])

    with col1:
        styled_header("The Importance of Fruits and Vegetables")
        
        st.write("""
        Fruits and vegetables are essential components of a healthy diet and are major contributors to human nutrition and well-being. They provide vital nutrients including vitamins, minerals, dietary fiber, and various bioactive compounds that promote health and help prevent diseases.

        ### Key Benefits:
        - **Rich in Nutrients**: Packed with vitamins, minerals, and antioxidants
        - **Disease Prevention**: Help reduce the risk of chronic diseases
        - **Dietary Fiber**: Support digestive health and maintain healthy weight
        - **Low in Calories**: Perfect for maintaining a healthy diet
        - **Hydration**: Many fruits and vegetables have high water content

        ### Our Tools:
        1. **Classification & Nutrition Guide**
           - Accurate classification using multiple models (VGG16, ResNet, EfficientNet)
           - Detailed nutritional information
           - Comprehensive food guidance
                 
        2. **Object Detection (YOLO)**
           - Real-time detection of fruits
           - Advanced computer vision technology
           - Supports both images and videos
        """)

    with col2:
        st.info("""
        ### Did You Know?
        - The color of fruits and vegetables often indicates their specific nutrient content
        - Eating a variety of colored produce ensures a wide range of nutrients
        - Fresh, frozen, and canned vegetables all count toward your daily intake
        """)

        st.warning("""
        ### Quick Tip
        Try to eat at least 5 portions of different fruits and vegetables every day for optimal health benefits!
        """)

    styled_header("How to Use Our Tools")
    st.write("""
    1. **For Classification and Nutrition:**
       - Choose "Classification & Nutrition Guide" from the sidebar
       - Upload your image
       - Receive classification results and nutritional information
             
    2. **For Object Detection:**
       - Select "Object Detection (YOLO)" from the sidebar
       - Upload images or videos
       - Get instant detection results
    """)
    
    st.markdown("---")
    styled_header("Copyright Notice")
    st.markdown("""
    This Agricultural Products Analysis Platform was developed by:
    - Yasser Salhi
    - Aymane Souiles
    - Rim Taouab
    - Khalil Berraho
    - Ahmed Akestaf

    Under the supervision of our Professor Aouatif Amine
    
    © 2025 All Rights Reserved
    """)

if __name__ == "__main__":
    main()