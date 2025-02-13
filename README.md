# Agricultural Products Analysis Platform 🌱

Welcome to the **Agricultural Products Analysis Platform**! This platform is designed to assist individuals and organizations in analyzing and classifying agricultural products, including fruits and vegetables, through modern machine learning and computer vision technologies. With two primary features, **Classification & Nutrition Guide** and **Object Detection (YOLO)**, users can seamlessly classify agricultural items and retrieve detailed nutritional information or perform object detection on images and videos.

## Table of Contents 📚
- [Introduction](#introduction)
- [Features](#features)
  - [Classification & Nutrition Guide](#classification--nutrition-guide)
  - [Object Detection (YOLO)](#object-detection-yolo)
- [Technologies Used](#technologies-used)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Step-by-step Installation](#step-by-step-installation)
- [How to Use](#how-to-use)
  - [Classification & Nutrition Guide](#classification--nutrition-guide-usage)
  - [Object Detection (YOLO) Usage](#object-detection-yolo-usage)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Introduction 👨‍🌾👩‍🌾

**Agricultural Products Analysis Platform** is a robust web application designed using **Streamlit**, focusing on two core functionalities: **classification** and **object detection** of agricultural products. Whether you are a researcher, a dietitian, or just someone passionate about healthy eating, this platform provides useful tools to help you classify fruits and vegetables, as well as understand their nutritional values in depth.

With built-in support for **YOLO object detection** (You Only Look Once), this tool can detect and classify agricultural products from both images and videos in real-time.

## Features ✨

### Classification & Nutrition Guide 🥦🍓
This feature leverages powerful deep learning models like **VGG16**, **ResNet**, and **EfficientNet** to classify agricultural products. Once the product is classified, the platform provides the following benefits:
- **Accurate Classification**: The tool identifies the agricultural product, from fruits to vegetables, ensuring precise results.
- **Detailed Nutritional Information**: After classification, users can view a comprehensive nutritional profile, which includes important nutrients like vitamins, minerals, fiber, and antioxidants.
- **Healthy Eating Recommendations**: Based on the classification, users can receive recommendations for healthy consumption and a balanced diet.

### Object Detection (YOLO) 🍏🍊
Powered by **YOLO (You Only Look Once)**, the **Object Detection** feature allows the platform to detect and locate fruits and vegetables in images or videos. This real-time detection feature offers:
- **Real-time Object Detection**: Upload images or videos, and the platform will detect and highlight agricultural products instantly.
- **Supports Images & Videos**: It works with both static images and dynamic video files, allowing for a versatile analysis of agricultural products.
- **Fast Processing**: Thanks to YOLO's advanced architecture, detection results are provided in real-time, ensuring a fast user experience.

---

## Technologies Used ⚙️

The Agricultural Products Analysis Platform is built using cutting-edge technologies to ensure performance, scalability, and ease of use. Here's an overview of the technologies used:

- **Streamlit**: A powerful Python framework for building interactive data applications. It’s fast, easy to use, and allows for quick development of machine learning applications.
- **YOLO (You Only Look Once)**: A state-of-the-art object detection algorithm that identifies objects in images and videos with high accuracy and speed.
- **OpenCV**: A computer vision library used for image and video processing, enabling the platform to handle video inputs seamlessly.
- **Pre-trained Deep Learning Models**: VGG16, ResNet, and EfficientNet models are used for image classification, all trained on large datasets to provide accurate predictions.

---

## Installation & Setup 🚀

To get started with this project, follow these steps to set up the platform on your local machine.

### Prerequisites 🛠️

Before you begin, ensure that you have the following installed:

- **Python 3.7 or higher**: The application is built using Python. Make sure you have the latest version installed. You can check your version with:
  ```bash
  python --version
