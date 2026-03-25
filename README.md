# 🚢 SHIP DETECTED AT SEA FROM SATELLITE IMAGES

## 📌 Project Overview
This repository contains the source code and research for the final project: **"Phát hiện tàu thuyền trên biển từ ảnh vệ tinh" (Ship Detection at Sea from Satellite Images)**. 

Detecting ships in high-resolution optical satellite images is a significant challenge due to the small size of the ships, complex backgrounds (such as waves, clouds, fog, and port infrastructure), and varying ship orientations. The primary goal of this project is to build and evaluate an Object Detection system capable of accurately locating and counting ships in these complex maritime environments.

## 📊 Dataset
The project utilizes the **SCCOS (Ship Collection in Complex Optical Scene)** dataset. This dataset includes high-resolution optical satellite images collected from various sources (Google Earth, Microsoft map, Worldview-3, etc.), featuring diverse environmental conditions and complex scenes to reflect real-world scenarios.

## 🤖 Models Evaluated
To handle the varying scales and arbitrary orientations of the ships, we trained and benchmarked 5 different state-of-the-art detection architectures:
* **YOLOv8** (One-stage)
* **Oriented R-CNN** (Two-stage, Oriented Bounding Box)
* **R3Det** (One-stage, Refined Rotated RetinaNet)
* **Faster R-CNN** (Two-stage, Horizontal Bounding Box)
* **Mask R-CNN** (Instance Segmentation)

## 💻 Application (OceanEye)
We also developed a web application named **OceanEye** to demonstrate the models in action. Key features include:
* **Image Analysis**: Detailed detection on satellite imagery.
* **Benchmarking**: Side-by-side performance comparison of multiple models on the same image.
* **Video Surveillance**: Real-time ship detection on video streams.

