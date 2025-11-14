# Automated Detection of Safety Helmet Compliance Using Deep Learning
This project applies modern computer vision techniques to identify whether individuals are wearing safety helmets in real-world scenes. Leveraging frameworks such as MXNet, GluonCV, and OpenCV, it provides a complete pipeline for detecting both helmet-wearing and non-helmet-wearing heads across diverse environments. Pretrained models and a curated dataset enable fast experimentation and reliable inference for industrial safety applications.
![](images/results_accuracy.png)


## Overview and Background
Identifying whether individuals are wearing safety helmets in complex, real-world scenes requires both precise object localization and reliable classification of headgear. This involves detecting the head region of a person, determining the presence or absence of a helmet, and handling challenges such as varying lighting conditions, diverse poses, and crowded industrial settings. Building on established research and practical datasets, this project applies deep-learning–based object detectors to accurately recognize helmet compliance in static images.

This repository investigates modern Convolutional Neural Network (CNN) architectures—particularly leveraging MXNet, GluonCV, and YOLO-based detection modules—to differentiate between “helmet” and “no-helmet” cases within natural scenes. Using publicly available and meticulously labeled datasets, the system achieves strong performance in both accuracy and inference speed, making it suitable for safety-monitoring applications in industrial, construction, and surveillance environments.
![](images/results_accuracy.png)

## Table of Contents
```
BibObjectDetection
|__ images
|   |__ results_accuracy.png 
|   |__ threshold.jpg
|   |__ yolo_application.png
|__ weights
|   |__ BDBD
|   |   |__ yolov8l.pt
|   |   |__ yolov8m.pt 
|   |   |__ yolov8s.pt 
|   |   |__ yolov8n.pt 
|   |__ People
|   |   |__ yolov8l.pt
|   |   |__ yolov8m.pt 
|   |   |__ yolov8s.pt 
|   |   |__ yolov8n.pt 
|   |__ SVHN
|   |   |__ yolov8l.pt
|   |   |__ yolov8m.pt 
|   |   |__ yolov8s.pt 
|   |   |__ yolov8n.pt 
|__ labels
|   |__ labels_test
|   |   |__ all the labels in txt format
|   |__ labels_train
|   |   |__ all the labels in txt format
|__ src
    |__ create_csv.py
    |__ create_yaml.py
    |__ data_augmentation.py
    |__ image_prediction.py
    |__ move_png_files.py
    |__ train.py
    |__ video_prediction.py
README.md
requirements.txt
```

## Getting started

### Resources used
A high-performance Acer Nitro 5 laptop, powered by an Intel Core i7 processor and an NVIDIA GeForce GTX 1650 GPU (4 GB VRAM), was used for model training and evaluation. Due to the large size of the dataset, the training process was computationally demanding and prolonged. Nevertheless, this hardware configuration provided a stable and efficient environment, enabling consistent experimentation and reliable validation of the gesture-recognition models.

### Installing
The project is deployed in a local machine, so you need to install the next software and dependencies to start working:

1. Create and activate the new virtual environment for the project

```bash
conda create --name helmet_detection python=3.11
conda activate helmet_detection
```

2. Clone repository

```bash
git clone https://github.com/rafamartinezquiles/safety-helmet-detection.git
```

3. In the same folder that the requirements are, install the necessary requirements

```bash
cd safety-helmet-detection
pip install -r requirements.txt
```

4. In addition to the existing requirements, PyTorch needs to be installed separately. Due to its dependence on various computational specifications, it's essential for each user to install it individually by following the provided link. [PyTorch](https://pytorch.org/). By default, training is conducted on the GPU. If a GPU is unavailable, we switch to CPU training, which, though slower, still allows for model training.

### Setup
It is worth noting that the "safety helmet detection" provides functionality to download it in YOLOv11 format, which is recommended. 


## Data Details
- [Safety Helmet Detection](https://universe.roboflow.com/zayed-uddin-chowdhury-ghymx/safety-helmet-wearing-dataset/dataset/3).

## References
- OpenCV: https://opencv.org/
- Ultralytics: https://github.com/ultralytics/ultralytics
- Pytorch: https://pytorch.org/