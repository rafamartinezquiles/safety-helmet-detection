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

1. Retrieve the Safety Helmet Detection Dataset in YOLOv11 format from the provided [link](https://universe.roboflow.com/zayed-uddin-chowdhury-ghymx/safety-helmet-wearing-dataset/dataset/3/download/yolov11). Download it as a zip file and ensure to place it within the main folder of the cloned repository named safety-helmet-detection.

```bash
mv /path/to/source /path/to/destination
```

2. Inside the cloned repository, execute the following command in order to unzip the BDBD dataset necessary for the project elaboration.

```bash
unzip Safety-Helmet-Wearing-Dataset.v3-base-dataset.yolov11.zip -d Safety-Helmet-Detection
```

3. One notable aspect of YOLO is its dependency on a .yaml file to delineate the paths for both training data (images and labels) and testing, as well as the classes to be identified. As we downloaded our dataset in YOLOv11 format, this already comes done by default and well-structured.

## Training of neural networks
The training of the neural network will be accomplished by executing the train.py file, passing a series of arguments that define the characteristics of the neural network. It's important to note that the training process entails just one phase, training the network responsible for detecting the helmets worn by each person. The arguments to be specified are:

- **data:** This parameter represents the path leading to the .yaml file associated with each dataset.
- **imgsz:** Refers to the image size utilized during training.
- **epochs:** Denotes the number of training epochs. The inclusion of the early stopping attribute allows for the termination of training if the model fails to demonstrate improvement after a specified number of epochs.
- **batch:** Specifies the batch size utilized during training.
- **name:** Represents the name assigned to the neural network.
- **model_size:** This parameter offers a selection of options ('n', 's', 'm', 'l', 'x') corresponding to different versions of YOLOv11 that can be trained.

```bash
python /complete/path/to/src/train.py --data /complete/path/to/Safety-Helmet-Detection/data.yaml --imgsz 640 --epochs 400 --batch 32 --name safety_helmet_yolov11s --model_size s
```

In case of not having the necessary time or resources to train the neural networks, the weights of a neural networks to try it is provided!

## Data Details
- [Safety Helmet Detection](https://universe.roboflow.com/zayed-uddin-chowdhury-ghymx/safety-helmet-wearing-dataset/dataset/3).

## References
- OpenCV: https://opencv.org/
- Ultralytics: https://github.com/ultralytics/ultralytics
- Pytorch: https://pytorch.org/