# Onboard Data Prioritization Using Multi-Class Image Segmentation for Nanosatellites
[![GitHub](https://img.shields.io/github/license/keenanch8ar/CubeSegNet)](https://github.com/keenanch8ar/CubeSegNet/LICENSE) [![DOI](https://img.shields.io/badge/DOI-10.3390/rs16101729-blue)](https://doi.org/10.3390/rs16101729)

![Graphical Abstract](https://i.imgur.com/DJVgCoK.png)

# Abstract

Nanosatellites are proliferating as low-cost, dedicated remote sensing opportunities for small nations. However, nanosatellite performance as remote sensing platforms is impaired by low downlink speeds typically ranging between 1200-9600bps. Additionally, an estimated 67\% of downloaded data is unusable for further applications due to excess cloud cover. To alleviate this issue, we propose an image segmentation and prioritization algorithm to classify and segment the contents of captured images onboard the nanosatellite. This algorithm prioritizes images with clear captures of water bodies and vegetated areas with high downlink priority. This in-orbit organization of images will aid ground station operators with downlinking images suitable for further ground-based remote sensing analysis. The proposed algorithm uses Convolutional Neural Network (CNN) models to classify and segment captured image data. In this study, we compare various model architectures and backbone designs for segmentation and assess their performance. The models are trained on a dataset that simulates captured data from nanosatellites and transferred to the satellite hardware to conduct inferences. Ground testing for the satellite has achieved a peak Mean IoU of 75\%  and an F1 Score of 0.85 for multi-class segmentation. The proposed algorithm is expected to improve data budget downlink efficiency by up to 42\% based on validation testing.

## Overview

This repository contains the implementation of CubeSegNet, a variant of SegNet tailored for CubeSat image processing. The model is designed to run onboard CubeSats, enabling real-time image analysis and data prioritization.

## Features

- **CubeSat Image Processing:** Process images captured by CubeSats to extract meaningful information.
- **Semantic Segmentation:** Classify each pixel in an image to identify objects or regions of interest.
- **Onboard Data Prioritization:** Use machine learning to prioritize image data for efficient downlink.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/username/CubeSegNet.git
cd CubeSegNet
pip install -r requirements.txt
```
## How to Use

To train the CubeSegNet model, follow these steps:

1. Make sure you have the training data prepared in the working directory.
2. Run the `train.py` script with the necessary arguments:

 ```bash
python train.py --architecture Unet --backbone efficientnetb0 --epochs 30
```
3. Run the `test.py` script with the necessary arguments:

 ```bash
python test.py --model_dir saved_models/Unet_efficientnetb0_20_03_24/my_seg_model
```
