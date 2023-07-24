# Tutorial on creating a custom object detector using Pytorch from scratch
This is a tutorial I learn on how to create a custom object detector using Pytorch from scratch using pretrained Faster RCNN

This repo follows the tutorial from this [link](https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/)

In this tutorial, we are going to detect the following object class:
1. Arduino_Nano
2. ESP8266
3. Raspberry_Pi_3
4. Heltec_ESP32_Lora

## Main programming language
- Python

## Tutorial outlier
1. configure software
2. project structure


## Configure software
1. Configure the necessary software for developing the project

:fire: Recommend to use the virtual environment (environment manager+package manager), so that the package you use will not disturb the pacakge version on base machine. Changes on package version may change the code's functions and usages, becareful on that.

- Python 3.8
- miniconda python 3.8
- CUDA toolkit 12.0
- torch 2.0.1
- torchvision 2.0.2
- opencv-contrib-python 4.8.0.74
- imutils 0.5.4
- tqdm 4.65.0

## Project structure
