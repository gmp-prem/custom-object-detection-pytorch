# Tutorial on creating a custom object detector using Pytorch from scratch
This is a tutorial I learn on how to create a custom object detector using Pytorch from scratch

This repo follows the tutorial from this [link](https://pyimagesearch.com/2021/11/01/training-an-object-detector-from-scratch-in-pytorch/)

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
- CUDA toolkit 11.7
- PyTorch
- opencv-contrib-python

## Project structure
This is the structure of project we are going to develop, slightly changes from the original tutorial
```
project_root
    ├── dataset
    |   ├── dataset
    ├── output
    │   ├── detector.pth
    │   ├── le.pickle
    │   ├── plots
    │   │   └── training.png
    │   └── test_paths.txt
    ├── predict.py
    ├── pyimagesearch
    │   ├── bbox_regressor.py
    │   ├── config.py
    │   ├── custom_tensor_dataset.py
    │   └── __init__.py
    └── train.py
```