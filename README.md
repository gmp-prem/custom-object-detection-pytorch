# Tutorial on creating a custom object detector using Pytorch from scratch
This is a tutorial I learn on how to create a custom object detector using Pytorch from scratch using pretrained Faster RCNN

This repo follows the tutorial from this [link](https://debuggercafe.com/custom-object-detection-using-pytorch-faster-rcnn/)

**:fire: Recommended source** for studying the deep learning, this includes the tutorial and the source code, so, you can use as a your project booster: [debuggercafe](https://debuggercafe.com/)

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

You can clone this repository and will have the following the structure
```
CUSTOM-OBJECT-DETECTION (project root)
    ├── Microcontroller_Detection
    │   ├── test
    |   |   ├── IMG_20181228_102636.jpg
    |   |   ├── IMG_20181228_102636.xml
    |   |   └── ...
    │   ├── train
    │   ├── test_labels.csv
    │   └── train_labels.csv
    ├── outputs
    │   ├── model100.pth
    │   ...
    │   └── valid_loss_98.png
    ├── src
    │   ├── config.py
    │   ├── datasets.py
    │   ├── engine.py
    │   ├── inference.py
    │   ├── model.py
    │   └── utils.py
    ├── test_data
    │   ├── arduino_nano.jpg
    │   ...
    ├── test_predictions
    │   ├── arduino_nano.jpg
    │   ...
    ...
```
in the project folder will generally contain 5 folders, in each folder

`Microcontroller_Detection` this is basically the dataset folder, this folder will contain two sub-folders which are `train`: dataset for training the model and `test`: dataset for testing the model. In the `test` folder, you can see the file JPG and xml that corresponds to the JPG file, and the csv file format in dataset folder contain same information as in xml file but in the tabular format. dataset format in these two sub-folders will be same except they are used in different way.
- in the XML file will mainly contain
    - image height and weight
    - class name
    - bounding box coordinates

`src` folder: this folder contains the Python codes that are used in different uses.

`outputs` this folder contains the output of the trained model which is called `checkpoint file` or in pth format and the loss graph that is generated from the model we train

`test_data` this folder contains images that we will use for inference after training
- **inference ?** : inferencing in deep learning is when you apply the knowledge of the deep learning model that you have trained to infer the result, it is like when the new unknown data (unseen data) is input into your trained nueral network, your network could predict the output in a good accuracy.

`test_prediction` this folder contains the inference image results.