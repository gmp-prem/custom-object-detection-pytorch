import torch

BATCH_SIZE = 4 # increase or decrease according to GPU memory
RESIZE_TO = 512 # resize the image for training and transform
NUM_EPOCHS = 100 # number of epochs to train 

CUDA_DEVICE = "cuda"
CPU_DVICE = "cpu"

# choose cuda if available (train using GPU) else using CPU (will be much slower)
DEVICE = torch.device(CUDA_DEVICE) if torch.cuda.is_available() else torch.device(CPU_DVICE)

TRAIN_DIR = "../Microcontroller_Detection/train"
VALID_DIR = "../Microcontroller_Detection/test"

# this contains 4 classes from CLASSES but plus one because pytorch faster rcnn expects
# additional class along with the class of dataset; that is background, reserved at index 0
CLASSES = [
    "background", "Arduino_Nano", "ESP8266", "Raspberry_Pi_3", "Heltec_ESP32_Lora"
]
NUM_CLASSESS = 4 + 1

# show the augmented images after augmentation before training
VISUALIZE_TRANSFORMED_IMAGES = False

# directory of output of training model
OUT_DIR = "../outputs"
# interval save for plotting and checkpoint file
SAVE_PLOTS_EPOCH = 2
SAVE_MODEL_EPOCH = 2