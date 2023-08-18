import torch
import cv2
import numpy as np
import os
import glob as glob

from xml.etree import ElementTree as et
from config import CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, get_train_transform, get_valid_transform

# microcontroller dataset class
class MicrocontrollerDataset(Dataset):
    def __init__(self, dir_path, width, height,
                 classes, transforms=None
            ) :
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes

        # get all image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        # short for-loop version: [thing for thing in list_of_things]
        self.all_images = [image_path.split("/")[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)
        print(self.all_images)

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)

        # read the image
        image = cv2.imread(image_path)
        # convert bgr to rgb and chnage type to float32
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) 
        # resize image to specified size from the config file
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255. # normalize

        # capture the corresponding XML file for getting annotations
        anno_filename = image_name[:-4] + ".xml"
        anno_file_path = os.path.join(self.dir_path, anno_filename)

        boxes = []
        labels = []
        tree = et.parse(anno_file_path)
        root = tree.getroot()

        # get height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]

        # box coord for xml files are extracted and corrected for image size given
        for member in root.findall("object"):
            # map the current object name to `classess` list to get the label index
            # and append to `labels` list
            labels.append(self.classes.index(member.find("name").text))

            xmin = int(member.find("bndbox").find("xmin").text)  # xmin = left corner x-coord
            xmax = int(member.find("bndbox").find("xmax").text)  # xmax = right corner x-coord
            ymin = int(member.find("bndbox").find("ymin").text)  # ymin = left corner y-coord
            ymax = int(member.find("bndbox").find("ymax").text)  # ymax = right corner y-coord

            # resize the bbox according to the desired width and height
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height

            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])

        # convert bbox to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bbox
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:,2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0], ), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # prepare final "target" dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # apply image transforms
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target["boxes"],
                                     labels = labels
                                    )
            image_resized = sample["image"]
            target["boxes"] = torch.Tensor(sample["bboxes"])

        print(target)
        return image_resized, target

    def __len__(self, ):
        return len(self.all_images)
    
# prepare the final dataset for the DataLoader
train_dataset = MicrocontrollerDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
valid_dataset = MicrocontrollerDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(valid_dataset)}\n")

# this main function is used to check the dataset and its augmentation
if __name__ == "__main__":
    dataset = MicrocontrollerDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )
    print(f"number of training image: {len(dataset)}")

    def visualize_sample(image, target):
        box = target["boxes"][0]
        label = CLASSES[target["labels"]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 255, 0),
            1
        )
        cv2.putText(
            image, label, (int(box[0]), int(box[1]-5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        cv2.imshow("Image", image)
        cv2.waitKey(0)

    NUM_SAMPLES_TO_VISUALIZE = 5
    for i  in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)