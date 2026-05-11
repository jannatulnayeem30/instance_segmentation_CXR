import matplotlib.pyplot as plt
import cv2
import os
import torch
from PIL import Image
import numpy as np
from dataset import CocoSegmentationDataset
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# convert a PIL image to a PyTorch tensor
def get_transform():
    return ToTensor()

train_dataset = CocoSegmentationDataset(
    root_dir=r'D:\segmentation_project\All_9 copy\All_9 copy',
    annotation_file=r"D:\segmentation_project\train.json",
    transforms=get_transform()  # define this if needed
)

valid_dataset = CocoSegmentationDataset(
    root_dir=r'D:\segmentation_project\All_9 copy\All_9 copy',
    annotation_file=r"D:\segmentation_project\val.json",
    transforms=get_transform()
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))



print("Success!")