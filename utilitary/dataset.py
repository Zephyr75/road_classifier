import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path):

        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image/255.0 ## (400, 400, 3)
        image = np.transpose(image, (2, 0, 1))  ## (3, 400, 400)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask/255.0   ## (400, 400)
        mask = np.expand_dims(mask, axis=0) ## (1, 400, 400)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples

class DriveDataset_vgg(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = sorted(os.listdir(image_dir))
        self.mask_names = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_idx = idx % len(self.image_names)
        image_path = os.path.join(self.image_dir, self.image_names[image_idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[image_idx])

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        image = np.array(image)
        mask = np.array(mask, dtype=np.float32)
        return image, mask

# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])