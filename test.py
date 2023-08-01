import numpy as np
import os, time
from operator import add
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from models import model_unet
from models import model_resnet
from models import model_cnn2
from models import model_cnn4
from models import model_cnn8
from models import model_cnn16

from utilitary.utils import create_dir, seeding
import sys


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)   
    mask = np.concatenate([mask, mask, mask], axis=-1) 
    return mask


if __name__ == "__main__":
    """ Read command line arguments """
    model_name = sys.argv[1]

    """ Setup """
    seeding(42)
    create_dir("results")
    create_dir("results/" + model_name)
    create_dir("results/" + model_name + "_400")

    """ Load dataset """
    images = sorted(glob("test_400/*"))

    """ Define hyperparameters """
    size = (400, 400)

    """ Load the model """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == "unet":
        model = model_unet.build_unet()
    elif model_name == "resnet":
        model = model_resnet.build_resnet()
    elif model_name == "cnn2":
        model = model_cnn2.build_cnn2()
    elif model_name == "cnn4":
        model = model_cnn4.build_cnn4()
    elif model_name == "cnn8":
        model = model_cnn8.build_cnn8()
    elif model_name == "cnn16":
        model = model_cnn16.build_cnn16()
    else:
        raise Exception("Please provide a model name")
    model = model.to(device)

    """ Load weights """
    checkpoint_path = "weights/checkpoint_" + model_name + ".pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    for i, x in tqdm(enumerate(images), total=len(images)):
        """ Extract the name """
        name = x.split("\\")[-1].split(".")[0]

        """ Reading image """
        image = cv2.imread(x, cv2.IMREAD_COLOR) ## (400, 400, 3)
        ## image = cv2.resize(image, size)
        x = np.transpose(image, (2, 0, 1))      ## (3, 400, 400)
        x = x/255.0
        x = np.expand_dims(x, axis=0)           ## (1, 3, 400, 400)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        with torch.no_grad():
            """ Prediction """
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            pred_y = pred_y[0].cpu().numpy()        ## (1, 400, 400)
            pred_y = np.squeeze(pred_y, axis=0)     ## (400, 400)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Saving masks """
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        cat_images = pred_y * 255
        name = model_name + "_400" + "/" + name
        print(f"results/{name}.png")
        imageio.imwrite(f"results/{name}.png", cat_images)
