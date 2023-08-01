import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import sys

from utilitary.utils import seeding, create_dir, epoch_time
from utilitary.loss import DiceLoss, DiceBCELoss, BCELoss
from utilitary.dataset import DriveDataset, DriveDataset_vgg

from models import model_unet
from models import model_resnet
from models import model_cnn2
from models import model_cnn4
from models import model_cnn8
from models import model_cnn16

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    """ Read command line arguments """
    model_name = sys.argv[1]

    """ Setup """
    seeding(42)
    create_dir("weights")

    """ Load dataset """
    train_x = sorted(glob("training/images/expanded/*"))
    train_y = sorted(glob("training/groundtruth/expanded/*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} \n"
    print(data_str)

    """ Hyperparameters """
    size = (400, 400)
    batch_size = 2
    num_epochs = 100
    lr = 1e-4

    """ Create dataloader """
    #train_dataset = DriveDataset_vgg(train_x, train_y) for of vgg16
    train_dataset = DriveDataset(train_x, train_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

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
    # elif model_name == "vgg16":	
    #     model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=False)
    else:
        raise Exception("Please provide a model name")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    checkpoint_path = "weights/checkpoint_" + model_name + ".pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Checkpoint loaded: {checkpoint_path}")

    """ Training the model """

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)

        """ Saving the model """ 
        #once every 5 five epoch to save some computing power
        if epoch % 5 == 0:
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        print(data_str)