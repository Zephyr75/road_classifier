# for every image in the "training" folder, copy the image in a training_256 folder and resize it to 256x256
import os
from PIL import Image
import glob

target_resolution = (256, 256)

folder_name = "training_" + str(target_resolution[0]) + "x" + str(target_resolution[1])

# create the folder
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    os.makedirs(folder_name + "/images")
    os.makedirs(folder_name + "/images/default")
    os.makedirs(folder_name + "/images/expanded")
    os.makedirs(folder_name + "/groundtruth")
    os.makedirs(folder_name + "/groundtruth/default")
    os.makedirs(folder_name + "/groundtruth/expanded")

for file in glob.glob("training/images/default/*.png"):
    print("Processing image: " + file)
    im = Image.open(file)
    imResize = im.resize(target_resolution, Image.ANTIALIAS)
    imResize.save(folder_name + "/images/" + file.split("/")[-1], 'PNG', quality=90)

for file in glob.glob("training/images/expanded/*.png"):
    print("Processing image: " + file)
    im = Image.open(file)
    imResize = im.resize(target_resolution, Image.ANTIALIAS)
    imResize.save(folder_name + "/images/" + file.split("/")[-1], 'PNG', quality=90)

for file in glob.glob("training/groundtruth/default/*.png"):
    print("Processing groundtruth: " + file)
    im = Image.open(file)
    imResize = im.resize(target_resolution, Image.ANTIALIAS)
    imResize.save(folder_name + "/groundtruth/" + file.split("/")[-1], 'PNG', quality=90)

for file in glob.glob("training/groundtruth/expanded/*.png"):
    print("Processing groundtruth: " + file)
    im = Image.open(file)
    imResize = im.resize(target_resolution, Image.ANTIALIAS)
    imResize.save(folder_name + "/groundtruth/" + file.split("/")[-1], 'PNG', quality=90)


