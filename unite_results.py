import os
from PIL import Image
import glob
from utilitary.utils import seeding, create_dir, epoch_time
import sys

model_name = sys.argv[1]
results_folder = "results/" + model_name
results_400_folder = "results/" + model_name + "_400"

images_map = {}

for file in glob.glob(results_400_folder + "/*.png"):
    name = file.split("\\")[-1][:-4]
    file_name = name.split("_")[0] + "_" + name.split("_")[1] + ".png"
    print(file_name)

    if images_map.get(file_name) is None:
        images_map[file_name] = []
    images_map[file_name].append(file)
    
for name in images_map:
    images = images_map[name]
    print("Processing image: " + name)
    im_top_left = Image.open(images[2])
    im_top_right = Image.open(images[3])
    im_bottom_left = Image.open(images[0])
    im_bottom_right = Image.open(images[1])
    (width, height) = (608, 608)
    im = Image.new('RGB', (width, height))
    im.paste(im_top_left, (0, 0))
    im.paste(im_top_right, (width - 400, 0))
    im.paste(im_bottom_left, (0, height - 400))
    im.paste(im_bottom_right, (width - 400, height - 400))
    im.save(results_folder + "/" + name, 'PNG', quality=90)
