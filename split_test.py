import os
from PIL import Image
import glob
from utilitary.utils import seeding, create_dir, epoch_time

test_folder = "test"
test_400_folder = "test_400"

create_dir(test_400_folder)

for file in glob.glob(test_folder + "/*.png"):
    print("Processing image: " + file)
    im = Image.open(file)
    (width, height) = im.size
    imResize = im.crop((0, 0, 400, 400))
    imResize.save(test_400_folder + "/" + file.split("\\")[-1][:-4] + '_top_left.png', 'PNG', quality=90)
    imResize = im.crop((width - 400, 0, width, 400))
    imResize.save(test_400_folder + "/" + file.split("\\")[-1][:-4] + '_top_right.png', 'PNG', quality=90)
    imResize = im.crop((0, height - 400, 400, height))
    imResize.save(test_400_folder + "/" + file.split("\\")[-1][:-4] + '_bottom_left.png', 'PNG', quality=90)
    imResize = im.crop((width - 400, height - 400, width, height))
    imResize.save(test_400_folder + "/" + file.split("\\")[-1][:-4] + '_bottom_right.png', 'PNG', quality=90)



