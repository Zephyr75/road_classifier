import os
from PIL import Image, ImageEnhance
from skimage.util import random_noise
import random
import numpy as np
import torchvision.transforms.functional as F
from tqdm import tqdm

# Set the location of the training images
train_dir = 'training/images/default'

# Set the location where the augmented images will be saved
save_dir = 'training/images/expanded'

# Loop through all images in the training folder
print("Performing data augmentation on the images")
for filename in tqdm(os.listdir(train_dir)):
    
    # for different rotations, namely 0, 90, 180, 270
    # basic rotations + cropping
    for i in range(4):
        im = Image.open(os.path.join(train_dir, filename))
        angle = 90 * i
        im = im.rotate(angle)
        im.save(os.path.join(save_dir, 'rotated_' + str(angle) + '_' + filename))

        width, height = im.size
        im = im.crop((width * 0.05, height * 0.05, width * 0.95, height * 0.95))
        im = im.resize((width, height))
        im.save(os.path.join(save_dir, 'cropped_and_rotated_' + str(angle) + '_' + filename))
        

    # adjust the brightness of the image
    im = Image.open(os.path.join(train_dir, filename))
    im = ImageEnhance.Brightness(im).enhance(random.uniform(0.8,1.2))
    im = im.rotate(90) #we first apply a rotation
    im.save(os.path.join(save_dir, 'brightness_' + filename))
    
    # play with contrast 
    im = Image.open(os.path.join(train_dir, filename))
    im = im.rotate(270) #first apply a rotation to not always train on variations of the same image
    contrast_factor = random.uniform(0.7, 1.3)
    im = ImageEnhance.Contrast(im).enhance(contrast_factor)
    im.save(os.path.join(save_dir, 'contrast_' + filename))

    # Adjust the huing of the image
    im = Image.open(os.path.join(train_dir, filename))
    im = F.adjust_hue(im, random.uniform(-0.1,0.1))    
    im.save(os.path.join(save_dir, 'hue_' + filename))

    #adjust the saturation of the image 
    im = Image.open(os.path.join(train_dir, filename))
    enhancer = ImageEnhance.Color(im)
    saturation = random.uniform(1,1.5)  # Increase saturation by 0 to 50%
    im = enhancer.enhance(saturation)
    im.save(os.path.join(save_dir, 'saturation_' + filename))

    # Flip the image horizontally and save it again
    im = Image.open(os.path.join(train_dir, filename))
    im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im.save(os.path.join(save_dir, 'flipped_LR_' + filename))
    
    im = Image.open(os.path.join(train_dir, filename))
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    im.save(os.path.join(save_dir, 'flipped_TB_' + filename))
    

######      disabled for performance reasons        #######     
 # Add random gaussian noise to the image and save it again
    # im = Image.open(os.path.join(train_dir, filename))
    # im_array = np.array(im)
    # im_array = random_noise(im_array, mode='gaussian')
    # # Convert the floating-point array to an integer array
    # im_array = (im_array * 255).astype(np.uint8)
    # im = Image.fromarray(im_array)
    # im.save(os.path.join(save_dir, 'noisy_' + filename))
        

# Set the location of the groundtruth images
train_dir = 'training/groundtruth/default'

# Set the location where the augmented images will be saved
save_dir = 'training/groundtruth/expanded'

print("Performing data augmentation on the grountruth images")
# Loop through all images in the training folder
for filename in tqdm(os.listdir(train_dir)):
    
    # for different rotations, namely 0, 90, 180, 270
    # basic rotations + cropping
    for i in range(4):
        im = Image.open(os.path.join(train_dir, filename))
        angle = 90 * i
        im = im.rotate(angle)
        im.save(os.path.join(save_dir, 'rotated_' + str(angle) + '_' + filename))

        width, height = im.size
        im = im.crop((width * 0.05, height * 0.05, width * 0.95, height * 0.95))
        im = im.resize((width, height))
        im.save(os.path.join(save_dir, 'cropped_and_rotated_' + str(angle) + '_' + filename))
        
    im = Image.open(os.path.join(train_dir, filename))
    # do not adjust the brightness of the mask
    im = im.rotate(90) # apply rotation before
    im.save(os.path.join(save_dir, 'brightness_' + filename))
    

    # Flip the image and save it again
    im = Image.open(os.path.join(train_dir, filename))
    im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im.save(os.path.join(save_dir, 'flipped_LR_' + filename))
    
    im = Image.open(os.path.join(train_dir, filename))
    im = im.transpose(Image.FLIP_TOP_BOTTOM)
    im.save(os.path.join(save_dir, 'flipped_TB_' + filename))

    # We do NOT add noise nor hue nor saturation nor constrast to the mask !
    im = Image.open(os.path.join(train_dir, filename))
   # im.save(os.path.join(save_dir, 'noisy_' + filename)) # No noise for perf reasons
    im.save(os.path.join(save_dir, 'hue_' + filename))
    im.save(os.path.join(save_dir, 'saturation_' + filename))
    im = im.rotate(270) #first apply a rotation to not always train on variations of the same image
    im.save(os.path.join(save_dir, 'constrast_' + filename))
