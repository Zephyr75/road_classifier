# road_classifier_satellite

This is the repository for project 2 from the EPFL CS-433 ML course for which we use different Deep Learning and data engineering techniques for data segmentation of roads, using satellite images for Google Maps. 

Our team is composed of :

#### - Marin Cornuot - <marin.cornuot@epfl.ch>
#### - Antoine Roger - <antoine.roger@epfl.ch>
#### - Hadi Hammoud - <hadi.hammoud@epfl.ch>

# Dependencies setup

To get a working development environment, you can:
- go to the **utilitary** folder and run this line in anaconda prompt: 

        conda env create -f environment.yml

- use your own environment and install PyTorch with cuda using:
  
        conda install -c pytorch torchvision cudatoolkit=10.1 pytorch


## Dataset setup


### Training data
Create a folder named **training** with the following structure:

    training
    ├── images
    │   └── default
    │   └── expanded
    └── groundtruth
        └── default
        └── expanded

Copy all training images in the `training/images/default` folder and all groundtruth images in the **training/groundtruth/default** folder.
Then run the `data_augmentation.py` script at the root of the project, this will create the augmented dataset in the folders mentionned above.

### Test data
Create a folder named `test`.
Take all test images out of their subfolders and copy them in the `test` folder.

## Folders and files

- `models` folder will contain all the trained models for the different architectures trained (unet, vgg, cnn ...).
- `test` contains all the test images (ie. the ones on which we will use our models to estimate the groundtruth mask).
- `results` where the masks of the test images predictions will be stored. This file will be created by running the `test.py`scripts.
- `training` which contains the images from AICrowd as well as the data after running `data_augmentation.py` as described above.
- `utilitary` contains different useful util functions used in this project.
- `data_augmentation` is used to generate the augmented training dataset.
- `submission_to_mask.py` is useful to display the predictions with 16x16 patches as explained in the report. 
- `test.py` runs the model on the test data-set and generates the mask predictions in the `results` folder.
- `train.py` allows to train the different model as explained below.


# Running the code

To get the full testing dataset, run:

    python split_test.py

Every command below needs to be run from the root of the project and with the name of the model as argument:
- unet
- resnet
- cnn2
- cnn4
- cnn8
- cnn16

To train a model, run:

    python train.py <model_name>

To test a model, run:

    python test.py <model_name>

To get the full size result images, run:

    python unite_results.py <model_name>

To generate the submission file, run:

    python mask_to_submission.py <model_name>
