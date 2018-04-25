# Chest X-Ray Dataset Explorer
ChestXray14 Explorer is a software coded in Python using OpenCV. It allows easy navigation with keyboard left and right keys, and also presents disease boundary boxes when available in ChestX-ray14. See a small demo on
https://youtu.be/FAoApo_HYL8

Tested only on Ubuntu 16.04 with Python 3 environment with Anaconda. 

Run chestX-ray14_explorer with:
* python chestX-ray14_explorer.py

# Prerequisites 
* Database of chest X-ray images should be availble on the "data/images" folder. 
* Python 3
* Numpy
* Pandas
* OpenCV (Install OpenCV on anaconda by running: conda install -c menpo opencv3)

# Files description:
* config.ini: Configurations
* Folder "data"
  * BBox_List_2017.csv: Information about diseases location from the original ChestX-ray14 dataset
  * Data_Entry_2017: Full ChestX-ray14 dataset with information about images and patients.
  * Data_Entry_2017_sample.csv: Reduced dataset with 5% of data, used by default. (See: https://www.kaggle.com/nih-chest-xrays/sample)
  * lung_masks.csv: Information about lung segmentation for 900 images. Loaded by default.
* Folder "data/images": Contain all images from the ChestXray14 dataset. Download from https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737 and unpack all images on this folder (no subfolder).
* Folder "data/masked_images": Default destination for masked images generation
