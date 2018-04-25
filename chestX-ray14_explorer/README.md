# chestX-ray14_explorer
ChestX-ray14 dataset explorer

Tested on on Ubuntu 16.04 with Python 3 environment with Anaconda. 

Run chestX-ray14_explorer with:
	python chestX-ray14_explorer.py

Database of chest X-ray images should be availble on the "data/images" folder. 
Download from https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737 and unpack all images on this folder (no subfolder).

Dependencies: 
	- numpy
	- pandas
	- opencv

Install OpenCV on anaconda by running:
	conda install -c menpo opencv3

Files in folder "data" description:
	BBox_List_2017.csv: Information about diseases location from the original ChestX-ray14 dataset
	Data_Entry_2017: Full ChestX-ray14 dataset with information about images and patients.
	Data_Entry_2017_sample.csv: Reduced dataset with 5% of data, used by default. (See: https://www.kaggle.com/nih-chest-xrays/sample)
	lung_masks.csv: Information about lung segmentation for 900 images. Loaded by default.

