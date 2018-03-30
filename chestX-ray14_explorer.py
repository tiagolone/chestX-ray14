# Import libraries necessary for this project
import cv2
import numpy as np
import pandas as pd

DATA_DIR = '/media/tiagolone/Extra/xray/'
IMAGE_DIR = '/media/tiagolone/Extra/xray/images/unzipped/'

CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

KEY_LEFT = 81
KEY_RIGHT = 83
KEY_ESC = 27

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

def click_and_mask(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 
		# draw a rectangle around the region of interest
		cv2.rectangle(param, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("X-Ray Image", param)


# Load the dataset
try:
    bbox = pd.read_csv(DATA_DIR + "BBox_List_2017.csv")
    print("Dataset has {} samples with {} features each.".format(*bbox.shape))

    data = pd.read_csv(DATA_DIR + "Data_Entry_2017.csv")
    print("Dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")
    exit()

df = data.join(bbox, on='Image Index', how='left', lsuffix='_l', rsuffix='_r')
 
index = 0
while True:
    img_name = df.ix[index, 0]
    all_labels = df.ix[index, 1]
    follow_up = df.ix[index, 2] 
    patient_id = df.ix[index, 3] 
    age = df.ix[index, 4] 
    gender = df.ix[index, 5] 
    view = df.ix[index, 6] 

    path = IMAGE_DIR + img_name
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    print("Image: ", index, img_name, patient_id, follow_up, age, gender, view)
    print("Labels: ", all_labels)

    img_desc = 'Index:{0}, File:{1}, PatientID:{2}, Follow Up:{3}, Age:{4}, Gender:{5}, View:{6}'.format(index, img_name, patient_id, follow_up, age, gender, view)
    cv2.putText(img, img_desc, (20,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0,255,255))
    cv2.putText(img, all_labels, (20,40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0,255,255))

    if not pd.isnull(df.ix[index, 13]):
        label = df.ix[index, 1]
        x = np.intp(df.ix[index, 2])
        y = np.intp(df.ix[index, 3])
        w = np.intp(df.ix[index, 4])
        h = np.intp(df.ix[index, 5])

        print("Box: ", label, x, y, w, h)
    
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 1)
        cv2.putText(img,label, (x,x+h+50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,255))

    cv2.namedWindow("X-Ray Image")
    cv2.imshow("X-Ray Image", img)
#    clone = img.copy()
    cv2.setMouseCallback("X-Ray Image", click_and_mask, img)
    
    # Keyboard
    keycode = cv2.waitKey()
    if keycode == KEY_ESC:
        break
    elif keycode == KEY_LEFT:
        if index==0:
            index = df.shape[0]-1 
        else:
            index -= 1
    elif keycode == KEY_RIGHT:
        if index<df.shape[0]-1:
            index += 1
        else:
            index = 0
            
 
