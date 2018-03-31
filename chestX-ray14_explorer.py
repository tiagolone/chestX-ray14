# Import libraries necessary for this project
import cv2
import numpy as np
import pandas as pd
import ast
import time

DATA_DIR = '/media/tiagolone/Extra/xray/'
IMAGE_DIR = '/media/tiagolone/Extra/xray/images/unzipped/'

CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

KEY_LEFT = 81
KEY_RIGHT = 83
KEY_DOWN = 84
KEY_ESC = 27
KEY_M = 109
KEY_R = 114
KEY_S = 115

FINAL_LINE_COLOR_L = (0, 255, 0)
FINAL_LINE_COLOR_R = (0, 255, 255)
WORKING_LINE_COLOR = (255, 255,0)

class Item(object):
    def __init__(self, index, df):
        self.index = index
        self.img_name = df.ix[index, 0]
        self.all_labels = df.ix[index, 1]
        self.follow_up = df.ix[index, 2] 
        self.patient_id = df.ix[index, 3] 
        self.age = df.ix[index, 4] 
        self.gender = df.ix[index, 5] 
        self.view = df.ix[index, 6] 
        self.path = IMAGE_DIR + self.img_name

        self.points_left = [] # List of points defining mask for left lung
        self.points_right = [] # List of points defining mask for right lung

        self.img = cv2.imread(self.path, cv2.IMREAD_COLOR)

        if pd.isnull(df.ix[index, 13]):
            self.box_available = False
            self.box_label = ''
            self.box_x = 0
            self.box_y = 0
            self.box_w = 0
            self.box_h = 0
        else:
            self.box_available = True
            self.box_label = df.ix[index, 1]
            self.box_x = np.intp(df.ix[index, 2])
            self.box_y = np.intp(df.ix[index, 3])
            self.box_w = np.intp(df.ix[index, 4])
            self.box_h = np.intp(df.ix[index, 5])

        self.load_points(df)

    def load_points(self, df): 
        left = df.ix[self.index, 21]
        if left != '':
            self.points_left = np.asarray(left)

        right = df.ix[self.index, 22]
        if right != '':
            self.points_right = np.asarray(right)

        #self.points_right = ast.literal_eval(right)

    def get_description(self):
        desc = 'Index:{0}, File:{1}, PatientID:{2}, Follow Up:{3}, Age:{4}, Gender:{5}, View:{6}'.format(self.index, 
            self.img_name, self.patient_id, self.follow_up, self.age, self.gender, self.view)

        return desc

    def get_labels(self):
        return self.all_labels

# Based on https://stackoverflow.com/questions/37099262/drawing-filled-polygon-using-mouse-events-in-open-cv-using-python
class chestXrayExplorer(object):
    def __init__(self, window_name):
        self.STATE_LEFT = 0
        self.STATE_RIGHT = 1
        self.STATE_DONE = 2

        self.window_name = window_name # Name for our window        

        self.state = self.STATE_LEFT
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.cur_item = None
        
        #self.points_left = [] # List of points defining our polygon left
        #self.points_right = [] # List of points defining our polygon right


    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            if self.state == self.STATE_LEFT:
                #print("Adding point #%d with position(%d,%d)" % (len(self.cur_item.points_left), x, y))
                self.cur_item.points_left.append((x, y))
            else:
                #print("Adding point #%d with position(%d,%d)" % (len(self.cur_item.points_right), x, y))
                self.cur_item.points_right.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done

            if self.state == self.STATE_LEFT:
                #print("Completing polygon with %d points." % len(self.cur_item.points_left))
                self.state = self.STATE_RIGHT
            else:
                #print("Completing polygon with %d points." % len(self.cur_item.points_right))
                self.state = self.STATE_DONE
                self.done = True

    def run(self, item):
        esc = False
        self.cur_item = item

        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name) #, flags=cv2.CV_WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, item.img)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse, item.img)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            canvas = item.img.copy() #np.zeros(CANVAS_SIZE, np.uint8)

            # Draw all the current polygon segments
            if (len(self.cur_item.points_left) > 0):
                #cv2.drawContours(canvas, np.array([self.cur_item.points_left]), False, FINAL_LINE_COLOR_L, 2)
                cv2.polylines(canvas, np.array([self.cur_item.points_left]), False, FINAL_LINE_COLOR_L, 2)
            if (len(self.cur_item.points_right) > 0):
                #cv2.drawContours(canvas, np.array([self.cur_item.points_right]), False, FINAL_LINE_COLOR_R, 2)
                cv2.polylines(canvas, np.array([self.cur_item.points_right]), False, FINAL_LINE_COLOR_R, 2)

            # And  also show what the current segment would look like
            if self.state == self.STATE_LEFT and len(self.cur_item.points_left) > 0:
                cv2.line(canvas, self.cur_item.points_left[-1], self.current, WORKING_LINE_COLOR)
            elif self.state == self.STATE_RIGHT and len(self.cur_item.points_right) > 0:
                cv2.line(canvas, self.cur_item.points_right[-1], self.current, WORKING_LINE_COLOR)

            # Update the window
            cv2.imshow(self.window_name, canvas)
            
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                esc = True
                self.done = True

        # And show it
        #cv2.imshow(self.window_name, canvas)
        #cv2.waitKey()

        #stencil = np.zeros(item.img.shape).astype(item.img.dtype)
        #contours = [np.array([self.cur_item.points_left]), np.array([self.cur_item.points_right])]
        #color = [255, 255, 255]
        #cv2.fillPoly(stencil, contours, color)
        #result = cv2.bitwise_and(item.img, stencil)
        #cv2.imshow(self.window_name, result)
        #cv2.waitKey()

        return esc

# ============================================================================

if __name__ == "__main__":
    # Load the dataset
    try:
        bbox = pd.read_csv(DATA_DIR + "BBox_List_2017.csv")
        print("Dataset has {} samples with {} features each.".format(*bbox.shape))

        data = pd.read_csv(DATA_DIR + "Data_Entry_2017.csv")
        print("Dataset has {} samples with {} features each.".format(*data.shape))
    except:
        print("Dataset could not be loaded. Is the dataset missing?")
        exit()

    df = data.join(bbox, on='Image Index', how='left', rsuffix='_r')
    df["points_left_lung"] = ''
    df["points_right_lung"] = ''
    print(df.columns.values)

    index = 0
    mask_mode = False
    while True:
        item = Item(index, df)

        #print("Image: ", item.get_description())
        #print("Labels: ", item.get_labels())
        cv2.putText(item.img, item.get_description(), (20,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0,255,255))
        cv2.putText(item.img, item.get_labels(), (20,40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0,255,255))

        if item.box_available:
            #print("Box: ", item.box_label, item.box_x, item.box_y, item.box_w, item.box_h)        
            cv2.rectangle(item.img, (item.box_x, item.box_y), (item.box_x+item.box_w, item.box_y+item.box_h), (0,255,0), 1)
            cv2.putText(item.img,item.box_label, (item.box_x,item.box_x+item.box_h+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,255))

        # User finised entering the polygon points, so let's make the final drawing
        canvas = item.img.copy() #np.zeros(CANVAS_SIZE, np.uint8)
        # of a filled polygon
        if (len(item.points_left) > 0):
            cv2.drawContours(canvas, np.array([item.points_left]), False, FINAL_LINE_COLOR_L, 2)
        if (len(item.points_right) > 0):
            cv2.drawContours(canvas, np.array([item.points_right]), False, FINAL_LINE_COLOR_R, 2)

        cv2.namedWindow("X-Ray Image")
        cv2.imshow("X-Ray Image", canvas)
        
        if(mask_mode):
            explorer = chestXrayExplorer("X-Ray Image")
            if explorer.run(item):
                mask_mode = False
            else:
                #cv2.imwrite("polygon.png", image)
                #print("Polygon Left = %s" % item.points_left)
                df.iat[index, 21] = item.points_left
                #print("Polygon Right = %s" % item.points_right)
                df.iat[index, 22] = item.points_right

                if index<df.shape[0]-1:
                    index += 1
                else:
                    index = 0

                if (index % 10)==0:
                    timestampTime = time.strftime("%H:%M:%S")
                    timestampDate = time.strftime("%d/%m/%Y")
                    timestampLaunch = timestampDate + '-' + timestampTime
                    print('[' + timestampDate + ' - ' + timestampTime + ']: ' + 'Autosaving... ')
                    header = ['Image Index', 'points_left_lung', 'points_right_lung']
                    if (index % 100)==0:
                        df.to_csv('autosave.10.csv', columns = header, index=False)
                    else:
                        df.to_csv('autosave.100.csv', columns = header, index=False)
        else:
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
            elif keycode == KEY_M or keycode == KEY_DOWN:
                mask_mode = not mask_mode
            elif keycode == KEY_R:
                df.iat[index, 21] = ''
                df.iat[index, 22] = ''
            elif keycode == KEY_S:
                header = ['Image Index', 'points_left_lung', 'points_right_lung']

                timestampTime = time.strftime("%H%M%S")
                timestampDate = time.strftime("%d%m%Y")
                timestampLaunch = timestampDate + '-' + timestampTime
                fileName = 'out-' + timestampLaunch + '.csv'

                df.to_csv(fileName, columns = header, index=False)
            else:
                print(keycode)
            




 
            
 
