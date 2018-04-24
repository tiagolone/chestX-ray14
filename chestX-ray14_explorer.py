# Import libraries necessary for this project
import cv2
import numpy as np
import pandas as pd
import ast
import time
from configparser import ConfigParser

CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

class Patient(object):
    def __init__(self, index, df, img_dir):
        self.index = index
        self.img_name = df.ix[index, 0]
        self.all_labels = df.ix[index, 1]
        self.follow_up = df.ix[index, 2] 
        self.patient_id = df.ix[index, 3] 
        self.age = df.ix[index, 4] 
        self.gender = df.ix[index, 5] 
        self.view = df.ix[index, 6] 
        self.path = img_dir + self.img_name

        self.points_left = []  # List of points defining mask for left lung
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
        left = ast.literal_eval(df.ix[self.index, 23])
        if len(left)>0:
            self.points_left = np.asarray(left)

        right = ast.literal_eval(df.ix[self.index, 24])
        if len(right)>0:
            self.points_right = np.asarray(right)

        #self.points_right = ast.literal_eval(right)

    def get_description(self):
        desc = 'Index:{0}, File:{1}, PatientID:{2}, Follow Up:{3}, Age:{4}, Gender:{5}, View:{6}'.format(self.index, 
            self.img_name, self.patient_id, self.follow_up, self.age, self.gender, self.view)

        return desc

    def get_labels(self):
        return self.all_labels


KEY_LEFT = 81
KEY_RIGHT = 83
KEY_DOWN = 84
KEY_ESC = 27
KEY_1 = 49
KEY_2 = 50
KEY_3 = 51
KEY_4 = 52
KEY_B = 98
KEY_C = 99
KEY_E = 101
KEY_H = 104
KEY_I = 105
KEY_M = 109
KEY_G = 103
KEY_R = 114
KEY_S = 115

FINAL_LINE_COLOR_L = (0, 255, 0)
FINAL_LINE_COLOR_R = (0, 255, 255)
WORKING_LINE_COLOR = (255, 255,0)

LEFT_LUNG  = 0   # Left lung
RIGHT_LUNG = 1  # Right lung
LUNG_NONE  = 2

# Based on https://stackoverflow.com/questions/37099262/drawing-filled-polygon-using-mouse-events-in-open-cv-using-python
class chestXrayExplorer(object):
    def __init__(self, window_name, data_file, lung_file, bbox_file, img_src_dir, img_out_dir):
        self.window_name = window_name # Name for our window        
        self.item_index = 0
        self.cur_item = None
        self.terminate = False
        self.img_src_dir = img_src_dir

        self.mask_mode_enabled = False
        self.mask_currrent_lung = LEFT_LUNG
        self.mask_done = False
        self.mask_current_pos = (0, 0)      # Current position, so we can draw the line-in-progress
        self.mask_left_lung = []
        self.mask_right_lung = []
        self.mask_out_dir = img_out_dir

        self.show_info_enabled = True
        self.show_help_enabled = True
        self.show_lung_mask_contour_enabled = True
        self.show_lung_mask_enabled = False
        self.show_bbox_enabled = True

        # Load data
        self.data = self.load_data(data_file, lung_file, bbox_file)      

    def load_data(self, data_file, lung_file, bbox_file):
        try:
            data = pd.read_csv(data_file)
            print("Dataset data has {} samples with {} features each.".format(*data.shape))

            lung = pd.read_csv(lung_file)
            print("Dataset lung has {} samples with {} features each.".format(*lung.shape))

            bbox = pd.read_csv(bbox_file)
            print("Dataset bbox has {} samples with {} features each.".format(*bbox.shape))

            # Join data with lung masks and bbox information
            df = data.join(bbox.reset_index(), on='Image Index', how='left', rsuffix='_mask')
            df = df.join(lung, how='left', rsuffix='_mask')

            left_lung = df[pd.notnull(df.points_left_lung)].points_left_lung     #.apply(ast.literal_eval)
            right_lung = df[pd.notnull(df.points_right_lung)].points_right_lung  #.apply(ast.literal_eval)

            df['points_left_lung'] = left_lung
            df['points_right_lung'] = right_lung

            df.loc[df['points_left_lung'].isnull(),['points_left_lung']] = df.loc[df['points_left_lung'].isnull(),'points_left_lung'].apply(lambda x: '[]')
            df.loc[df['points_right_lung'].isnull(),['points_right_lung']] = df.loc[df['points_right_lung'].isnull(),'points_right_lung'].apply(lambda x: '[]')

            # Print columns to verify
            # Should rename and exclude some columns and use only names instead of index
            print(df.columns.values)

            return df
        except:
            print("Dataset could not be loaded. Is the dataset missing?")
            exit()        

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)
        if self.mask_mode_enabled and not self.mask_done:
            if event == cv2.EVENT_MOUSEMOVE:
                # We want to be able to draw the line-in-progress, so update current mouse position
                self.mask_current_pos = (x, y)
            elif event == cv2.EVENT_LBUTTONDOWN:
                # Left click means adding a point at current position to the list of points
                if self.mask_currrent_lung == LEFT_LUNG:
                    self.mask_left_lung.append((x, y))
                elif self.mask_currrent_lung == RIGHT_LUNG:
                    self.mask_right_lung.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right click means we're done
                if self.mask_currrent_lung == LEFT_LUNG:
                    self.mask_currrent_lung = RIGHT_LUNG
                else:
                    self.mask_currrent_lung = LUNG_NONE
                    self.mask_done = True

    def process_key(self, keycode):
        terminate = False

        if keycode == KEY_H:
            self.show_help_enabled = not self.show_help_enabled
        elif keycode == KEY_1:
            self.show_info_enabled = not self.show_info_enabled
        elif keycode == KEY_2:
            self.show_bbox_enabled = not self.show_bbox_enabled
        elif keycode == KEY_3:
            self.show_lung_mask_contour_enabled = not self.show_lung_mask_contour_enabled
        elif keycode == KEY_4:
            self.show_lung_mask_enabled = not self.show_lung_mask_enabled
        elif keycode == KEY_G:
            try:
                index = int(input('Index:'))
                self.goto(index)
            except ValueError:
                print("Not a number")
        elif keycode == KEY_LEFT:
            self.prev()
        elif keycode == KEY_RIGHT:
            self.next()
        elif keycode == KEY_E:
            self.mask_mode_enabled = not self.mask_mode_enabled
            self.mask_reset()
        elif keycode == KEY_R:
            self.mask_reset()
        elif keycode == KEY_S:
            self.mask_save_points_to_file("output.csv", True)
        elif keycode == KEY_I:
            self.mask_save_images_to_dir(self.mask_out_dir)

        elif keycode == KEY_ESC:
            # If ESC pressed and editing mask, stop. If not editing exit.
            if self.mask_mode_enabled:
                self.mask_mode_enabled = False
                self.mask_reset()
            else:
                terminate = True
        else:
            if keycode!=255:
                print(keycode)

        return terminate

    def show_info(self, canvas, item):        
        #color = (50,50,50)
        #pts = np.array( [[[10,10], [10,50], [900,50], [900,10]]], dtype=np.int32 )
        #cv2.fillPoly(canvas, np.array(pts), color)          
        cv2.putText(canvas, item.get_description(), (20,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0,255,255))
        cv2.putText(canvas, item.get_labels(), (20,40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0,255,255))

    def show_help(self, canvas):  
        help_text = ['Help:',
            '  h: Show/Hide help',  
            '  1: Show/Hide patient info',
            '  2: Show/Hide disease region box',
            '  3: Show/Hide lung mask countour',
            '  4: Show/Hide lung mask',
            '  g: Goto patient',
            '  ->: Next patient',
            '  <-: Previous patient',
            '  e: Edit current patient lung mask',
            '  r: Reset current lung mask being edited',
            '  s: Save lung masks to file (points)',
            '  i: Save lung masks to file (images)',
            '  esc: Exit']

        width = 450
        height = 20*len(help_text)+10
        top = 110
        left = int((canvas.shape[0]-width)/2)

        color = (50,50,50)
        pts = np.array( [[[left,top], [left,top+height], [left+width,top+height], [left+width,top]]], dtype=np.int32 )

        cv2.fillPoly(canvas, np.array(pts), color)  
        x = left+10
        y = top+20        
        for text in help_text:            
            cv2.putText(canvas, text, (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.75, (0,255,255))
            y += 20

    def show_bbox(self, canvas, item):
        if item.box_available:
            cv2.rectangle(canvas, (item.box_x, item.box_y), (item.box_x+item.box_w, item.box_y+item.box_h), (0,255,0), 1)
            cv2.putText(canvas,item.box_label, (item.box_x,item.box_x+item.box_h+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,255))

    def show_lung_mask_contour(self, canvas, item):
        if not self.mask_mode_enabled or (len(self.mask_left_lung)==0 and len(self.mask_right_lung)==0):
            if len(item.points_left) > 0:
                cv2.drawContours(canvas, np.array([item.points_left]), False, FINAL_LINE_COLOR_L, 2)
            if len(item.points_right) > 0:
                cv2.drawContours(canvas, np.array([item.points_right]), False, FINAL_LINE_COLOR_R, 2)
                
    def show_lung_mask(self, canvas, item):
        if not self.mask_mode_enabled or (len(self.mask_left_lung)==0 and len(self.mask_right_lung)==0):
            if (len(item.points_left) > 0) and (len(item.points_right) > 0):
                stencil = np.zeros(item.img.shape).astype(item.img.dtype)
                contours = [np.array([item.points_left]), np.array([item.points_right])]
                color = [255, 255, 255]
                cv2.fillPoly(stencil, contours, color)
                canvas = cv2.bitwise_and(canvas, stencil)

        return canvas

    def update_lung_mask(self, canvas):
        # Draw all the current polygon segments
        if (len(self.mask_left_lung) > 0):
            if self.mask_currrent_lung == LEFT_LUNG and len(self.mask_left_lung) > 0:
                cv2.polylines(canvas, np.array([self.mask_left_lung]), False, FINAL_LINE_COLOR_L, 2)
                cv2.line(canvas, self.mask_left_lung[-1], self.mask_current_pos, WORKING_LINE_COLOR)
            else:
                cv2.drawContours(canvas, np.array([self.mask_left_lung]), False, FINAL_LINE_COLOR_L, 2)

        if (len(self.mask_right_lung) > 0):
            if self.mask_currrent_lung == RIGHT_LUNG and len(self.mask_right_lung) > 0:                    
                cv2.polylines(canvas, np.array([self.mask_right_lung]), False, FINAL_LINE_COLOR_R, 2)
                cv2.line(canvas, self.mask_right_lung[-1], self.mask_current_pos, WORKING_LINE_COLOR)
            else:
                cv2.drawContours(canvas, np.array([self.mask_right_lung]), False, FINAL_LINE_COLOR_L, 2)
        

    def goto(self, index):
        self.item_index = index
        self.cur_item = Patient(index, self.data, self.img_src_dir)

        if self.mask_mode_enabled:
            self.mask_reset()

    def prev(self):
        if self.item_index==0:
            self.item_index = self.data.shape[0]-1 
        else:
            self.item_index -= 1

        self.goto(self.item_index)

    def next(self):
        if self.item_index<self.data.shape[0]-1:
            self.item_index += 1
        else:
            self.item_index = 0

        self.goto(self.item_index)

    def mask_reset(self):
        self.mask_done = False
        self.mask_currrent_lung = LEFT_LUNG
        self.mask_left_lung = []
        self.mask_right_lung = []

    def mask_save(self):
        self.data.iloc[self.item_index, 23] = str(self.mask_left_lung)
        self.data.iloc[self.item_index, 24] = str(self.mask_right_lung)

    def mask_save_points_to_file(self, file_name, include_timestamp):
        header = ['Image Index', 'points_left_lung', 'points_right_lung']

        if include_timestamp:
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%Y%m%d")
            timestampLaunch = timestampDate + '-' + timestampTime
            file_name = timestampLaunch + '_' + file_name
            print('[' + timestampDate + ' - ' + timestampTime + ']: ' + 'Saving... ')

        self.data.to_csv(file_name, columns = header, index=False)

    def mask_save_images_to_dir(self, dst_path):
        img_with_masks = self.data[self.data.points_left_lung!='[]']#[['Image Index', 'points_left_lung', 'points_right_lung']]

        for index, row in img_with_masks.iterrows():
            patient = Patient(index, img_with_masks, self.img_src_dir)    
            print(str(index) + ': ' + self.mask_out_dir + patient.img_name)

            masked_img = patient.img.copy()
            stencil = np.zeros(patient.img.shape).astype(patient.img.dtype)
            contours = [np.array([patient.points_left]), np.array([patient.points_right])]
            color = [255, 255, 255]
            cv2.fillPoly(stencil, contours, color)
            masked_img = cv2.bitwise_and(masked_img, stencil)

            cv2.imwrite(self.mask_out_dir + patient.img_name, masked_img)        

    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        
        # Goto patient 0 and enter in the update loop
        self.goto(0)
        while not self.terminate:
            canvas = self.cur_item.img.copy() #np.zeros(CANVAS_SIZE, np.uint8)

            if self.show_lung_mask_enabled:
                canvas = self.show_lung_mask(canvas, self.cur_item)

            if self.show_lung_mask_contour_enabled:
                self.show_lung_mask_contour(canvas, self.cur_item)

            if self.show_bbox_enabled:
                self.show_bbox(canvas, self.cur_item)

            if self.show_info_enabled:
                self.show_info(canvas, self.cur_item)

            if self.show_help_enabled:
                self.show_help(canvas)

            if(self.mask_mode_enabled):
                self.update_lung_mask(canvas)

                if self.mask_done:
                    # If done, update mask information on DataFrame and go to next patient
                    self.mask_save()
                    self.next()

                    # Autosave after 10 or 100 masks edited
                    if (self.item_index % 10)==0:
                        if (self.item_index % 100)==0:
                            self.mask_save_points_to_file('autosave.100.csv', True)
                        else:
                            self.mask_save_points_to_file('autosave.10.csv', True)                    

            # Update windows
            cv2.imshow(self.window_name, canvas)
            
            # Process keyboard 
            keycode = cv2.waitKey(50)
            if self.process_key(keycode):
                break

# ============================================================================
if __name__ == "__main__":
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    image_source_dir = cp["DEFAULT"].get("image_source_dir")
    image_output_dir = cp["DEFAULT"].get("image_output_dir")
    data_entry_file = cp["DEFAULT"].get("data_entry_file")
    lung_masks_file = cp["DEFAULT"].get("lung_masks_file")
    bbox_list_file = cp["DEFAULT"].get("bbox_list_file")

    # Run explorer
    explorer = chestXrayExplorer("X-Ray Image", 
        data_entry_file, lung_masks_file, bbox_list_file, 
        image_source_dir, image_output_dir)
    explorer.run()

