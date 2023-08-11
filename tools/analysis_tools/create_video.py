import random as rd
import cv2 as cv
import numpy as np



class RecordMovie(object):

    def __init__(self, img_width, img_height):
        self.video_writer = None 
        self.is_end = False 
        self.img_width = img_width  
        self.img_height = img_height 

   
    def start(self, file_name, freq):

        four_cc = cv.VideoWriter_fourcc(*'mp4v')
        img_size = (self.img_width, self.img_height)  


        self.video_writer = cv.VideoWriter()
        self.video_writer.open(file_name, four_cc, freq, img_size, True)

      
    def record(self, img):
        if self.is_end is False:
            self.video_writer.write(img)


    def end(self):
        self.is_end = True
        self.video_writer.release()

import os
import mmcv
def main_waymo():
    rm = RecordMovie(200, 200)
    rm.start("test_waymo.mp4", 10)
    # base_path = 'test/anchor_traintest_noflip_1.0/Fri_Jun__3_17_10_33_2022/show_dirs/testing_camera/image_0/'
    files = os.listdir('/mount/data/lsbevv2/vis')
    for i in range(320):
       
        imgs = cv.imread(os.path.join('/mount/data/lsbevv2/vis', f'a_{i}.png'))
        print(i)
        print(imgs.shape)
        rm.record(imgs)
    rm.end()

if __name__ == '__main__':
    #main_nuscenes()
    main_waymo()