import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from src.binarize_image import *
from src.warp import *
from src.find_lane_pixels import *
from src.Line import *
from src.search_around_poly import *
from src.fit_polynomial import *
from src.sanity_check import *
from src.process_image import *


save_imgs = True
img_counter = 0

left_lane = Line()
right_lane = Line()




doImage =True
if doImage is True:
    foldername = 'test_images/' 
    files = os.listdir(foldername)
#    files = [
#    'test1.jpg'
#    'test2.jpg'
#    'test3.jpg'
#    'test4.jpg'
#    'test5.jpg'
#    'straight_lines1.jpg'
#    'straight_lines2.jpg'
#    ]
#    
    for filename in files:
        filename_full = foldername + filename
        print('Processing '+filename)
        img = mpimg.imread(filename_full)
        plt.figure(1)
        plt.imshow(img)
        plt.title('original')
        
        #reset Filters, when applying on single images

        result = process_image(img, filename, True)
        plt.figure(55) 
        plt.imshow(result)
        plt.title('Final result')
        
        #write output image
        cv2.imwrite('output_images/result_'+filename, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
else:    

    fname ='project_video.mp4' 
   # fname ='challenge_video.mp4' 
    clip1 = VideoFileClip(fname)#.subclip(40,41)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile('annot_'+fname, audio=False)
