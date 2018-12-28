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


fname ='project_video.mp4' 
# fname ='harder_challenge_video.mp4' 
clip1 = VideoFileClip(fname)#.subclip(40,41)
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile('result_'+fname, audio=False)
