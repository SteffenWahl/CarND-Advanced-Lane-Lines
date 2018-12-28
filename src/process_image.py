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


mtx = np.load('data/mtx.npy') 
dist = np.load('data/dist.npy') 

left_lane = Line()
right_lane = Line()

ym_per_pix = 75/350 # meters per pixel in y dimension
xm_per_pix = 3.7/400 # meters per pixel in x dimension

# this is the image processing pipeline
def process_image(img, filename=None, reset=False):
    if reset is True:
        left_lane = Line()
        right_lane = Line()

    #1: undistort image
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    if filename is not None: 
        cv2.imwrite('output_images/undistorted_'+filename, cv2.cvtColor(undist, cv2.COLOR_RGB2BGR))

    #1: warp image
    warped = unwarp_image(undist)
    if filename is not None: 
        out_img = warped
        cv2.imwrite('output_images/warped_'+filename, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))


    try:
        #3: Binarize image
        binary = binarize_image(warped)
        if filename is not None: 
            out_img = np.dstack((binary, binary, binary))
            cv2.imwrite('output_images/binary_'+filename, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    
        #4: Fit polynomial 
        out, left_fitx, right_fitx, ploty_left, ploty_right = fit_polynomial(binary, left_lane, right_lane)
        if filename is not None: 
            out_img = out
            cv2.imwrite('output_images/polynomial_'+filename, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
   
        #5: Do a sanity check 
        val = sanity_check(left_lane, right_lane, xm_per_pix, ym_per_pix)
        fontColor              = (255,255,255)
        if not val:
            fontColor              = (255,0,0)
            print('Sanity check went wrong')

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty_left]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty_right])))])
        pts = np.hstack((pts_left, pts_right))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
        #calculate centerline
        l_range = left_lane.y_range()
        r_range = right_lane.y_range()
        all_range = [ max(l_range[0],r_range[0]),min(l_range[1],r_range[1])]
        xl,yl = left_lane.get_vals(all_range)
        xr,yr = right_lane.get_vals(all_range)
        avg_x = np.zeros_like(xl)

        for i in range(0,len(xl)):
            avg_x[i] = (xl[i]+xr[i])/2

        c_lane_px = Line()
        c_lane_px.update_force(avg_x,yl)
        x,y = c_lane_px.get_vals((0,color_warp.shape[0]-1))
        color_warp[np.int_(y),np.int_(x)] = [255,0,0]
        
        c_lane = Line()
        c_lane.update_force(avg_x*xm_per_pix,yl*ym_per_pix)
        radi = c_lane.get_radius()

        #calculate offset of vehicle position in lane
        car_img = np.array([[img.shape[1]/2,img.shape[0]-1]],dtype='float32').reshape((-1, 1, 2))
        car_warped = cv2.perspectiveTransform(car_img,M)
        x_line_car = c_lane.get_val(car_warped[0][0][1]*ym_per_pix)
        diff = x_line_car-car_warped[0][0][0]*xm_per_pix
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 0.7, newwarp, 0.3, 0)

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        text_pos= (100,50)
        fontScale              = 1
        lineType               = 2
        
        cv2.putText(result,'Radius: ' + str(int(radi)) + ' m;   Diff: '+str(round(diff,2))+' m', 
            text_pos, 
            font, 
            fontScale,
            fontColor,
            lineType) 

        return result 
    except Nothing:
        return undist
