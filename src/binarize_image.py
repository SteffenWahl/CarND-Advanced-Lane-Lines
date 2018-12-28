import numpy as np
import cv2

def binarize_image(img):
    #use HSV image space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    # create edge image
    sobelx_v = cv2.Sobel(v, cv2.CV_64F, 1, 0)
    sobelx_s = cv2.Sobel(s, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx_v)+np.absolute(sobelx_s)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # extend edges to the left and the right by using asymetric box filter
    kernel = np.ones((1,31))
    kernel_left = kernel.copy()
    kernel_left[0,15::] = 0
    kernel_left = kernel_left/np.sum(kernel_left) 
    kernel_right = kernel.copy()
    kernel_right[0,0:26] = 0 
    kernel_right = kernel_right/np.sum(kernel_right) 
    sxleft = cv2.filter2D(scaled_sobel, -1, kernel_left)
    sxright = cv2.filter2D(scaled_sobel, -1, kernel_right)

    # create binary image by thresholding
    res = np.zeros_like(scaled_sobel)
    res[(sxleft > 10) & (sxright > 10) & (((s > 120) & (v > 100)) | (v > 200))] = 255

    
    return res

