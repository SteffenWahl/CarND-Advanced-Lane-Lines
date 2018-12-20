import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def binarize_image(img):
    for i in range(0,img.shape[0]):
        for ii in range(0,img.shape[1]):
            if (img[i,ii,0] == img[i,ii,1]) & (img[i,ii,1] ==img[i,ii,2]) & (img[i,ii,0] != 0):
                img[i,ii,0] = img[i,ii,0]-1

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #plt.figure(2)
    #plt.imshow(gray,cmap='gray')
    #plt.title('gray')

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h = hls[:,:,0]
    l = hls[:,:,1]
    s = hls[:,:,2]

    #plt.figure(3)
    #plt.imshow(h,cmap='gray')
    #plt.title('H')
    #plt.figure(4)
    #plt.imshow(l,cmap='gray')
    #plt.title('L')
    #plt.figure(5)
    #plt.imshow(s,cmap='gray')
    #plt.title('S')

    s_thresh=(170, 255)
    sx_thresh=(20, 100)
#   sobel this one
    sobelx = cv2.Sobel(s, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    kernel = np.ones((1,51))
    kernel_left = kernel.copy()
    kernel_left[0,25::] = 0
    kernel_left = kernel_left/np.sum(kernel_left) 
    kernel_right = kernel.copy()
    kernel_right[0,0:26] = 0 
    kernel_right = kernel_right/np.sum(kernel_right) 
    sxleft = cv2.filter2D(scaled_sobel, -1, kernel_left)
    sxright = cv2.filter2D(scaled_sobel, -1, kernel_right)
    #plt.figure(15)
    #plt.imshow(sxleft,cmap='gray')
    #plt.title('SX left')

    #plt.figure(16)
    #plt.imshow(sxright,cmap='gray')
    #plt.title('SX right')

    color_sx = np.dstack(( np.zeros_like(sxright), sxright, sxleft))*5
    #plt.figure(20)
    #plt.imshow(color_sx)
    #plt.title('color sx')

    res = np.zeros_like(scaled_sobel)
    res[(sxleft > 5) & (sxright > 5) & (s > 120) & (h < 50)] = 255
    #plt.figure(30)
    #plt.imshow(res, cmap='gray')
    #plt.title('Result')


    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
 
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s)
    s_binary[(s >= s_thresh[0]) & (s <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    #plt.figure(6)
    #plt.imshow(color_binary)
    #plt.title('Binary')

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 255
    blur = cv2.GaussianBlur(combined_binary,(15,15),0)
    #plt.figure(7)
    #plt.imshow(blur, cmap='gray')
    #plt.title('Combined Blured')

    return res


src = np.float32([[274,680],[598,450],[683,450],[1030,680]])
dst = np.float32([[200,400],[200,50],[600,50],[600,400]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

ym_per_pix = 30/350 # meters per pixel in y dimension
xm_per_pix = 3.7/400 # meters per pixel in x dimension
def unwrap_image(img):
    warped = cv2.warpPerspective(img, M, (700,450), flags=cv2.INTER_LINEAR)
    return warped

def wrap_image(img, shape):
    warped = cv2.warpPerspective(img, Minv, shape, flags=cv2.INTER_LINEAR)
    return warped

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin # Update this
        win_xleft_high = leftx_current + margin # Update this
        win_xright_low = rightx_current - margin # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds= ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

# class RingBuffer copied from https://www.oreilly.com/library/view/python-cookbook/0596001673/ch05s19.html
class RingBuffer:
    """ class that implements a not-yet-full buffer """
    def __init__(self,size_max):
        self.max = size_max
        self.xdata = []
        self.ydata = []

    class __Full:
        """ class that implements a full buffer """
        def append(self, x, y):
            """ Append an element overwriting the oldest one. """
            self.xdata[self.cur] = x
            self.ydata[self.cur] = y
            self.cur = (self.cur+1) % self.max
        def get(self):
            """ return list of elements in correct order """
            return self.xdata[self.cur:]+self.xdata[:self.cur], self.ydata[self.cur:]+self.ydata[:self.cur]

    def append(self, x, y):
        """append an element at the end of the buffer"""
        self.xdata.append(x)
        self.ydata.append(y)
        if len(self.xdata) == self.max:
            self.cur = 0
            # Permanently change self's class from non-full to full
            self.__class__ = self.__Full

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.xdata, self.ydata 
        

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.buffer = RingBuffer(10)
        self.notdetected = 0

    def update(self, x, y):
        self.allx = x
        self.ally = y
        if len(x) < 50:
            self.detected = False
            self.notdetected = self.notdetected + 1
        else:
            self.detected = True
            self.current_fit = np.polyfit(y, x, 2)
            self.notdetected = 0
        self.buffer.append(x,y)
        ax, ay = self.buffer.get()
        aax = np.concatenate(ax)
        aay = np.concatenate(ay)
        self.best_fit = np.polyfit(aay,aax,2)

    def update_force(self, x, y):
        self.allx = x
        self.ally = y
        self.detected = True
        self.current_fit = np.polyfit(y, x, 2)
        self.notdetected = 0
        self.buffer.append(x,y)
        ax, ay = self.buffer.get()
        aax = np.concatenate(ax)
        aay = np.concatenate(ay)
        self.best_fit = np.polyfit(aay,aax,2)


    def y_range(self):
        ax, ay = self.buffer.get()
        aay = np.concatenate(ay)
        return (min(aay),max(aay))

    def get_vals(self, rang=None):
        if rang is None:
            rang = self.y_range()
        ploty = np.linspace(rang[0], rang[1], rang[1]-rang[0] )
        fit = self.best_fit
        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
        return fitx, ploty            

left_lane = Line()
right_lane = Line()

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    left_lane.update(leftx,lefty)
    right_lane.update(rightx,righty)

    # Generate x and y values for plotting
    try:
        left_fitx,ploty_left = left_lane.get_vals()
        right_fitx,ploty_right = right_lane.get_vals()
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty_left**2 + 1*ploty_left
        right_fitx = 1*ploty_right**2 + 1*ploty_right

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fitx, right_fitx, ploty_left, ploty_right


mtx = np.load('data/mtx.npy') 
dist = np.load('data/dist.npy') 


#y_eval = np.max(ploty)
#left_curverad = (1+(2*left_fit_cr[0]*y_eval*ym_per_pix+left_fit_cr[1])**2)**(3/2)/abs(2*left_fit_cr[0])
#right_curverad = (1+(2*right_fit_cr[0]*y_eval*ym_per_pix+right_fit_cr[1])**2)**(3/2)/abs(2*right_fit_cr[0])



def process_image(img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    plt.figure(701)
    plt.imshow(undist)
    plt.title('Undistorted')
    try:
    
        binar = binarize_image(undist)
        plt.figure(710)
        plt.imshow(binar, cmap='gray')
        plt.title('Binary')
     
        unwrap = unwrap_image(binar)
        plt.figure(70)
        plt.imshow(unwrap, cmap='gray')
        plt.title('Unwrapped')
    
        out, left_fitx, right_fitx, ploty_left, ploty_right = fit_polynomial(unwrap)
        plt.figure(90)
        plt.imshow(out, cmap='gray')
        plt.title('Unwrapped')
    
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(unwrap).astype(np.uint8)
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
        c_lane = Line()
        c_lane.update_force(avg_x,yl)
        xl,yl = c_lane.get_vals()
        color_warp[np.int_(xl),np.int_(yl)] = [255,0,0]

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
 
        return result 
    except Nothing:
        return undist


doImage =True
if doImage is True:
    foldername = 'test_images/' 
    files = os.listdir(foldername)
    files = [
    #'test1.jpg'
    #'test2.jpg'
    #'test3.jpg'
    #'test4.jpg'
    #'test5.jpg'
    'straight_lines1.jpg'
    #'straight_lines2.jpg'
    ]
    
    for filename in files:
        filename_full = foldername + filename
        img = mpimg.imread(filename_full)
        plt.figure(1)
        plt.imshow(img)
        plt.title('original')
        
        result = process_image(img)
        plt.figure(55) 
        plt.imshow(result)
        plt.title('Final result')
        
        #write output image
        cv2.imwrite('output_images/result_'+filename, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
else:    

    fname ='project_video.mp4' 
    #fname ='harder_challenge_video.mp4' 
    clip1 = VideoFileClip(fname).subclip(0,1)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile('annot_'+fname, audio=False)
    
plt.show()
