


# Advanced Lane Finding Project

This markup file describes the Advanced Lane Finding Project.

[//]: # (Image References)

[img_calib_in]: ./camera_cal/calibration2.jpg "Original"
[img_calib_annot]: ./camera_cal_output/annot_calibration2.jpg "Annotated"
[img_calib_undist]: ./camera_cal_output/undist_calibration2.jpg "Undistorted"

[img_in]: ./test_images/test2.jpg "Original"
[img_undistorted]: ./output_images/undistorted_test2.jpg "Undistorted"
[img_warped]: ./output_images/warped_test2.jpg "Warped"
[img_binary]: ./output_images/binary_test2.jpg "Binary"
[img_poly]: ./output_images/polynomial_test2.jpg "polynomial"
[img_result]: ./output_images/result_test2.jpg "Result"


## Camera Calibration

The code for the camera calibration can be found in file `./camera_calib.py`. 

Camera calibration is done in lines 20 to 44.
For each image in folder `./camera_cal`, the folowing steps are done:

1. The image is opened. (line 22)
2. The image is converted to grayscale image. (line 28)
3. Chessboard corners are found. (line 35)
4. If chessboard corners are found, the object- and image-points are stored in a container. (line 38, 39)

Then, with all the image- and object-points, the calibration parameters are calculated (line 44) and the camera matrix and the distortion parameters are stored in files `./data/mtx.npy` and `./data/dist.npy`. (line 47, 48) 

The following figures show a original calibration image:
![Original image][img_calib_in]
a calibration image with detected chessboard pattern: 
![Image with chessboard pattern][img_calib_annot]
and a calibration image after undistortion with the calculated distortion parameters:
![Undistorted image][img_calib_undist]
 

## Pipeline (single images)

The pipeline for single images (as well as for videos) is given in file `.src/process_image.py`. The pipeline consist of the following steps:
1. Undistort input image
2. Warp image (Transformation to 'birds eye')
3. Binarize image
4. Fit polynomial
5. Sanity check
6. Calculation of curvature of lane
7. Calculation of vehicle offset in lane

The pipeline is described in more detail by using the following input image as an example:
![Original image][img_in]

### 1. Undistort input image

The undistortion of the image is done in file `./src/process_image.py`. First the calibration parameters are loaded by 
```python
mtx = np.load('data/mtx.npy') 
dist = np.load('data/dist.npy') 
```
Then the image is undistorted by:
``` python
undist = cv2.undistort(img, mtx, dist, None, mtx)
```
The following firgure shows an undistorted image
![Undistorted image][img_undistorted]

### 2. Warp image (Transformation to 'birds eye')

The transformation of the input image to a birds eye view is done in function `warp_image(img)` file `./src/warp.py`. As well as the unwarping of an image.
The transformation matrices is defined with:
```python
src = np.float32([[274,680],[598,450],[683,450],[1030,680]])
dst = np.float32([[300,400],[300,50],[700,50],[700,400]])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
``` 

The image is then warped with:
``` python
warped = cv2.warpPerspective(img, M, (900,450), flags=cv2.INTER_LINEAR)

``` 

The following figure shows the input image after the birds eye transform:
![Warped image][img_warped]

### 3. Binarize image
The binarization is done in file `./src/binarize_image.py`. 
At first, the input image is transformed into HSV color space. 
``` python
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
``` 

Then an edge image is created by application of a sobel filter on v and s chanel. The absolute values of the images are added and the resulting image is scaled:
``` python
sobelx_v = cv2.Sobel(v, cv2.CV_64F, 1, 0)
sobelx_s = cv2.Sobel(s, cv2.CV_64F, 1, 0)
abs_sobelx = np.absolute(sobelx_v)+np.absolute(sobelx_s)
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
``` 
Then 2 convolution box kernels are created. One for the left edge and one for the right edge. These kernels are then applied:
``` python
kernel = np.ones((1,31))
kernel_left = kernel.copy()
kernel_left[0,15::] = 0
kernel_left = kernel_left/np.sum(kernel_left) 
kernel_right = kernel.copy()
kernel_right[0,0:26] = 0 
kernel_right = kernel_right/np.sum(kernel_right) 
sxleft = cv2.filter2D(scaled_sobel, -1, kernel_left)
sxright = cv2.filter2D(scaled_sobel, -1, kernel_right)
``` 
Finally a binary image is created by applying some thresholds on the edge image, and the s and v channel:
``` python
res = np.zeros_like(scaled_sobel)
res[(sxleft > 10) & (sxright > 10) & (((s > 120) & (v > 100)) | (v > 200))] = 255
``` 
The values are selected this way, so that a selected pixel has to be enveloped with a left and a right edge. Also the value of s space has to be high enough. In the s channel, the yellow color is detected the best. With the threshold in v channel, the white lane markings are detected the best.

The following figure shows the result of the binarized image:
![Binary image][img_binary]


### 4. Fit polynomial

The fitting of the polynomial is done in files `./src/fit_polynomial.py`, `./src/find_lane_pixels.py` and `search_around_poly.py`.

If, in the last execution of the pipeline the lanes were detected, then the polynomial is searched based on the polynomial in the last step in file `./src/search_around_poly.py`. Otherwise the polynomial is searched as seen in file `./src/find_lane_pixels.py`.
Both files are mainly copies of code snippets from the udacity lessons and quizzes.
The `./src/find_lane_pixels.py` works as follows:

1. First, a histogram of the lower half of the image is created.
2. The peaks in the left and the right half of the histogram are detected.
3. Then windows, x-centered around the peaks are created and the nonzero pixels inside this window are selected.
4. The windows are moved in y-direction and are x-centered around the mean of the detections of the last windows.

The `./src/search_around_poly.py` works almost the same way. But with the windows centered around the positions of the last fitted polyline.

Finally the polyline is fitted by using the function `update()` of file `./src/Line.py`. There a polyfit is done:
```python
self.best_fit = np.polyfit(aay,aax,2)
```

The result is shown in the following figure:
![Polynomial][img_poly]

### 5. Sanity check

The sanity check is shown in file `./src/sanity_check.py`.

There, the y-range of both lane markings is calculated and the y range, where both lines are detected are taken:
``` python
l_range = left_lane.y_range()
r_range = right_lane.y_range()
all_range = [ max(l_range[0],r_range[0]),min(l_range[1],r_range
``` 
Then the differences (in m) of the lane markins are calculated:
``` python
diff0 = abs(xl0-xr0)*xm_per_pix 
diff1 = abs(xl1-xr1)*xm_per_pix
``` 

If the differences of the near or far measurement differ too much, the sanity check return `False`.
``` python
if (diff0 < 2.7) | (diff0 > 4.7):
    return False 
if (diff1 < 3.2) | (diff1 > 4.2):
    return False
return True
``` 

Different threshold values for near and far are used because the far mesurements are not precise as the near measurements.

### 6. Calculation of curvature of lane

Lane curvature is calculated in file `./src/process_image.py`

Therefore both lane markings are averaged and a new line is fitted into the centerline:
``` python
l_range = left_lane.y_range()
r_range = right_lane.y_range()
all_range = [ max(l_range[0],r_range[0]),min(l_range[1],r_range[1])]
xl,yl = left_lane.get_vals(all_range)
xr,yr = right_lane.get_vals(all_range)
avg_x = np.zeros_like(xl)

for i in range(0,len(xl)):
    avg_x[i] = (xl[i]+xr[i])/2

c_lane = Line()
c_lane.update_force(avg_x*xm_per_pix,yl*ym_per_pix)
radi = c_lane.get_radius()
``` 

The radius is calculated in the file `./src/Line.py`:
``` python
rad = (1+(2*fit[0]*pos + fit[1])**2)**(3/2)/abs(2*fit[0])
``` 


### 7. Calculation of vehicle offset in lane

The calculation of the vehicle offset in the lane is done in file `./src/process_image.py`.

There, the lower centerpoint of the image is taken and transformed into birds eye view. Then, the difference to the centerline calculated in 7. is calculated. 

``` python
car_img = np.array([[img.shape[1]/2,img.shape[0]-1]],dtype='float32').reshape((-1, 1, 2))
car_warped = cv2.perspectiveTransform(car_img,M)
x_line_car = c_lane.get_val(car_warped[0][0][1]*ym_per_pix)
diff = x_line_car-car_warped[0][0][0]*xm_per_pix

``` 

### Result

The following figure shows the result of the pipeline:
![Result image][img_result]


##Video

The pipeline is applied to the video in file `main_video.py`.
The result  can be found  [here](./result_project_video.mp4) (`./result_project_video.mp4`)

##Discussion

### Problems / issues faced during implementation

Several issues were faced during implementation. These are:

* Selecting of algorithms and thresholds for binarization of input image. One issue with this was the different colors of lines (yellow and white), as well as the different surface colors of the roads. Shadows also made the binarization difficult.
* Definition of y-resolution. Since there was no real calibration image present, where one could map the pixels in y direction to some meter values, only a guess was possible.

### Where will the pipeline likely fail

The pipeline will fail, when

* the lane markings are too close to the grass strip as in test video `harder_challenge.mp4`.
* double lines are given, as shown in `harder_challenge.mp4`.
* lighting changes. For instance, when it is getting dark, the thresholds will not work anymore.

 
### What be done to make it more robust

There are some options to make the system more robust:

* First of all, the binarization could be improved. Threrefore, a lot of line data could be collected and labeled. These data could be used to create a better binarization algorithm by some machine learning approach.
* Binarization could also be made more robust, by detection of different road surfaces and or lighting. Then,  different algorithms and thresholds can be selected based on the surfaces or lighting.
* Another improvement can be the use of movement data from the vehicle. With this information, one could be able to transform the detected lane markings from the last image exactly to the next image.