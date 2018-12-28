import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# input parameters
nx = 9 #TODO: enter the number of inside corners in x
ny = 6 #TODO: enter the number of inside corners in y

# prepare object points
objpoints = np.zeros((nx*ny,3), np.float32)
objpoints[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

all_objpoints = []
all_imgpoints = []

foldername = 'camera_cal/' 

files = os.listdir(foldername)
for filename in files:
    filename_full = foldername + filename
    print('Processing File ' + filename_full)
    img = cv2.imread(filename_full)

    img_shape = img.shape
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    # If corners found, draw corners and add to arrays
    if ret == True:
        print('corners found')
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
       
        cv2.imwrite('camera_cal_output/annot_'+filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        all_objpoints.append(objpoints)
        all_imgpoints.append(corners)
    else:
        print("No corners found")

# calibrate
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(np.array(all_objpoints), np.array(all_imgpoints), img_shape[:-1], None, None)

#save calibration parameters
np.save('data/mtx', mtx)
np.save('data/dist', dist)
# load with: mtx = np.loadtxt('data/mtx')
# load with: mtx = np.loadtxt('data/dist')

# undistort all calibration images and save them
for filename in files:
    filename_full = foldername + filename
    img = cv2.imread(filename_full)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('camera_cal_output/undist_'+filename, cv2.cvtColor(undist, cv2.COLOR_RGB2BGR))

