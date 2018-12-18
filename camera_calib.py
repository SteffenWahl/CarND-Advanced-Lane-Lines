import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# prepare object points
nx = 9 #TODO: enter the number of inside corners in x
ny = 6 #TODO: enter the number of inside corners in y
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = np.zeros((nx*ny,3))
for y in range(ny):
    for x in range(nx):
        objpoints[x+y*nx] = [x,y,0]

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
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

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    
    
    # If found, draw corners
    if ret == True:
        print('corners found')
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        
        corners_subpix = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        plt.imshow(img)
        cv2.imwrite('camera_cal_output/'+filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        all_objpoints.append(objpoints)
        all_imgpoints.append(corners_subpix)
    else:
        print("No corners found")


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(np.array(all_objpoints), np.array(all_imgpoints), img_shape[:-1], None, None)

np.save('data/mtx.dat', mtx)
np.save('data/dist.dat', dist)
# load with: mtx = np.loadtxt('data/mtx.dat')
# load with: mtx = np.loadtxt('data/dist.dat')

for filename in files:
    filename_full = foldername + filename
    img = cv2.imread(filename_full)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('camera_cal_output/undist_'+filename, cv2.cvtColor(undist, cv2.COLOR_RGB2BGR))

plt.show()


