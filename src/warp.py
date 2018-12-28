import numpy as np
import cv2

src = np.float32([[274,680],[598,450],[683,450],[1030,680]])
dst = np.float32([[300,400],[300,50],[700,50],[700,400]])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

def unwarp_image(img):
    warped = cv2.warpPerspective(img, M, (900,450), flags=cv2.INTER_LINEAR)
    return warped

def wrap_image(img, shape):
    warped = cv2.warpPerspective(img, Minv, shape, flags=cv2.INTER_LINEAR)
    return warped


