import cv2
import numpy as np
#import skimage as ski

array = np.zeros((150,150), np.float32)

array[int(150/2),int(150/2)] = 1

g = cv2.GaussianBlur(array, (75,75), 0)
g2 = g/g.max()*255

cv2.imshow('gaussian', g2.astype(np.uint8))
cv2.waitKey(0)