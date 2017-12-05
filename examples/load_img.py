import numpy as np
import cv2
from matplotlib import pyplot as plt
import skimage.transform

# # Load an color image in grayscale
# img = cv2.imread('../data/test.jpg',0) # 0 = gray
img = cv2.imread('data/no-left-turn.jpg')
# print(img.shape)
width, height = img.shape[:2]
_min = min(width, height)
img2 = cv2.resize(img[int(width/2 - _min/2):int(width/2 + _min/2), int(height/2 - _min/2):int(height/2 + _min/2)].copy(), (280, 280))
# img2 = cv2.resize(img, (280, 280), interpolation = cv2.INTER_CUBIC)
# print(img.shape)
# window = img[0:500, 0:500].copy()
# img = skimage.transform.resize(img, (5,5))

# cv2.namedWindow('Khi', cv2.WINDOW_NORMAL)
# cv2.imshow('Khi',img)
# k = cv2.waitKey(0)
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()
# elif k == ord('s'): # wait for 's' key to save and exit
#     cv2.imwrite('khi.png',img)
#     cv2.destroyAllWindows()
    
plt.imshow(img2, cmap = 'gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

