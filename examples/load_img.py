import numpy as np
import cv2
from matplotlib import pyplot as plt
import skimage.transform

# # Load an color image in grayscale
# img = cv2.imread('../data/test.jpg',0) # 0 = gray
img = cv2.imread('../data/test.jpg')
img = skimage.transform.resize(img, (5,5))
print(type(img))
print(img.shape)
print(img)
# cv2.namedWindow('Khi', cv2.WINDOW_NORMAL)
# cv2.imshow('Khi',img)
# k = cv2.waitKey(0)
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()
# elif k == ord('s'): # wait for 's' key to save and exit
#     cv2.imwrite('khi.png',img)
#     cv2.destroyAllWindows()
    
# plt.imshow(img, cmap = 'gray')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

