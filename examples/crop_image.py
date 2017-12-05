import os
import skimage.data
import skimage.transform
import numpy as np
import cv2
import pickle

def crop_img(data_dir, out_dir):
    images = []
    file_names = [os.path.join(data_dir, f) 
                  for f in os.listdir(data_dir) if f.endswith(".ppm") or f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")]
    print(len(file_names))

    # For each label, load it's images and add them to the images list.
    # And add the label number (i.e. directory name) to the labels list.
    for no, f in enumerate(file_names):
        img = cv2.imread(f)
        size = min(img.shape[:2])
        print(size)
        stride = 100
        for i in range(int(size/stride)):
            for j in range(int(size/stride)):
                path = out_dir + "/a" + str(no) + "_" + str(i) + "_" +  str(j) + ".jpg"
                print(path)
                cv2.imwrite(path, img[i*stride:i*stride+stride, j*stride:j*stride+stride])


data_dir = "data/raw/training/99"
out_dir = "data/raw/training/negative"

crop_img(data_dir, out_dir)


