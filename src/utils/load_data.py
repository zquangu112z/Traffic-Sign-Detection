import os
# import skimage.data
# import skimage.transform
import numpy as np
import cv2
import pickle
# from utils import jittering


def to_one_hot(d):
    # Warning sign: 0,..18
    # Priority signs: 19,20,21
    # Prohibitory signs: 22,...,32
    # Mandatory signs: 33, 39
    # Parking signs: 40,50

    # ret = np.zeros(6)
    # if d < 19:
    #     ret[0] = 1
    # elif d < 22:
    #     ret[1] = 1
    # elif d < 33:
    #     ret[2] = 1
    # elif d < 40:
    #     ret[3] = 1
    # elif d < 51:
    #     ret[4] = 1
    # else:
    #     ret[5] = 1
    # return ret

    # sign: 0,...,50
    # not -sign: 99
    ret = np.zeros(2)
    if d < 51:
        ret[0] = 1
    else:
        ret[1] = 1
    return ret


# print(to_one_hot(61))
# print(to_one_hot(0))

IMG_SIZE = 56


def load_data(data_dir):
    """Loads a data set and returns two lists:

    images: a list of Numpy 1-D arrays with dimesion 784,
            each representing an image in size 28x28.
    labels: a list of one-hot vector that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm") or
                      f.endswith(".jpg") or f.endswith(".png")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            # images.append(skimage.transform.resize(
            # skimage.data.imread(f), (28, 28)))
            # images.append(np.reshape(cv2.resize(cv2.imread(f, 0), (28, 28)),
            # -1)) #TODO upgrade to use all 3 chanels
            try:
                img = cv2.resize(cv2.imread(f), (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(to_one_hot(int(d)))
            except:
                pass

            # transformed_img = translatingImg(img, -10, 10)
            # images.append(transformed_img)
            # labels.append(to_one_hot(int(d)))

            # transformed_img = rescalingImg(img, 2)
            # images.append(transformed_img)
            # labels.append(to_one_hot(int(d)))

            # transformed_img = shearingImg(img, 100, 0)
            # images.append(transformed_img)
            # labels.append(to_one_hot(int(d)))

            # transformed_img = stretchingImg(img, 0.5)
            # images.append(transformed_img)
            # labels.append(to_one_hot(int(d)))

    return images, labels
