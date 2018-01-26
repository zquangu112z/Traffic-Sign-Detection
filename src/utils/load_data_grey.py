import os
import numpy as np
import cv2

NUM_CLASSES = 8


def to_one_hot(d):
    map = {
        14: 0,
        34: 1,
        53: 1,
        33: 2,
        54: 2,
        51: 3,
        52: 4,
        17: 5,
        0: 6,
        99: 7,
        32: 7}

    ret = np.zeros(NUM_CLASSES)
    ret[map[d]] = 1
    return ret


'''
    1: 'Dung',
    2: 'Re trai',
    3: 'Re phai',
    4: 'Cam re trai',
    5: 'Cam re phai',
    6: 'Mot chieu',
    7: 'Toc do toi da',
    8: 'Others'
'''


IMG_SIZE = 28


def load_data_grey(data_dir):
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
                      f.endswith(".jpg") or f.endswith(".png") or
                      f.endswith(".JPEG")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            # images.append(np.reshape(cv2.resize(cv2.imread(f, 0), (28, 28)),
            # -1)) #TODO upgrade to use all 3 chanels
            try:
                images.append(np.reshape(cv2.resize(
                    cv2.imread(f, 0), (28, 28)), -1))
                labels.append(to_one_hot(int(d)))
            except Exception as e:
                raise e

    return images, labels
