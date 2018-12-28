import os
import numpy as np
import cv2


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
    # if d < 51:
    #     ret[0] = 1
    # else:
    #     ret[1] = 1
    # return ret

    map = {
        14: 1,
        34: 2,
        53: 2,
        33: 3,
        54: 3,
        51: 4,
        52: 5,
        17: 6,
        0: 7,
        99: 8,
        32: 8}

    ret = np.zeros(len(map))
    ret[map[d]] = 1
    return ret


# print(to_one_hot(61))
# print(to_one_hot(0))


IMG_SIZE = 28


def load_data(data_dir):
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
            # images.append(skimage.transform.resize(
            # skimage.data.imread(f), (28, 28)))
            # images.append(np.reshape(cv2.resize(cv2.imread(f, 0), (28, 28)),
            # -1)) #TODO upgrade to use all 3 chanels
            try:
                img = cv2.resize(cv2.imread(f), (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(img.flatten())
                labels.append(to_one_hot(int(d)))
            except Exception:
                pass

    return images, labels
