'''
Data directory tree must look like:
input_dir/
    00000/
        img1.jpg
        img2.jpg
    00001/
        img1.jpg
        img2.jpg
    ...

'''
import os
import numpy as np
import cv2
import pickle
from utils.base import DataSet
import logging as logger


def to_one_hot(d):
    # sign: 0,...,90
    # not -sign: 99
    ret = np.zeros(2)
    if d < 91:
        ret[0] = 1
    else:
        ret[1] = 1
    return ret


class PickleDataSet():
    def __init__(self, pickle_dir):
        self.pickle_filenames = os.listdir(pickle_dir)
        self.epoch = 0
        self.num_pickle = len(self.pickle_filenames)
        self.current_pickle_idx = 0
        # every time we load an instance of DataSet, we shuffle it
        self.current_pickle = pickle.load(
            self.pickle_filenames[self.current_pickle_idx]).shuffle_data()

    def next_batch(self, batch_size):
        # in case finishing reading all data-points of a DataSet object
        if self.current_pickle._index_batch_start + batch_size < \
                self.current_pickle._num_examples:
            return self.current_pickle.next_batch()
        else:  # turn into another pickle
            self.current_pickle_idx = self.current_pickle_idx + 1
            if self.current_pickle_idx == self.num_pickle:  # new epoch
                self.current_pickle_idx = 0
                self.epoch = self.epoch + 1

            self.current_pickle = pickle.load(
                self.pickle_filenames[self.current_pickle_idx]).shuffle_data()
            self.next_batch(batch_size)


def possible_batch_size(vector_length,
                        lable_length,
                        RAM=8,
                        RAM_available=0.4):
    '''
    Caculate the maximum number of datapoints that RAM can handle
    '''
    from sys import getsizeof
    vector_memory = getsizeof([0] * vector_length)
    lable_memory = getsizeof([0] * lable_length)
    datapoint_memory = ((vector_memory + lable_memory) / 1024 / 1024 / 1024)
    return (RAM * RAM_available) / datapoint_memory


def dumb_img_dataset(input_dir,
                     pickle_dir="data/pickle/dataset/",
                     img_width=56,
                     img_height=56,
                     num_batch=10):
    '''
    Divide the whole dataset into *num_batch* parts
    Pickle these parts into the *pickle_dir*
    '''

    def pickle_batch(data_batch, idx):
        labels = []
        vectors = []
        for tup in data_batch:
            try:
                f = tup[0]
                d = tup[1]
                # @TODO: replace by a feature extraction method
                img = cv2.resize(cv2.imread(f), (img_height, img_width))
                vectors.append(img)
                labels.append(to_one_hot(int(d)))
            except Exception as e:
                # logger.warning(e) # debug purpose only
                pass

        dataset = DataSet(np.array(vectors), np.array(labels))

        with open(pickle_dir + str(idx) + ".pkl", 'wb') as f:
            pickle.dump(dataset, f)

    # Get list of tuple filename and label
    directories = [d for d in os.listdir(input_dir)
                   if os.path.isdir(os.path.join(input_dir, d))]
    data = np.array([])  # list of tuple(file_name, parent_dir)
    for d in directories:
        label_dir = os.path.join(input_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm") or
                      f.endswith(".jpg") or
                      f.endswith(".png") or
                      f.endswith(".jpeg")]
        for f in file_names:
            data = np.append(data, (f, d))

    # Shuffle the whole dataset
    perm = np.arange(len(data))
    np.random.shuffle(perm)
    data = data[perm]

    # batch_size = int(len(data) / num_batch)
    batch_size = int(possible_batch_size(img_width * img_height, 2, 8, 0.5))
    num_batch = int(len(data) / batch_size)

    for idx in range(num_batch):
        pickle_batch(data[idx * batch_size:(idx + 1) * batch_size], idx)

    pickle_batch(data[num_batch * batch_size + 1:], "last")

    logger.warning("possible_batch_size: %s" % batch_size)


if __name__ == '__main__':
    # print(possible_batch_size(56 * 56, 2))
    dumb_img_dataset('data/raw/training/')
