# Author: @Marsch
import numpy


def dense_to_one_hot(labels_dense, num_classes=7):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    index = list(index_offset + labels_dense.ravel())
    labels_one_hot.flat[index] = 1
    return labels_one_hot


def one_hot_transform(vector):
    labels = []
    for label in vector:
        label = numpy.array(label)
        labels.append(dense_to_one_hot(label))
    return numpy.array(labels)


class DataSet(object):
    def __init__(self, vectors, labels, values=[]):
        self._num_examples = vectors.shape[0]
        self._vectors = vectors
        self._labels = labels
        self._values = values
        self._index_batch_start = 0

    @property
    def vectors(self):
        return self._vectors

    @property
    def values(self):
        return self._values

    @property
    def labels(self):
        # @TODO: validate whether length of labels equals to length of vectors
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_batch_start
        self._index_batch_start += batch_size
        if self._index_batch_start > self._num_examples:  # finish 1 epoch
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._vectors = self._vectors[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_batch_start = batch_size
        end = self._index_batch_start
        return self._vectors[start:end], self._labels[start:end]

    def shuffle_data(self):
        '''
        Shuffle the data
        '''
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._vectors = self._vectors[perm]
        self._labels = self._labels[perm]
