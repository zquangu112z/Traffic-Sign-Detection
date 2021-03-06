from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import argparse
import sys
import tempfile
import pickle
from sklearn.utils import shuffle
# import cv2
import numpy as np
from utils import load_data
import tensorflow as tf

TRAIN_DATA_DIR = "data/raw/training"
TEST_DATA_DIR = "data/raw/testing"
# CNN_MODEL_DIR = "model/CNN/3cnn_test.ckpt" # 3 lop conv, 10 epoch
CNN_MODEL_DIR = "model/CNN/3cnn_4conv_30ep_DEC14_2017.ckpt"  # 0.9688
PICKLE_IMGS_DIR = "data/pickle/train_imgs.pkl"
PICKLE_LABELS_DIR = "data/pickle/test_labels.pkl"
NUM_CLASSES = 2
IMG_SIZE = 56


class MyCNN():
    def __init__(self, x):
        self.y_conv, self.keep_prob = self.deepnn(x)
        self.num_datapoint = len(images)
        self.batch_size = 100
        self.num_epochs = 30
        pass

    def deepnn(self, x):
        with tf.name_scope('reshape'):
            x_image = x
            # x_image = tf.placeholder([-1, 28, 28, 3])

        # First convolutional layer - maps an RGB image to 32 feature maps.
        with tf.name_scope('conv1'):
            W_conv1 = self.weight_variable([5, 5, 3, 32])
            b_conv1 = self.bias_variable([32])
            h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            h_pool1 = self.max_pool_2x2(h_conv1)

        # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            W_conv2 = self.weight_variable([5, 5, 32, 64])
            b_conv2 = self.bias_variable([64])
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            h_pool2 = self.max_pool_2x2(h_conv2)

        # Third convolutional layer -- maps 64 feature maps to 64.
        with tf.name_scope('conv3'):
            W_conv3 = self.weight_variable([5, 5, 64, 64])
            b_conv3 = self.bias_variable([64])
            h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3) + b_conv3)

        # Forth convolutional layer -- maps 64 feature maps to 64.
        with tf.name_scope('conv4'):
            W_conv4 = self.weight_variable([5, 5, 64, 64])
            b_conv4 = self.bias_variable([64])
            h_conv4 = tf.nn.relu(self.conv2d(h_conv3, W_conv4) + b_conv4)

        # Third pooling layer.
        with tf.name_scope('pool3'):
            h_pool3 = self.max_pool_2x2(h_conv4)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28
        # image is down to 7x7x64 feature maps -- maps this to 1024 features.
        with tf.name_scope('fc1'):
            W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
            b_fc1 = self.bias_variable([1024])

            h_pool3_flat = tf.reshape(h_pool3, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

        # Dropout - controls the complexity of the model,
        # prevents co-adaptation of features.
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Map the 1024 features to NUM_CLASSES classes, one for each digit
        with tf.name_scope('fc2'):
            W_fc2 = self.weight_variable([1024, NUM_CLASSES])
            b_fc2 = self.bias_variable([NUM_CLASSES])

            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        return y_conv, keep_prob

    def conv2d(self, x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(self, shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def train():
        



def main(_):
    # Import data
    # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    try:
        with open(PICKLE_IMGS_DIR, 'rb') as f:
            images = pickle.load(f)
        with open(PICKLE_LABELS_DIR, 'rb') as f:
            labels = pickle.load(f)
    except Exception:
        logging.warning("No pickled data. Loading data from local...")
        images, labels = load_data(TRAIN_DATA_DIR)
        with open(PICKLE_IMGS_DIR, 'wb') as f:
            pickle.dump(images, f)
        with open(PICKLE_LABELS_DIR, 'wb') as f:
            pickle.dump(labels, f)

    # evaluation set
    num_validation = 4000
    images, labels = shuffle(images, labels, random_state=0)
    images_eval, labels_eval = images[:num_validation], labels[:num_validation]
    images, labels = images[num_validation:], labels[num_validation:]

    num_datapoint = len(images)
    batch_size = 100
    num_epochs = 30

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # Build the graph for the deep net
    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    saver = tf.train.Saver()
    count = 0
    count_max = 5
    last_accuracy = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # epoch
        for i in range(num_epochs):
            images, labels = shuffle(images, labels, random_state=0)

            for batch_idx in range(int(num_datapoint / batch_size)):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                # batch = mnist.train.next_batch(50)
                if batch_idx % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: images[start_idx:end_idx],
                        y_: labels[start_idx:end_idx],
                        keep_prob: 1.0})
                    print('Epoch %d, batch_idx %d, training accuracy %g' %
                          (i, batch_idx, train_accuracy))
                train_step.run(feed_dict={x: images[start_idx:end_idx],
                                          y_: labels[start_idx:end_idx],
                                          keep_prob: 0.8})

            # Evaluation
            count = count + 1
            accuracy_ = accuracy.eval(feed_dict={
                x: images_eval,
                y_: labels_eval,
                keep_prob: 1.0})
            if accuracy_ > last_accuracy:
                # lan train cho ra ket qua tot hon lan truoc
                count = 0
                last_accuracy = accuracy_
                # luu lai model tot nhat hien tai
                saver.save(sess, CNN_MODEL_DIR)
                print('Saved snapshot at epoch: %d' % i)
            elif count == count_max:
                print("Cannot improve the model. Finish training at epoch %d..." % i)
                return

        # save model
        # saver.save(sess, CNN_MODEL_DIR)


def evaluate():
    x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    y_conv, keep_prob = deepnn(x)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, CNN_MODEL_DIR)
        images, labels = load_data(TEST_DATA_DIR)
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: images, y_: labels, keep_prob: 1.0}))


def predict(img):
    x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])
    y_conv, keep_prob = deepnn(x)
    predict = tf.argmax(y_conv, 1)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, CNN_MODEL_DIR)
        print('Label %g' % sess.run(
            predict, feed_dict={x: img, keep_prob: 1.0}))


if __name__ == '__main__':
    # Train
    # tf.app.run(main=main, argv=[sys.argv[0]])

    # Evaluation
    evaluate()

    # Predict
    # img = cv2.imread('data/00011_00000.ppm')
    # img = cv2.resize(img, (28,28))
    # predict([img])
