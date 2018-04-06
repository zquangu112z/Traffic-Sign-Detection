from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tempfile
import pickle
from sklearn.utils import shuffle
from src.utils import load_data_grey
import tensorflow as tf
import logging as logger
import cv2
import numpy as np


TRAIN_DATA_DIR = "data/raw/training/augmented"
# TRAIN_DATA_DIR = "data/raw/testing"
TEST_DATA_DIR = "data/raw/testing"
PICKLE_IMGS_DIR = "data/pickle/train_imgs.pkl"
PICKLE_LABELS_DIR = "data/pickle/test_labels.pkl"

# CONFIGURATION
NUM_CLASSES = 9
IMG_SIZE = 28
BATCH_SIZE = 90
NUM_EPOCHS = 20
CNN_MODEL_DIR = "model/CNN/1cnn_3conv_" + \
    str(NUM_EPOCHS) + "epoch_s" + str(IMG_SIZE) + ".ckpt"


def deepnn(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Third convolutional layer -- maps 64 feature maps to 64.
    with tf.name_scope('conv3'):
        W_conv3 = weight_variable([5, 5, 64, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    # Second pooling layer.
    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([4 * 4 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to NUM_CLASSES classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # Use pickle to dump/load dataset -> not good in case the dataset is large
    # try:
    #     with open(PICKLE_IMGS_DIR, 'rb') as f:
    #         images = pickle.load(f)
    #     with open(PICKLE_LABELS_DIR, 'rb') as f:
    #         labels = pickle.load(f)
    # except Exception:
    images, labels = load_data_grey(TRAIN_DATA_DIR)
    # with open(PICKLE_IMGS_DIR, 'wb') as f:
    #     pickle.dump(images, f)
    # with open(PICKLE_LABELS_DIR, 'wb') as f:
    #     pickle.dump(labels, f)

    # evaluation set
    num_validation = 10
    images, labels = shuffle(images, labels, random_state=0)
    images_eval, labels_eval = images[:num_validation], labels[:num_validation]
    images, labels = images[num_validation:], labels[num_validation:]

    num_datapoint = len(images)
    logger.warning('num_datapoint: %s' % num_datapoint)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE])

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
    count_max = 3
    last_accuracy = 0

    config = tf.ConfigProto()
    # Prevent Tensorflow exploits all the power of CPU
    # config = tf.ConfigProto(intra_op_parallelism_threads=3,
    #                         inter_op_parallelism_threads=3)
    # Replace by allow_growth in order to automatically choose fraction value
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        logger.warning(">>>----------Init sess-------->>>")
        sess.run(tf.global_variables_initializer())
        # epoch
        for i in range(NUM_EPOCHS):
            logger.warning(i)
            images, labels = shuffle(images, labels, random_state=0)

            for batch_idx in range(int(num_datapoint / BATCH_SIZE)):
                logger.warning(batch_idx)
                start_idx = batch_idx * BATCH_SIZE
                end_idx = start_idx + BATCH_SIZE
                # Show the current accuracy
                # if batch_idx % 500 == 0:
                #     train_accuracy = sess.run(accuracy,
                #                               feed_dict={
                #                                   x: images_eval,
                #                                   y_: labels_eval,
                #                                   keep_prob: 1.0})

                #     logger.warning('Epoch %d, batch_idx %d, \
                #         Evaluation accuracy %g' %
                #                    (i, batch_idx, train_accuracy))
                train_step.run(feed_dict={x: images[start_idx:end_idx],
                                          y_: labels[start_idx:end_idx],
                                          keep_prob: 0.7})

            # Evaluation
            # count = count + 1
            # accuracy_ = sess.run(accuracy,
            #                      feed_dict={
            #                          x: images_eval,
            #                          y_: labels_eval,
            #                          keep_prob: 1.0})
            # logger.warning('Epoch %d, training accuracy %g' % (i, accuracy_))
            # if accuracy_ > last_accuracy:  # Better
            #     count = 0
            #     last_accuracy = accuracy_
            #     # Save the current model
            #     saver.save(sess, CNN_MODEL_DIR)
            #     logger.warning('Saved snapshot at epoch: %d' % i)
            # elif count == count_max:
            #     logger.warning("Cannot improve the model. \
            #         Finish training at epoch %d..." % i)
            #     return

        # Save model
        saver.save(sess, CNN_MODEL_DIR)


def evaluate():
    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])
    y_conv, keep_prob = deepnn(x)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, CNN_MODEL_DIR)
        images, labels = load_data_grey(TEST_DATA_DIR)
        # Old code: it breaks when using tensorflow-GPU (not fixed yet)
        # logger.warning('test accuracy %g' % accuracy.eval(feed_dict={
        #     x: images, y_: labels, keep_prob: 1.0}))
        logger.warning('test accuracy %g' % sess.run(accuracy,
                                                     feed_dict={
                                                         x: images,
                                                         y_: labels,
                                                         keep_prob: 1.0}))


def predict(img):
    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE])
    y_conv, keep_prob = deepnn(x)
    predict = tf.argmax(y_conv, 1)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, CNN_MODEL_DIR)
        logger.warning('Label %g' % sess.run(
            predict, feed_dict={x: img, keep_prob: 1.0}))


if __name__ == '__main__':
    # Train
    # tf.app.run(main=main, argv=[sys.argv[0]])

    # Evaluation
    evaluate()

    # Predict
    # img = np.reshape(cv2.resize(cv2.imread(
    #     'data/00011_00000.ppm', 0), (IMG_SIZE, IMG_SIZE)), -1)
    # predict([img])
