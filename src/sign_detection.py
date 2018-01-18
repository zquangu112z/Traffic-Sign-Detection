import imutils
import time
import cv2
import argparse
import sys
import tempfile
import pickle
from sklearn.utils import shuffle
import numpy as np
from src.CNN import deepnn
# from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

TRAIN_DATA_DIR = "data/raw/training"
TEST_DATA_DIR = "data/raw/test"
CNN_MODEL_DIR = "model/CNN/cnn.ckpt"
PICKLE_IMGS_DIR = "data/pickle/train_imgs.pkl"
PICKLE_LABELS_DIR = "data/pickle/test_labels.pkl"


#https://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
 
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
 
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
 
		# yield the next image in the pyramid
		yield image


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# load the image and define the window width and height
image = cv2.imread('data/no-left-turn.jpg',0)
(winW, winH) = (28, 28)

x_placeholder = tf.placeholder(tf.float32, [None, 784])
y_conv, keep_prob = deepnn(x_placeholder)
predict = tf.argmax(y_conv, 1)
y_sm = tf.nn.softmax(y_conv)

with tf.Session() as sess:
	saver = tf.train.Saver()
	saver.restore(sess, CNN_MODEL_DIR)
	

	# loop over the image pyramid
	for resized in pyramid(image, scale=1.5):
		# loop over the sliding window for each layer of the pyramid
		for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
			# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue

			
			img = cv2.resize(window, (28,28))
			img = np.reshape(img, -1)
			_y_conv, lable = sess.run([y_sm, predict], feed_dict={x_placeholder: [np.reshape(img, -1)], keep_prob: 1.0})


			if lable == 0:
				print(_y_conv)
				# print(lable)
			# if _value > 0.7:
				clone = resized.copy()
				cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
				cv2.imshow("Window", clone)
				cv2.waitKey(1)
				time.sleep(1)