import cv2
import numpy as np
import tensorflow as tf
from src.CNN import deepnn


CNN_MODEL_DIR = "model/CNN/3cnn.ckpt"
IMG_SIZE = 56
def detect(path):
	x_placeholder = tf.placeholder(tf.float32, [None, IMG_SIZE,IMG_SIZE,3])
	y_conv, keep_prob = deepnn(x_placeholder)
	predict = tf.argmax(y_conv, 1)
	y_sm = tf.nn.softmax(y_conv)

	with tf.Session() as sess:
		saver = tf.train.Saver()
		saver.restore(sess, CNN_MODEL_DIR)

		cap = cv2.VideoCapture(path)
		while (cap.isOpened()):
			# Take each frame
			ret, frame = cap.read()
			if ret == True:
				# Use Gaussian Blur to reduce high frequency noise
				# and allow us to focus on the structural objects inside the frame
				blurred = cv2.GaussianBlur(frame, (3, 3), 0)
				#TODO: implement "Bilateral Filtering" and see if it's cost
				
				# Convert BGR to HSV
				hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

				
				# Threshold the HSV image to get only red
				red1 = cv2.inRange(hsv, (0, 100, 100), (15, 255, 255))
				red2 = cv2.inRange(hsv, (160, 100, 120), (180, 255, 255))
				red_mask = cv2.add(red1, red2)

				# Threshold the HSV image to get only blue
				blue_mask = cv2.inRange(hsv, (100, 120, 100), (120, 255, 255))

				mask = cv2.add(red_mask, blue_mask)

				# Erode to reduce noise and dilate to focus
				mask = cv2.erode(mask, None, iterations = 1)
				mask = cv2.dilate(mask, None, iterations = 5)

				# Find contours in the mask
				# cnts = cv2.findContours(image = mask.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)[-2]
				cnts = cv2.findContours(image = mask, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)[-2]

				# Proceed if at least one contour was found
				if len(cnts) > 0:
					# Draw all contours and fill the contour interiors -> mask
					cv2.drawContours(image = mask, contours = cnts, contourIdx = -1, color = 255, thickness = -1)
					mask = cv2.dilate(mask, None, iterations = 5)
					mask = cv2.erode(mask, None, iterations = 2)

				# Draw a rectangle outside each contour
				cnts = cv2.findContours(image = mask.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)[-2]
				for cnt in cnts:
					x, y, w, h = cv2.boundingRect(cnt)
					if w > 20 and h > 20 and float(h)/w > 0.8 and float(h)/w < 1.5:
					# if True:
						#TODO: check if it is a sign
						isSign = False
						# window = cv2.resize(frame[x:x+w, y:y+h], (28,28)) #TODO bug: imgwarp.cpp:3229: error: (-215) ssize.area() > 0 in function resize
						
						# resize -> IMG_SIZE*IMG_SIZE
						x_center, y_center = x + int(w/2), y + int(h/2)
						try:
							_max = max(w,h)
							window = cv2.resize(frame[x_center-int(_max/2):x_center+int(_max/2), y_center-int(_max/2):y_center+int(_max/2)], (IMG_SIZE,IMG_SIZE))
						except:
							_min = min(w,h)
							window = cv2.resize(frame[x_center-int(_min/2):x_center+int(_min/2), y_center-int(_min/2):y_center+int(_min/2)], (IMG_SIZE,IMG_SIZE))

						window = cv2.cvtColor(window, cv2.COLOR_HSV2RGB) #hsv2rgb
						_y_conv, lable = sess.run([y_sm, predict], feed_dict={x_placeholder: [window], keep_prob: 1.0})
						if lable == 0:
							isSign = True

						# if _y_conv[0][0] == 1:
						# 	isSign = True

						# if True:
						if isSign:
							cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
				
				cv2.imshow("frame", frame)
				cv2.imshow("mask", mask)
				if cv2.waitKey(25) & 0xFF == ord('q'):
					break
			else:
				break
		cap.release()
		cv2.destroyAllWindows()

if __name__ == "__main__":
	detect("data/MVI_1049.avi")
	# detect("data/MVI_1082.avi")
	# detect("data/test_sign.avi")
	# detect("data/test_sign2.avi")