import cv2
import tensorflow as tf
from src.CNN.CNN_1channels_2conv_s28 import deepnn
# from src.CNN.CNN_1channels_3conv_s28 import deepnn
import logging
import numpy as np


# nham lan bien gioi han toc do
CNN_MODEL_DIR = "model/CNN/1cnn_2conv_10epoch_s28.ckpt" # best
# CNN_MODEL_DIR = "model/CNN/1cnn_2conv_15epoch_s28.ckpt"  # better than 10 epoch
# CNN_MODEL_DIR = "model/CNN/1cnn_2conv_15epoch_s28_newdata.ckpt"  # van con nham lan
# CNN_MODEL_DIR = "model/CNN/1cnn_2conv_20epoch_s28_newdata.ckpt" 


IMG_SIZE = 28

LABEL = {
    0: 'Dung',
    1: 'Re trai',
    2: 'Re phai',
    3: 'Cam re trai',
    4: 'Cam re phai',
    5: 'Mot chieu',
    6: 'Toc do toi da',
    7: 'Others'
}

f = open('result.txt', 'a')


def detect(path):
    x_placeholder = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE])
    y_conv, keep_prob = deepnn(x_placeholder)
    predict = tf.argmax(y_conv, 1)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, CNN_MODEL_DIR)

        cap = cv2.VideoCapture(path)
        frame_idx = 0
        while (cap.isOpened()):
            # Take each frame
            frame_idx += 1
            ret, frame = cap.read()
            if frame is None:
                break
            # Use Gaussian Blur to reduce high frequency noise
            # and allow us to focus on the structural objects
            # inside the frame
            blurred = cv2.GaussianBlur(frame, (3, 3), 0)
            # @TODO: implement "Bilateral Filtering" and see if it's cost

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
            # mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=3)
            # mask = cv2.erode(mask, None, iterations=1)

            # Find contours in the mask
            # cnts = cv2.findContours(image = mask.copy(),
            # mode = cv2.RETR_EXTERNAL,
            # method = cv2.CHAIN_APPROX_SIMPLE)[-2]
            cnts = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL,
                                    method=cv2.CHAIN_APPROX_SIMPLE)[-2]

            # Proceed if at least one contour was found
            # if len(cnts) > 0:
            #  # Draw all contours and fill the contour interiors -> mask
            #     cv2.drawContours(image=mask, contours=cnts,
            #                      contourIdx=-1,
            #                      color=255, thickness=-1)
            #     mask = cv2.dilate(mask, None, iterations=3)
            #     mask = cv2.erode(mask, None, iterations=3)

            # Draw a rectangle outside each contour
            cnts = cv2.findContours(image=mask.copy(),
                                    mode=cv2.RETR_EXTERNAL,
                                    method=cv2.CHAIN_APPROX_SIMPLE)[-2]
            i = 0
            for cnt in cnts:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 20 and h > 20 and float(h) / w > 0.8 and \
                        float(h) / w < 1.5:
                    try:
                        window = frame[y:y + h, x:x + w]
                        # temp = window
                        i += 1

                        window = cv2.resize(window, (IMG_SIZE, IMG_SIZE))
                        window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                        # cv2.imshow("window %d" % i, window)
                        window = np.reshape(window, -1)

                        label = sess.run(predict,
                                         feed_dict={
                                             x_placeholder: [window],
                                             keep_prob: 1.0})

                        if label[0] != 8:
                            # logging.warning(label[0], str(_y_conv))
                            cv2.rectangle(
                                frame,
                                (x, y),
                                (x + w, y + h),
                                (0, 255, 100), 1)
                            cv2.putText(
                                frame, LABEL[label[0]],
                                (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.3, (0, 255, 0), 1, cv2.LINE_AA)

                            # Save result to text file
                            f.write('\n%d %d %d %d %d %d' % (frame_idx,
                                                             label[0],
                                                             x,
                                                             y,
                                                             x + w,
                                                             y + h))
                        # Extract data
                        # if label[0] == 6:
                        #     logging.warning("save one-way")
                        #     cv2.imwrite("data/raw/training/00060/5" +
                        #                 str(frame_idx) + ".jpg", temp)
                    except Exception as e:
                        print(e)

            cv2.imshow("frame", frame)
            # cv2.imshow("mask", mask) # debug purpose @TODO: remove
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        f.close()


VIDEO = [
    # 'data/00202.MTS',
    'data/MVI_1049.avi',
    'data/MVI_1054.avi',
    'data/MVI_1061.avi',
    'data/MVI_1062.avi',
    'data/MVI_1063.avi',
]

if __name__ == "__main__":
    detect(VIDEO[3])
