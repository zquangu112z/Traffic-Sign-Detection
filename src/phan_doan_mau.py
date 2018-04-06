import cv2
import tensorflow as tf
from src.CNN.CNN_3channels_2conv_s28 import deepnn
import logging

CNN_MODEL_DIR = "model/CNN/3cnn_2conv_10epoch_s28.ckpt"
# CNN_MODEL_DIR = "model/CNN/3cnn_2conv.ckpt" # 1 epoch, work, very bad
IMG_SIZE = 28

LABEL = {
    0: 'Han che toc do',
    1: 'Stop',
    2: 'One-Way',
    3: 'Diploma',
    4: 'Turn Right',
    5: 'Turn Left',
    6: 'No Left',
    7: 'No Right',
    8: 'Others',
}


def detect(path):
    x_placeholder = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])
    y_conv, keep_prob = deepnn(x_placeholder)
    predict = tf.argmax(y_conv, 1)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, CNN_MODEL_DIR)

        cap = cv2.VideoCapture(path)
        while (cap.isOpened()):
            # Take each frame
            ret, frame = cap.read()
            if ret is True:
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
                mask = cv2.erode(mask, None, iterations=1)
                mask = cv2.dilate(mask, None, iterations=3)

                # Find contours in the mask
                # cnts = cv2.findContours(image = mask.copy(),
                # mode = cv2.RETR_EXTERNAL,
                # method = cv2.CHAIN_APPROX_SIMPLE)[-2]
                cnts = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL,
                                        method=cv2.CHAIN_APPROX_SIMPLE)[-2]

                # Proceed if at least one contour was found
                if len(cnts) > 0:
                    # Draw all contours and fill the contour interiors -> mask
                    cv2.drawContours(image=mask, contours=cnts, contourIdx=-1,
                                     color=255, thickness=-1)
                    mask = cv2.dilate(mask, None, iterations=3)
                    mask = cv2.erode(mask, None, iterations=3)

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
                            i += 1
                            window = cv2.resize(window, (IMG_SIZE, IMG_SIZE))
                            # cv2.imshow("window %d" % i, window)

                            # window = cv2.cvtColor(
                            #     window, cv2.COLOR_HSV2RGB)  # hsv2rgb
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
                        except Exception as e:
                            print(e)

                        cv2.imshow("frame", frame)
                    # cv2.imshow("mask", mask) # debug purpose @TODO: remove
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detect("data/MVI_1049.avi")
