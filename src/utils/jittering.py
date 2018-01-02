import cv2
import numpy as np


def rotatingImg(img, angle):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def translatingImg(img, width, height):
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, width], [0, 1, height]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def rescalingImg(img, factor):
    height, width = img.shape[:2]
    dst = cv2.resize(img, (int(factor*width), int(factor*height)), interpolation=cv2.INTER_CUBIC)
    return dst


def flippingImg(img, direction):
    dst = flip(img, direction)
    return dst


def shearingImg(img, factor_width, factor_heigh):
    rows, cols = img.shape[:2]
    distLeft = 0
    distRight = 0
    distTop = 0
    distBottom = 0

    if(factor_width >= 0):
        distRight += factor_width
    if(factor_width < 0):
        distLeft -= factor_width
    if(factor_heigh >= 0):
        distTop += factor_heigh
    if(factor_heigh < 0):
        distBottom -= factor_heigh

    pts1 = np.float32([[0, 0], [cols, 0], [0, rows]])
    pts2 = np.float32([[distRight, distBottom], [cols-distLeft, distTop], [distLeft, rows-distTop]])

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def stretchingImg(img, alpha):
    rows, cols, channels = img.shape
    for i in range(rows):
        for j in range(cols):
            for k in range(channels):
                img[i, j, k] = min(255, alpha*img[i, j, k])
    return img

if __name__ == "__main__":
    img = cv2.imread('data/test.jpg')
    height, width = img.shape[:2]

    res = cv2.resize(img, (400, 700))

    res = translatingImg(res, -10, 10)
    cv2.imshow('img', res)
    res = rescalingImg(res, 2)
    cv2.imshow('img', res)

    dst = shearingImg(img, 100, 0)
    # cv2.imshow('img', dst)
    img2 = stretchingImg(img, 0.5)
    # cv2.imshow('img2', img2)

    cv2.waitKey(0)
