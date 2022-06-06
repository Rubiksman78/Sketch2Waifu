import numpy as np
import cv2

def img_kmeans(img_blur, K=8):
    Z = img_blur.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape((img_blur.shape))
    return res

def resize(img, height, width, centerCrop=True, interp='bilinear'):
    imgh, imgw = img.shape[0:2]
    if centerCrop and imgh != imgw:
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]
    img = cv2.resize(img, [height, width])
    return img