import scipy.ndimage as nd
import matplotlib.pyplot as plt
import numpy as np
import cv2

#This function applies a median filter to an image
def median_filter(img, kernel_size):
    return nd.median_filter(img, kernel_size)

#This function applies a k means with opencv to an image
def k_means(img, k):
    img2 = img.reshape((-1,3))
    img2 = np.float32(img2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts=10
    ret,label,center=cv2.kmeans(img2,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    return result_image

#This function applies a median filter then a k means then a median filter to an image
def median_k_means_median_filter(img, kernel_size, k):
    img = median_filter(img, kernel_size)
    img = k_means(img, k)
    img = median_filter(img, kernel_size)
    return img

#This function plots an image, applies the previous function and plot the result too
def plot_image(path, kernel_size, k):
    img = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
    img_filter = median_k_means_median_filter(img, kernel_size, k)
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.subplot(122), plt.imshow(img_filter), plt.title('Median K-Means Median Filter')
    plt.axis('off')
    plt.show()

plot_image("D:/Datasets/anime_face/0000002.jpg", 50, 2)