import numpy as np
import cv2
import matplotlib.pyplot as plt

#This function take a rgb image and return edge image with canny algorithm
def edge_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 300, 370)
    return edges

#This function plot an image in rgb,extract its edges using edge_extractor and plot the edges too
def plot_image(path):
    img = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
    edges = edge_extractor(img)
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.subplot(122), plt.imshow(edges), plt.title('Edge')
    plt.show()

#plot_image("D:/Datasets/anime_face/0000001.jpg")
