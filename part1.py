#%%
import math
import numpy as np
import scipy as sp
from scipy import signal
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import cv2

import skimage as sk
import skimage.feature
import skimage.io as skio
import skimage.transform as sktr
import skimage.color as skc
import time

# crops the outer 15% of the image. maintains inner 85%
def crop_center(pil_img):
    img_width, img_height = pil_img.shape

    crop_width = (int)(0.65*img_width)
    remaining_width = (int)((img_width - crop_width)/2)

    crop_height = (int)(0.65*img_height)
    remaining_height = (int)((img_height - crop_height) / 2)

    im = pil_img[remaining_width: img_width-remaining_width, remaining_height: img_height - remaining_height]
    return im

def edge_detect(im, threshold):
    dx = np.array([[1, -1]])
    dy = np.array([[1], [-1]])

    partial_x = signal.convolve2d(im, dx, boundary='symm', mode='same')
    partial_y = signal.convolve2d(im, dy, boundary='symm', mode='same')

    skio.imshow(partial_x, cmap='gray')
    skio.show()
    skio.imshow(partial_y, cmap='gray')
    skio.show()

    magnitude = np.sqrt(np.add(np.square(partial_x), np.square(partial_y)))
    skio.imshow(magnitude)
    skio.show()

    binarize = magnitude > threshold
    skio.imshow(binarize)
    skio.show()

    skio.imsave("edge_image.png", binarize)

    return binarize

def dimensional_crop(img, new_width=None, new_height=None):

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img

def sharpen(name):
    image = cv2.imread(name)
    firstpart = name.split('.')[0]
    new_image = firstpart + "_unsharp.jpg"

    gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)

    unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
    cv2.imwrite(new_image, unsharp_image)

def blur_sharpen(name):
    image = cv2.imread(name)
    image = cv2.GaussianBlur(image, (0, 0), 1.0)
    cv2.imwrite(name+ "_blurred.jpg", image)

    firstpart = name.split('.')[0]
    new_image = firstpart + "_sharpblur.jpg"

    gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
    cv2.imwrite(new_image, unsharp_image)

def straighten(im):
    skio.imshow(im)
    skio.show()

    grayscale = skc.rgb2gray(im)

    best_percent = 0
    best_angle = 0
    best_list = []

    first_height = 0
    first_width = 0

    show_plot = True

    for x in range (-10,11):
        rotated = sp.ndimage.rotate(grayscale, x)

        if x == -10:
            rotated = crop_center(rotated)
            first_height, first_width = rotated.shape
        else:
            rotated = dimensional_crop(rotated, first_width, first_height)
            # skio.imshow(rotated)
            # skio.show()


        rotated = ndimage.gaussian_filter(rotated, 2)

        dx = signal.convolve2d(rotated, np.array([[1, -1]]), boundary='symm', mode='same')
        dy = signal.convolve2d(rotated, np.array([[1], [-1]]), boundary='symm', mode='same')

        # gradient direction
        angles = np.arctan2(dy,dx).flatten()
        angles = np.degrees(angles)

        # TODO SHOW PLOT
        if show_plot:
            if x == 0:
                plt.hist(angles, bins=50)
                plt.show()

        num_90 = np.sum(np.abs(angles-90)<=1) + np.sum(np.abs(angles--90)<=1)
        num_0 = np.sum(np.abs(angles-0)<=1)
        num_180 = np.sum(np.abs(angles-180)<=1) + np.sum(np.abs(angles--180)<=1)

        num_edges = num_90 + num_0 + num_180

        percent_edges = (1.0*num_edges) / (1.0*len(angles))

        if (percent_edges > best_percent):
            best_percent = percent_edges
            best_angle = x
            best_list = angles

    print("rotated by " + str(best_angle) + " degrees\n")
    skio.imshow(sp.ndimage.rotate(im, best_angle))
    skio.show()

    if show_plot:
        plt.hist(best_list, bins=50)
        plt.show()

    return sp.ndimage.rotate(im, best_angle)


edge = True
to_straighten = True
sharpen = True
blur_then_sharpen = True

from os import listdir
from os.path import isfile, join
mypath = 'straighten'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


if edge:
    imname = "cameraman.png"
    im = skio.imread(imname)
    skio.imshow(im)
    skio.show()

    grayscale = skc.rgb2gray(im)
    dx = np.array([[1, -1]])
    dy = np.array([[1],[-1]])

    gradient_magnitude = edge_detect(grayscale, 0.14)

    gauss = ndimage.gaussian_filter(grayscale, 3)
    gauss_deriv = edge_detect(gauss, 0.022)

if to_straighten:
    for file in onlyfiles:
        print(file)
        im = skio.imread('straighten/' + file)
        straightened = straighten(im)
        skio.imsave("straightened_" + file, straightened)


if sharpen:
    #uses open cv to sharpen, saves as imagename + _unsharpen
    sharpen("taj.jpg")

if blur_then_sharpen:
    blur_sharpen("sharpen/snail.jpg")

