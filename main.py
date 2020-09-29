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

    cv2.imshow('image', unsharp_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(new_image, unsharp_image)

def blur_sharpen(name):
    image = cv2.imread(name)
    image = cv2.GaussianBlur(image, (0, 0), 1.0)
    cv2.imwrite(name+ "_blurred.jpg", image)

    firstpart = name.split('.')[0]
    new_image = firstpart + "_sharpblur.jpg"

    gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)

    cv2.imshow('image', unsharp_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)


def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int)(np.abs(2 * r + 1 - R))
    cpad = (int)(np.abs(2 * c + 1 - C))
    return np.pad(
        im, [(0 if r > (R - 1) / 2 else rpad, 0 if r < (R - 1) / 2 else rpad),
             (0 if c > (C - 1) / 2 else cpad, 0 if c < (C - 1) / 2 else cpad),
             (0, 0)], 'constant')


def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy


def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape

    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2


def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    len2 = np.sqrt((p4[1] - p3[1]) ** 2 + (p4[0] - p3[0]) ** 2)
    dscale = len2 / len1
    if dscale < 1:
        im1 = sktr.rescale(im1, (dscale, dscale, 1))
    else:
        im2 = sktr.rescale(im2, (1. / dscale, 1. / dscale, 1))
    return im1, im2


def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta * 180 / np.pi)
    return im1, dtheta


def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape
    if h1 < h2:
        im2 = im2[int(np.floor((h2 - h1) / 2.)): -int(np.ceil((h2 - h1) / 2.)), :, :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1 - h2) / 2.)): -int(np.ceil((h1 - h2) / 2.)), :, :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2 - w1) / 2.)): -int(np.ceil((w2 - w1) / 2.)), :]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1 - w2) / 2.)): -int(np.ceil((w1 - w2) / 2.)), :]
    assert im1.shape == im2.shape
    return im1, im2


def align_images(im1, im2):
    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2


def hybrid_image(im1, im2):
    assert im1.ndim == im2.ndim
    im1, im2 = align_images(im1, im2)
    im1 = skc.rgb2gray(im1)
    im2 = skc.rgb2gray(im2)

    if im1.ndim == 2:
        firstIm = im1[:, :]
        secondIm = im2[:, :]
        fin1 = ndimage.gaussian_filter(firstIm, 25)
        fin2 = secondIm - ndimage.gaussian_filter(secondIm, 20)
        hybrid = np.dot(fin1 + fin2, 1 / 2)
        skio.imshow(hybrid, cmap='gray')
        skio.show()
        return hybrid

    hybrid = np.zeros(im1.shape)
    plots = False

    if plots:
        plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(skc.rgb2gray(im1))))))
        plt.show()
        plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(skc.rgb2gray(im2))))))
        plt.show()

    for i in range(3):
        img1 = im1[:, :, i]
        img2 = im2[:, :, i]

        lowpass = ndimage.gaussian_filter(img1, 20)

        highpass = img2 - ndimage.gaussian_filter(img2, 20)

        hybrid[:, :, i] = np.dot(lowpass + highpass, 1 / 2)

    skio.imshow(hybrid)
    skio.show()

    if plots:
        plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(skc.rgb2gray(hybrid))))))
        plt.show()

    return hybrid


def stacks(im):
    gauss_stack = gaussian_stack(im)
    lap_stack = laplacian_stack(im)

    print("stack len " + str(len(gauss_stack)))

    for x in gauss_stack:
        skio.imshow(x, cmap='gray')
        skio.show()

    for x in lap_stack:
        skio.imshow(x,cmap='gray')
        skio.show()


def gaussian_stack(im):
    gauss_stack = []
    for x in range(6):
        im = ndimage.gaussian_filter(im, sigma=2)
        gauss_stack.append(im)
    return gauss_stack


def laplacian_stack(im):
    gauss_stack = gaussian_stack(im)

    for i in range(5):
        result = gauss_stack[i] - gauss_stack[i + 1]
        gauss_stack[i] = result

    return gauss_stack


def multires_blending(im1, im2, mask):
    assert im1.shape == im2.shape == mask.shape

    lap1 = laplacian_stack(im1)
    lap2 = laplacian_stack(im2)
    mask_gauss = gaussian_stack(mask)
    lap_mask = mask_gauss
    lap_mask2 = (1 - np.array(lap_mask))

    lap1_out = np.multiply(lap_mask, lap1)
    lap2_out = np.multiply(lap_mask2, lap2)

    final_lap = lap1_out + lap2_out

    output = np.zeros(lap1[0].shape)

    for i in range(len(lap1)):
        output += final_lap[i]
        # skio.imshow(output)
        # skio.show()

    output2 = np.clip(output, 0, 1)

    skio.imshow(output2)
    skio.show()

    return output2


def create_mask(im1):
    height = len(im1)
    width = len(im1[0])
    depth = len(im1[0][0])

    mask = np.zeros((height, width, depth))
    for i in range(int(width / 2)):
        mask[:, i] = np.ones((height, 1))
    mask = ndimage.gaussian_filter(mask, 30)

    skio.imsave("mask.jpg", mask)

    return mask


## PART 1
edge = True
to_straighten = True
to_sharpen = True
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


if to_sharpen:
    #uses open cv to sharpen, saves as imagename + _unsharpen
    sharpen("sharpen/taj.jpg")

if blur_then_sharpen:
    blur_sharpen("sharpen/snail.jpg")

## PART 2
hybrid = True
stack = True
multires = True

if hybrid:
    imname = "kitten.jpg"
    im1 = skio.imread(imname)
    imname = "puppy.jpg"
    im2 = skio.imread(imname)
    hybrid_image(im1, im2)

if stack:
    imname = "lincoln.jpg"
    im = skio.imread(imname)
    stacks(skc.rgb2gray(im))

if multires:
    imname = "bowser.jpeg"
    im1 = skio.imread(imname) / 255

    imname = "tesla.jpg"
    im2 = skio.imread(imname) / 255

    mask = create_mask(im1)
    mask = skio.imread("mariomask.jpg") / 255

    mr = multires_blending(im2, im1, mask)
    skio.imsave("tesla_kart.jpg", mr)
