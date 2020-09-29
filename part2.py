# %%
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
import copy


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

    for x in gauss_stack:
        skio.imshow(x)
        skio.show()

    for x in lap_stack:
        skio.imshow(x)
        skio.show()

    # for x in gauss_stack:
    #     skio.imshow(x)
    #     skio.show()

    # for x in range(6):
    #     skio.imshow(lap_stack[x], cmap='gray')
    #     skio.imsave("multires_" + str(x) + ".jpg", lap_stack[x])
    #     skio.show()
    #
    # for x in lap_stack:
    #     skio.imshow(x, cmap='gray')
    #     skio.imsave("multires_"+"+.jpg", x)
    #     skio.show()

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


hybrid = False
stack = False
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
    stacks(skc.rgb2gray(mr))


    #skio.imsave("tesla_kart.jpg", mr)
