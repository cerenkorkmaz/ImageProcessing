import numpy as np
import cv2
import math


def pad_image(source, k):
    padding = int(k / 2)
    padded_image = cv2.copyMakeBorder(source, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    return padded_image


def mean_filter(img, kernel):
    h = img.shape[0]
    w = img.shape[1]

    padded_img = pad_image(img, kernel)
    mean_img = img.copy()

    mean_matrix = np.ones((kernel, kernel), np.float32) / (kernel * kernel)
    for i in range(h):
        for j in range(w):
            for rgb in range(3):
                mean_img[i, j, rgb] = np.sum(padded_img[i:i + kernel, j:j + kernel, rgb] * mean_matrix)
    return mean_img


def gauss_filter(img, kernel, sigma):
    h = img.shape[0]
    w = img.shape[1]

    padded_img = pad_image(img, kernel)
    gauss_img = img.copy()

    x, y = np.meshgrid(np.linspace(-sigma, sigma, kernel), np.linspace(-sigma, sigma, kernel))
    gauss_matrix = np.exp(-((x * x + y * y) / (2 * sigma * sigma)))
    gauss_matrix = gauss_matrix / (2 * math.pi * sigma * sigma)
    gauss_matrix = gauss_matrix / gauss_matrix.sum()

    for i in range(h):
        for j in range(w):
            for rgb in range(3):
                gauss_img[i, j, rgb] = np.sum(padded_img[i:i + kernel, j:j + kernel, rgb] * gauss_matrix)
    return gauss_img


def kuwahara_filter(img, kernel):
    h = img.shape[0]
    w = img.shape[1]

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    padded_hsv = cv2.split((pad_image(hsv_img, kernel)))[2]
    padded_img = pad_image(img, kernel)
    kuwahara_img = img.copy()
    for i in range(kernel // 2, h + kernel // 2):
        for j in range(kernel // 2, w + kernel // 2):
            a = padded_hsv[i - kernel // 2: i + 1, j - kernel // 2: j + 1]
            b = padded_hsv[i:i + kernel // 2 + 1, j - kernel // 2: j + 1]
            c = padded_hsv[i - kernel // 2: i + 1, j:j + kernel // 2 + 1]
            d = padded_hsv[i:i + kernel // 2 + 1, j:j + kernel // 2 + 1]
            std_a, std_b, std_c, std_d = np.std(a), np.std(b), np.std(c), np.std(d)
            stds = [std_a, std_b, std_c, std_d]
            for rgb in range(3):
                i_a = padded_img[i - kernel // 2: i + 1, j - kernel // 2: j + 1, rgb]
                i_b = padded_img[i:i + kernel // 2 + 1, j - kernel // 2: j + 1, rgb]
                i_c = padded_img[i - kernel // 2: i + 1, j:j + kernel // 2 + 1, rgb]
                i_d = padded_img[i:i + kernel // 2 + 1, j:j + kernel // 2 + 1, rgb]
                if min(stds) == std_a:
                    kuwahara_img[i - kernel // 2, j - kernel // 2, rgb] = np.mean(i_a)
                if min(stds) == std_b:
                    kuwahara_img[i - kernel // 2, j - kernel // 2, rgb] = np.mean(i_b)
                if min(stds) == std_c:
                    kuwahara_img[i - kernel // 2, j - kernel // 2, rgb] = np.mean(i_c)
                if min(stds) == std_d:
                    kuwahara_img[i - kernel // 2, j - kernel // 2, rgb] = np.mean(i_d)

    return kuwahara_img


img = cv2.imread(r'D:\2021-GUZ\BBM413-Image Processing\ass2\france.jpg')
kernel_size = 7
sigma = 2
mean = mean_filter(img, kernel_size)
gauss = gauss_filter(img, kernel_size, sigma)
kuwahara = kuwahara_filter(img, kernel_size)
cv2.imshow("Original Image", img)
cv2.imshow(f"{kernel_size}x{kernel_size} Mean Filter", mean)
cv2.imshow(f"{kernel_size}x{kernel_size} Gauss Filter", gauss)
cv2.imshow(f"{kernel_size}x{kernel_size} Kuwahara Filter", kuwahara)
cv2.waitKey(0)
