import cv2
import numpy as np
from pa1_2 import *

path = r'D:\2021-GUZ\BBM413-Image Processing\ass1\dithering\1.png'
img_o = cv2.imread(path, 0)


def quantization(img, q):
    new_img = img.copy()
    h = np.size(new_img, 0)
    w = np.size(new_img, 1)
    for y in range(h):
        for x in range(w):
            temp = np.round(q * img[y, x] / 255) * int(255 / q)
            if temp > 255:
                temp = 255
            if temp < 0:
                temp = 0
            new_img[y, x] = temp
    return new_img


q = 1
img_q = quantization(img_o, q)
img_d = FloydSteinberg(img_o, q)

cv2.imshow('Original', img_o)

cv2.imshow(f'Quantized q:{q}', img_q)

cv2.imshow(f'Floyd-Steinberg Dithering q:{q}', img_d)
cv2.waitKey()
