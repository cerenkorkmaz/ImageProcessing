import cv2
import numpy as np


def FloydSteinberg(img, q):
    img_fs = img.copy()
    h = np.size(img_fs, 0)
    w = np.size(img_fs, 1)

    for y in range(h-1):
        for x in range(w-1):
            oldpixel = img_fs[y][x]
            newpixel = np.round(np.round(oldpixel / 255 * q) * 255 / q)
            img_fs[y][x] = newpixel
            quant_error = oldpixel - newpixel

            temp1 = img_fs[y][x + 1] + quant_error * 7 / 16
            if temp1 > 255:
                img_fs[y][x + 1] = 255
            if temp1 < 0:
                img_fs[y][x + 1] = 0
            elif 0 <= temp1 <= 255:
                img_fs[y][x + 1] = temp1

            temp2 = img_fs[y + 1][x - 1] + quant_error * 3 / 16
            if temp2 > 255:
                img_fs[y + 1][x - 1] = 255
            if temp2 < 0:
                img_fs[y + 1][x - 1] = 0
            elif 0 <= temp1 <= 255:
                img_fs[y + 1][x - 1] = temp2

            temp3 = img_fs[y + 1][x] + quant_error * 5 / 16
            if temp3 > 255:
                img_fs[y + 1][x] = 255
            if temp3 < 0:
                img_fs[y + 1][x] = 0
            elif 0 <= temp1 <= 255:
                img_fs[y + 1][x] = temp3

            temp4 = img_fs[y + 1][x + 1] + quant_error * 1 / 16
            if temp4 > 255:
                img_fs[y + 1][x + 1] = 255
            if temp4 < 0:
                img_fs[y + 1][x + 1] = 0
            elif 0 <= temp1 <= 255:
                img_fs[y + 1][x + 1] = temp4

    return img_fs
