import cv2
from pa2_2 import *

path = r'D:\2021-GUZ\BBM413-Image Processing\ass1\colortransfer'
source = cv2.imread(path + '\scotland_house.jpg')
source = source.astype('uint8')
target = cv2.imread(path + '\scotland_plain.jpg')
target = target.astype('uint8')

final_image = ColorTransfer(source, target)

cv2.imshow('Source Image', source)
cv2.imshow('Target Palette', target)
cv2.imshow('Result of Color Transfer', final_image)
cv2.waitKey()
