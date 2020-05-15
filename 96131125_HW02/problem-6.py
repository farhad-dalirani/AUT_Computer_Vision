
# import the necessary packages
from skimage import exposure
import numpy as np
import argparse
import cv2 as cv


##############################################
#   Extract all contours
##############################################
image = cv.imread('.\\Images\\Cont.png')

# convert input image to grayscale and find edges by Canny's algorithm
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
edged = cv.Canny(gray, 10, 100)

# find contours in the edged image
im2, contours, hierarchy = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
sum_ = sum([len(a) for a in contours])
print('segments= CHAIN_APPROX_SIMPLE', sum_)

# display contours
contours_img = cv.drawContours(image, contours, -1, (255,0,0), 7)
cv.imshow('d', cv.resize(contours_img,(800,800)))
cv.imwrite('.\\6\\all-contours.jpg', contours_img)

##############################################
#   Effect of different method
##############################################
# mode 1
image = cv.imread('.\\Images\\Cont.png')

# convert input image to grayscale and find edges by Canny's algorithm
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
edged = cv.Canny(gray, 10, 100)

# find contours in the edged image
im2, contours, hierarchy = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

sum_ = sum([len(a) for a in contours])
print('segments CHAIN_APPROX_NONE=', sum_)

# display contours
contours_img = cv.drawContours(image, contours, -1, (255,0,0), 7)
cv.imshow('d', cv.resize(contours_img,(800,800)))
cv.imwrite('.\\6\\all-contours-model1.jpg', contours_img)

# mode 2
# convert input image to grayscale and find edges by Canny's algorithm
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
edged = cv.Canny(gray, 10, 100)

# find contours in the edged image
im2, contours, hierarchy = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)
sum_ = sum([len(a) for a in contours])
print('segments CHAIN_APPROX_TC89_L1=', sum_)

# display contours
contours_img = cv.drawContours(image, contours, -1, (255,0,0), 7)
cv.imshow('d', cv.resize(contours_img,(800,800)))
cv.imwrite('.\\6\\all-contours-model2.jpg', contours_img)

cv.waitKey(0)
