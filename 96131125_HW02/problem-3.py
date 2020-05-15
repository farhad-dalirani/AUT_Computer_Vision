import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

###########################
#   Effect of changing low threshold
###########################
# read image
img = cv.imread(filename='.\\Images\\Edge2.jpg', flags=cv.IMREAD_GRAYSCALE)

# details of LoG filters (t_l, t_h, size)
canny_details = [(5, 150, 3), (45, 150, 3), (75, 150, 3), (130, 150, 3)]

images = []

fig = plt.figure()
plt.imshow(img, cmap='gray')
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.title('Input Image')

# apply filters on image
for id in range(len(canny_details)):
    # canny edge detector
    canny_img = cv.Canny(image=img, threshold1=canny_details[id][0], threshold2=canny_details[id][1],
                         apertureSize=canny_details[id][2])
    images.append(canny_img)

    cv.imshow('Canny, {}*{} t_l {}, t_h {}'.format(
        canny_details[id][2], canny_details[id][2],
        canny_details[id][0], canny_details[id][1]),
        canny_img)
    cv.imwrite('.\\3\\Canny size{} t_l {} t_h {}.jpg'.format(
         canny_details[id][2],
        canny_details[id][0], canny_details[id][1]),
                                                    canny_img)

###########################
#   Effect of changing high threshold
###########################
# read image
img = cv.imread(filename='.\\Images\\Edge2.jpg', flags=cv.IMREAD_GRAYSCALE)

# details of LoG filters (t_l, t_h, size)
canny_details = [(50, 70, 3), (50, 100, 3), (50, 150, 3), (50, 200, 3)]

images = []

fig = plt.figure()
plt.imshow(img, cmap='gray')
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.title('Input Image')

# apply filters on image
for id in range(len(canny_details)):
    # canny edge detector
    canny_img = cv.Canny(image=img, threshold1=canny_details[id][0], threshold2=canny_details[id][1],
                         apertureSize=canny_details[id][2])
    images.append(canny_img)

    cv.imshow('Canny, {}*{} t_l {}, t_h {}'.format(
        canny_details[id][2], canny_details[id][2],
        canny_details[id][0], canny_details[id][1]),
        canny_img)
    cv.imwrite('.\\3\\Canny size{} t_l {} t_h {}.jpg'.format(
         canny_details[id][2],
        canny_details[id][0], canny_details[id][1]),
                                                    canny_img)


###########################
#   Effect of changing size
###########################
# read image
img = cv.imread(filename='.\\Images\\Edge2.jpg', flags=cv.IMREAD_GRAYSCALE)

# details of LoG filters (t_l, t_h, size)
canny_details = [(100, 200, 3), (100, 200, 5), (100, 200, 7)]

images = []

fig = plt.figure()
plt.imshow(img, cmap='gray')
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.title('Input Image')

# apply filters on image
for id in range(len(canny_details)):
    # canny edge detector
    canny_img = cv.Canny(image=img, threshold1=canny_details[id][0], threshold2=canny_details[id][1],
                         apertureSize=canny_details[id][2])
    images.append(canny_img)

    cv.imshow('Canny, {}*{} t_l {}, t_h {}'.format(
        canny_details[id][2], canny_details[id][2],
        canny_details[id][0], canny_details[id][1]),
        canny_img)
    cv.imwrite('.\\3\\c-Canny size{} t_l {} t_h {}.jpg'.format(
         canny_details[id][2],
        canny_details[id][0], canny_details[id][1]),
                                                    canny_img)

# laplacian and sobel
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
sobel = np.sqrt(sobelx**2+sobely**2)

cv.imwrite('.\\3\\laplacian.jpg', laplacian)
cv.imwrite('.\\3\\sobel.jpg', sobelx)

cv.waitKey(0)


