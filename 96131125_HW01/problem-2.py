import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def gamma_transform(img, c=1.0, gamma=1.0):
    """
    Gamma Transform, this function avoid repetition in calculation
    :param img: input image
    :param c:
    :param gamma:
    :return:
    """
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)])
    table = np.uint8(table)

    # apply gamma correction using the lookup table
    return cv.LUT(img, table)


# path of images
img_path_1 = '.\Images\Cont1.jpg'
img_path_2 = '.\Images\Cont2.png'

# read image form file
img_1 = cv.imread(filename=img_path_1, flags=cv.IMREAD_GRAYSCALE)
img_2 = cv.imread(filename=img_path_2, flags=cv.IMREAD_GRAYSCALE)

# calculate histogram of images
hist_img_1 = cv.calcHist([img_1], [0], None, [256], [0,256])
hist_img_2 = cv.calcHist([img_2], [0], None, [256], [0,256])

# plot histograms
fig = plt.figure(figsize=(8,8))
ax = plt.subplot(2,2,1)
plt.imshow(img_1, cmap='gray')
ax = plt.subplot(2,2,2)
plt.plot(hist_img_1)
plt.xlim([0,256])
ax.set_title('Histogram Of Input Cont1.jpg')
ax = plt.subplot(2,2,3)
plt.imshow(img_2, cmap='gray')
ax = plt.subplot(2,2,4)
plt.plot(hist_img_2)
plt.xlim([0,256])
ax.set_title('Histogram Of Input Cont2.jpg')

# enhance first image
#img_1_eq = cv.equalizeHist(img_1)
# gamma transformation
img_1_gamma_trans = gamma_transform(img_1, c=1.0, gamma=0.55)
# histogram of enhanced image
hist_img_1_eq = cv.calcHist([img_1_gamma_trans], [0], None, [256], [0, 256])

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(1,2,1)
plt.imshow(img_1, cmap='gray')
ax.set_title('Input Cont1.jpg')
ax = plt.subplot(1,2,2)
plt.imshow(img_1_gamma_trans, cmap='gray')
ax.set_title('Input Cont1.jpg after enhance by Gamma Transformation(Gamma=0.5)')

# enhance second image
# histogram equalization
#img_2_eq = gamma_transform(img_2, c=1.0, gamma=1.55)
img_2_eq = cv.equalizeHist(img_2)

# histogram of enhanced image
hist_img_2_eq = cv.calcHist([img_2_eq], [0], None, [256], [0,256])

fig = plt.figure(figsize=(8,8))
ax = plt.subplot(1,2,1)
plt.imshow(img_2, cmap='gray')
ax.set_title('Input Cont2.jpg')
ax = plt.subplot(1,2,2)
plt.imshow(img_2_eq, cmap='gray')
ax.set_title('Input Cont2.jpg after Histogram Equalization')

# plot histograms of enhanced images
fig = plt.figure(figsize=(8,8))
ax = plt.subplot(2,2,1)
plt.imshow(img_1_gamma_trans, cmap='gray')
ax = plt.subplot(2,2,2)
plt.plot(hist_img_1_eq)
plt.xlim([0,256])
ax.set_title('Histogram Of Input Cont1.jpg after enhance')
ax = plt.subplot(2,2,3)
plt.imshow(img_2_eq, cmap='gray')
ax = plt.subplot(2,2,4)
plt.plot(hist_img_2_eq)
plt.xlim([0,256])
ax.set_title('Histogram Of Input Cont2.jpg after enhance')

plt.show()

