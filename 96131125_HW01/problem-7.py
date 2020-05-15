import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.util import random_noise

# all noisy images were saved in '.\\noisy_images\\'

###################################################
#    Recover Images That Contain Gaussian Noise   #
###################################################
# path of images
img_path_g_o = '.\\Images\\Gaussian.jpeg'
img_path_g_n = ['.\\noisy_images\\gaussain-{}.jpg'.format(i) for i in range(0, 4)]

# read noise free image form file
img_g_o = cv.imread(filename=img_path_g_o, flags=cv.IMREAD_GRAYSCALE)
# read images with gaussian noise form file
img_g_n = [cv.imread(filename=path, flags=cv.IMREAD_GRAYSCALE) for path in img_path_g_n]

# smoothing filter
kernel_g = np.ones((3,3), dtype=np.float32) / (3*3)

# apply filter to all noisy images
for img in img_g_n:
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(2, 2, 1)
    plt.imshow(img_g_o, cmap='gray')
    ax.set_title('Input')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 2, 3)
    plt.imshow(img, cmap='gray')
    ax.set_title('Noisy Image')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    # condoling the kernel with image
    new_img = cv.filter2D(src=img, ddepth=-1, kernel=kernel_g)
    ax = plt.subplot(2, 2, 4)
    plt.imshow(new_img, cmap='gray')
    ax.set_title('Denoised Image')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)


###################################################
#    Recover Images That Contain Uniform Noise   #
###################################################
# path of images
img_path_u_o = '.\\Images\\Uniform.jpg'
img_path_u_n = ['.\\noisy_images\\uniform-{}.jpg'.format(i) for i in range(1, 5)]

# read noise free image form file
img_u_o = cv.imread(filename=img_path_u_o, flags=cv.IMREAD_GRAYSCALE)
# read images with gaussian noise form file
img_u_n = [cv.imread(filename=path, flags=cv.IMREAD_GRAYSCALE) for path in img_path_u_n]

# smoothing filter
kernel_u = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / (24)

# apply filter to all noisy images
for img in img_u_n:
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(2, 2, 1)
    plt.imshow(img_u_o, cmap='gray')
    ax.set_title('Input')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 2, 3)
    plt.imshow(img, cmap='gray')
    ax.set_title('Noisy Image')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    # condoling the kernel with image
    new_img = cv.filter2D(src=img, ddepth=-1, kernel=kernel_u)
    ax = plt.subplot(2, 2, 4)
    plt.imshow(new_img, cmap='gray')
    ax.set_title('Denoised Image')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)



###################################################
#    Recover Images That Contain Salt and Pepper Noise   #
###################################################
# path of images
img_path_sp_o = '.\\Images\\Pepper.jpg'
img_path_sp_n = ['.\\noisy_images\\sp-{}.jpg'.format(i) for i in range(0, 4)]

# read noise free image form file
img_sp_o = cv.imread(filename=img_path_sp_o, flags=cv.IMREAD_GRAYSCALE)
# read images with gaussian noise form file
img_sp_n = [cv.imread(filename=path, flags=cv.IMREAD_GRAYSCALE) for path in img_path_sp_n]

# apply filter to all noisy images
for img in img_sp_n:
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(2, 2, 1)
    plt.imshow(img_sp_o, cmap='gray')
    ax.set_title('Input')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 2, 3)
    plt.imshow(img, cmap='gray')
    ax.set_title('Noisy Image')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    # condoling the kernel with image
    new_img = cv.medianBlur(src=img, ksize=5)
    ax = plt.subplot(2, 2, 4)
    plt.imshow(new_img, cmap='gray')
    ax.set_title('Denoised Image')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)


###################################################
#    Recover Images That Contain Strike Noise   #
###################################################
# path of images
img_path_s_o = '.\\Images\\Strike.jpg'
img_path_s_n = ['.\\noisy_images\\strike-{}.jpg'.format(i) for i in range(0, 4)]

# read noise free image form file
img_s_o = cv.imread(filename=img_path_s_o, flags=cv.IMREAD_GRAYSCALE)
# read images with gaussian noise form file
img_s_n = [cv.imread(filename=path, flags=cv.IMREAD_GRAYSCALE) for path in img_path_s_n]

# apply filter to all noisy images
for img in img_s_n:
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(2, 2, 1)
    plt.imshow(img_s_o, cmap='gray')
    ax.set_title('Input')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 2, 3)
    plt.imshow(img, cmap='gray')
    ax.set_title('Noisy Image')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    # condoling the kernel with image
    new_img = cv.medianBlur(src=img, ksize=5)
    ax = plt.subplot(2, 2, 4)
    plt.imshow(new_img, cmap='gray')
    ax.set_title('Denoised Image')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

plt.show()




