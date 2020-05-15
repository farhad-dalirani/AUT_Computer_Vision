import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.util import random_noise

# all noisy images were saved in '.\\noisy_images\\'

# path of images
img_path_g_o = '.\\Images\\Gaussian.jpeg'
img_path_g_n = ['.\\noisy_images\\gaussain-{}.jpg'.format(i) for i in range(0, 4)]

# read noise free image form file
img_g_o = cv.imread(filename=img_path_g_o, flags=cv.IMREAD_GRAYSCALE)
# read images with gaussian noise form file
img_g_n = [cv.imread(filename=path, flags=cv.IMREAD_GRAYSCALE) for path in img_path_g_n]

img_path_u_o = '.\\Images\\Uniform.jpg'
img_path_u_n = ['.\\noisy_images\\uniform-{}.jpg'.format(i) for i in range(1, 5)]

# read noise free image form file
img_u_o = cv.imread(filename=img_path_u_o, flags=cv.IMREAD_GRAYSCALE)
# read images with gaussian noise form file
img_u_n = [cv.imread(filename=path, flags=cv.IMREAD_GRAYSCALE) for path in img_path_u_n]

# path of images
img_path_sp_o = '.\\Images\\Pepper.jpg'
img_path_sp_n = ['.\\noisy_images\\sp-{}.jpg'.format(i) for i in range(0, 4)]

# read noise free image form file
img_sp_o = cv.imread(filename=img_path_sp_o, flags=cv.IMREAD_GRAYSCALE)
# read images with gaussian noise form file
img_sp_n = [cv.imread(filename=path, flags=cv.IMREAD_GRAYSCALE) for path in img_path_sp_n]

# path of images
img_path_s_o = '.\\Images\\Strike.jpg'
img_path_s_n = ['.\\noisy_images\\strike-{}.jpg'.format(i) for i in range(0, 4)]

# read noise free image form file
img_s_o = cv.imread(filename=img_path_s_o, flags=cv.IMREAD_GRAYSCALE)
# read images with gaussian noise form file
img_s_n = [cv.imread(filename=path, flags=cv.IMREAD_GRAYSCALE) for path in img_path_s_n]

# all noisy and noise free images
img_o = [img_g_o, img_u_o, img_sp_o, img_s_o]
img_n = [img_g_n, img_u_n, img_sp_n, img_s_n]


# different sigma for gaussian filter
sigmas = [0.05, 0.3, 0.8, 1.3]

# apply filter to all noisy images
for index in range(len(img_n)):
    for img in img_n[index]:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(2, 4, 1)
        plt.imshow(img_o[index], cmap='gray')
        ax.set_title('Input')
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

        ax = plt.subplot(2, 4, 2)
        plt.imshow(img, cmap='gray')
        ax.set_title('Noisy Image')
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

        # convolving the kernel with image
        for idx, sigma in enumerate(sigmas):
            new_img = np.zeros(shape=img.shape)
            new_img = cv.GaussianBlur(src=img, dst=new_img, ksize=(5, 5), sigmaX=sigma, sigmaY=sigma)
            ax = plt.subplot(2, 4, 5+idx)
            plt.imshow(new_img, cmap='gray')
            ax.set_title('Denoised Image, sigma {}'.format(sigma))
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)

plt.show()




