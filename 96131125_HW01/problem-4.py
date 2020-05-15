import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.util import random_noise


def uniform_noise(img, interval):
    """
    Add uniform noise to image
    :param img: input image
    :param interval: [a,b]
    :return: image with uniform noise
    """
    # create noise
    uniform_matrix = np.random.randint(low=interval[0], high=interval[1], size=img.shape)
    # add noise
    img_noisy = np.int32(img) + uniform_matrix
    # normalize intensities
    norm_img_noisy = np.zeros(img_noisy.shape)
    norm_img_noisy = cv.normalize(img_noisy, norm_img_noisy, 0, 255, cv.NORM_MINMAX)
    # convert to uint8
    norm_img_noisy = np.uint8(norm_img_noisy)
    return norm_img_noisy


# path of images
img_path = '.\\Images\\Uniform.jpg'

# read image form file
img = cv.imread(filename=img_path, flags=cv.IMREAD_GRAYSCALE)

# different standard deviations
intervals = [[0,21], [-10,11], [0,31], [0,51], [0,81], [-60,61], [0,161], [0,191]]

# histograms
hists = []

# display image
fig = plt.figure(figsize=(10,10))
ax = plt.subplot(3, 3, 1)
ax.set_title("Input Image")
plt.imshow(img, cmap='gray')
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
# histogram
hists.append(cv.calcHist([img], [0], None, [256], [0, 256]))

for idx, interval in enumerate(intervals):
    # add uniform noise
    new_img = uniform_noise(img=img, interval=interval)
    ax = plt.subplot(3, 3, idx+2)
    ax.set_title("Interval {}".format(intervals[idx]))
    plt.imshow(new_img, cmap='gray')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    # histogram
    hists.append(cv.calcHist([new_img], [0], None, [256], [0, 256]))
    cv.imwrite('.\\noisy_images\\uniform-{}.jpg'.format(idx), new_img)


# plot histograms
fig = plt.figure(figsize=(10, 10))
for idx, hist in enumerate(hists):
    ax = plt.subplot(3, 3, idx + 1)
    if idx != 0:
        ax.set_title("Interval {}".format(intervals[idx-1]))
    else:
        ax.set_title("Input")
    plt.plot(hist)
    #plt.gca().axes.get_xaxis().set_visible(False)
    #plt.gca().axes.get_yaxis().set_visible(False)

plt.show()


