import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.util import random_noise


def strike_noise(img, interval, threshold_L):
    """
    Adding strike noise to image
    :param img: input image
    :param interval: [I_min, I_max]
    :return: image with strike noise
    """
    # create uniform variavle x
    uniform_x = np.random.rand(*img.shape)
    uniform_y = np.random.rand(*img.shape)

    img_new = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if uniform_x[i,j] >= threshold_L:
                img_new[i,j] = interval[0] + (interval[1]-interval[0])*uniform_y[i,j]

    # add noise
    #img_noisy = np.int32(img) + uniform_x
    # normalize intensities
    #norm_img_noisy = np.zeros(img_noisy.shape)
    #norm_img_noisy = cv.normalize(img_noisy, norm_img_noisy, 0, 255, cv.NORM_MINMAX)
    # convert to uint8
    #norm_img_noisy = np.uint8(norm_img_noisy)
    #return norm_img_noisy
    return img_new


# path of images
img_path = '.\\Images\\Strike.jpg'

# read image form file
img = cv.imread(filename=img_path, flags=cv.IMREAD_GRAYSCALE)

# different standard deviations
thresholds = [0.9, 0.9, 0.9, 0.6, 0.6, 0.6, 0.3, 0.3]
intervals = [[10,21], [10, 51], [100,121], [10,21], [10, 51], [100,121], [10,21], [10, 51]]

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
    # add strike noise
    new_img = strike_noise(img=img, interval=interval,threshold_L=thresholds[idx])
    ax = plt.subplot(3, 3, idx+2)
    ax.set_title("I_min_max {}, L= {}".format(intervals[idx], thresholds[idx]))
    plt.imshow(new_img, cmap='gray')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    # histogram
    hists.append(cv.calcHist([new_img], [0], None, [256], [0, 256]))
    cv.imwrite('.\\noisy_images\\strike-{}.jpg'.format(idx), new_img)


# plot histograms
fig = plt.figure(figsize=(10, 10))
for idx, hist in enumerate(hists):
    ax = plt.subplot(3, 3, idx + 1)
    if idx != 0:
        ax.set_title("I_min_max {}, L= {}".format(intervals[idx-1], thresholds[idx-1]), y=0.98)
    else:
        ax.set_title("Input",y=0.98)
    plt.plot(hist)
    #plt.gca().axes.get_xaxis().set_visible(False)
    #plt.gca().axes.get_yaxis().set_visible(False)

plt.show()


