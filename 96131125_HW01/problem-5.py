import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.util import random_noise

# path of images
img_path = '.\\Images\\Pepper.jpg'

# read image form file
img = cv.imread(filename=img_path, flags=cv.IMREAD_GRAYSCALE)

# different standard deviations
powers = [0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]

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

for idx, power in enumerate(powers):
    # add salt and pepper noise
    new_img = random_noise(image=img, mode='s&p', amount =power)
    new_img = np.uint8(new_img * 255)
    ax = plt.subplot(3, 3, idx+2)
    ax.set_title("Power ({})".format(powers[idx]))
    plt.imshow(new_img, cmap='gray')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    # histogram
    #print(new_img)
    hists.append(cv.calcHist([new_img], [0], None, [256], [0, 256]))
    cv.imwrite('.\\noisy_images\\sp-{}.jpg'.format(idx), new_img)


# plot histograms
fig = plt.figure(figsize=(10, 10))
for idx, hist in enumerate(hists):
    ax = plt.subplot(3, 3, idx + 1)
    if idx != 0:
        ax.set_title("Power ({})".format(powers[idx-1]))
    else:
        ax.set_title("Input")
    plt.plot(hist)
    #plt.gca().axes.get_xaxis().set_visible(False)
    #plt.gca().axes.get_yaxis().set_visible(False)

plt.show()


