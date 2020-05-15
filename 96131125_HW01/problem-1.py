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


# path of image
img_path = '.\Images\Mazandaran.jpg'

# read image form file
img = cv.imread(filename=img_path, flags=cv.IMREAD_GRAYSCALE)

# different gammas
gammas = [0.04, 0.10, 0.20, 0.40, 0.67, 1, 1.5, 2.5, 5, 10, 25]

# display image
fig = plt.figure()
ax = plt.subplot(3,4,1)
ax.set_title("Input Image")
plt.imshow(img, cmap='gray')
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)


for idx, gamma in enumerate(gammas):
    new_img = gamma_transform(img=img, c=1, gamma=gamma)
    ax = plt.subplot(3, 4, idx+2)
    ax.set_title("Gamma ({})".format(gamma))
    plt.imshow(new_img, cmap='gray')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

plt.show()



