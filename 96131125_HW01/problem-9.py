import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# path of image
img_paths = ['.\\Images\\blur.jpg', '.\\Images\\Pepper.jpg', '.\\Images\\Gaussian.jpeg', '.\\Images\\Cont1.jpg']

for img_path in img_paths:
    # read image form file
    img = cv.imread(filename=img_path, flags=cv.IMREAD_GRAYSCALE)

    # different A
    multiplies = [1.0, 1.05, 1.2, 1.6, 2.5]


    # display image
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(2, 3, 1)
    ax.set_title("Input Image")
    plt.imshow(img, cmap='gray')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)


    for idx, multiply in enumerate(multiplies):
        # create high boost filter
        kernel = multiply * np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32) +\
                    np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)

        # sharpening image by highboost filter
        img_sharp = cv.filter2D(src=img, ddepth=-1, kernel=kernel)

        ax = plt.subplot(2, 3, idx + 2)
        ax.set_title("A= {}".format(multiplies[idx]))
        plt.imshow(img_sharp, cmap='gray')
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)

plt.show()