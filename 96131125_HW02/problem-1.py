import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def convolution(img, filter):
    """
    Convolving an image by a filter
    :param img: input image, uint8
    :param filter: m*n filter with float values
    :return: result of convolving image
    """
    if filter.shape[0] % 2 == 0 or filter.shape[1] % 2 == 0:
        ValueError('Filter size should be odd')
    # size of filter
    m, n = filter.shape

    # rotating filter 180 degree
    filter_90_r = np.array(list(zip(*filter[::-1])))
    filter_r = np.array(list(zip(*filter_90_r[::-1])))

    img = np.float32(img)

    # allocate an image for output
    img_out = np.zeros(shape=img.shape, dtype=np.float32)

    # pad image with zero
    img_pad = np.pad(array=img, pad_width=[(m//2,m//2),(n//2,n//2)], mode='constant', constant_values=0)

    #print(img_out.shape, img_pad.shape)
    # convolving
    p = 0
    q = 0
    for i in range(m//2, img_pad.shape[0]-(m//2)):
        for j in range(n // 2, img_pad.shape[1] - (n // 2)):
            #print(i,j, '---', p, q)
            # put filter on position i,j
            neighbour_hood = img_pad[i-(m//2):i+(m//2)+1, j-(n//2):j+(n//2)+1]

            # point-wise multiplication
            multi_neig = np.multiply(neighbour_hood, filter_r)

            # sum of products
            sum_neig = np.sum(np.sum(multi_neig))

            img_out[p, q] = sum_neig

            q = q + 1
        q = 0
        p = p + 1

    #return np.uint8(img_out)
    return img_out

# read image
img = cv.imread(filename='.\\Images\\Edge1.jpg', flags=cv.IMREAD_GRAYSCALE)

sobel_3 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
sobel_5 = np.array([[-5,-4,0,4,5],[-8,-10,0,10,8],[-10,-20,0,20,10],[-8,-10,0,10,8],[-5,-4,0,4,5]])
sobel_7 = np.array([[-3/18,-2/13,-1/10,0,1/10,2/13,3/18],[-3/13,-2/8,-1/5,0,1/5,2/8,3/13],
                    [-3/10,-2/5,-1/2,0,1/2,2/5,3/10],[-3/9,-2/4,-1/1,0,1/1,2/4,3/9],
                    [-3/10,-2/5,-1/2,0,1/2,2/5,3/10],[-3/13,-2/8,-1/5,0,1/5,2/8,3/13],
                    [-3/18,-2/13,-1/10,0,1/10,2/13,3/18]])
sobel_9 = np.array([[-4/32,-3/25,-2/20,-1/17,0/16,1/17,2/20,3/25,4/32],[-4/25,-3/18,-2/13,-1/10,0/9,1/10,2/13,3/18,4/25],
                    [-4/20,-3/13,-2/8,-1/5,0/4,1/5,2/8,3/13,4/20],[-4/17,-3/10,-2/5,-1/2,0/1,1/2,2/5,3/10,4/17],
                    [-4/16,-3/9,-2/4,-1/1,0,1/1,2/4,3/9,4/16],[-4/17,-3/10,-2/5,-1/2,0/1,1/2,2/5,3/10,4/17],
                    [-4/20,-3/13,-2/8,-1/5,0/4,1/5,2/8,3/13,4/20],[-4/25,-3/18,-2/13,-1/10,0/9,1/10,2/13,3/18,4/25],
                    [-4/32,-3/25,-2/20,-1/17,0/16,1/17,2/20,3/25,4/32]])

sobels = [sobel_3, sobel_5, sobel_7, sobel_9]
sobels_size = [3,5,7,9]

image_mags = []

# apply filters on image
for id, sobel in enumerate(sobels):
    # convolving sobel
    sobel_img_x = convolution(img=img, filter=sobel)
    sobel_img_y = convolution(img=img, filter=sobel.transpose())

    fig = plt.figure(figsize=(8, 8))
    plt.title('size {}'.format(sobels_size[id]))
    ax = plt.subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    ax.set_title('Input Image')

    ax = plt.subplot(2, 2, 2)
    plt.imshow(sobel_img_x, cmap='gray')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    ax.set_title('Sobel_X {}*{}'.format(sobels_size[id], sobels_size[id]))

    ax = plt.subplot(2, 2, 4)
    plt.imshow(sobel_img_y, cmap='gray')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    ax.set_title('Sobel_Y {}*{}'.format(sobels_size[id], sobels_size[id]))

    ax = plt.subplot(2, 2, 3)
    # magnitude
    img_mag = np.sqrt(np.float32(sobel_img_x)**2 + np.float32(sobel_img_y)**2)
    image_mags.append(img_mag)
    plt.imshow(img_mag, cmap='gray')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    ax.set_title('Sobel Magnitude {}*{}'.format(sobels_size[id], sobels_size[id]))

fig = plt.figure()
for i in range(1, 5):
    ax = plt.subplot(2, 2, i)
    plt.imshow(image_mags[i-1], cmap='gray')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    ax.set_title('Sobel Magnitude {}*{}'.format(sobels_size[i-1], sobels_size[i-1]))


plt.show()

# Explanation about sobel with different sized
# https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size/41065243#41065243
