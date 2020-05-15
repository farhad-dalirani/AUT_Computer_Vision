import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def generate_filter(sigma, n):
    """
    Generate Laplacian of a Gaussian with arbitrary size and
    standard deviation
    :param sigma: Standard Deviation
    :param n: size
    :return: n*n filter
    """
    if n == 0:
        ValueError('Filter size should be odd')

    filter = np.zeros(shape=(n,n), dtype=np.float32)
    for i in range(0, n):
        for j in range(0, n):
            x = i - (n//2)
            y = j - (n // 2)

            filter[i, j] = ((x**2+y**2-2*(sigma**2))/(sigma**4))*np.exp(-(x**2+y**2)/(2*sigma**2))
    return filter


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


###########################
#   Effect of changing size
###########################
# read image
img = cv.imread(filename='.\\Images\\Edge2.jpg', flags=cv.IMREAD_GRAYSCALE)

# details of LoG filters
LoGs_details = [(0.8, 3), (0.8, 5), (0.8, 7), (0.8, 17)]
LoGs = [generate_filter(sigma=sigma, n=n) for sigma, n in LoGs_details]

#print(LoGs[0],'\n',LoGs[1])

images = []

# apply filters on image
for id, log in enumerate(LoGs):
    # convolving LOG
    log_img = convolution(img=img, filter=log)
    images.append(log_img)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    ax.set_title('Input Image')

    ax = plt.subplot(1, 2, 2)
    plt.imshow(log_img, cmap='gray')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    ax.set_title('LOG {}*{} sigma={}'.format(LoGs_details[id][1], LoGs_details[id][1], LoGs_details[id][0]))


fig = plt.figure()
for i in range(1, 5):
    ax = plt.subplot(2, 2, i)
    plt.imshow(images[i-1], cmap='gray')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    ax.set_title('LOG {}*{} sigma={}'.format(LoGs_details[i-1][1], LoGs_details[i-1][1], LoGs_details[i-1][0]))


###########################
#   Effect of changing sigma
###########################
# read image
img = cv.imread(filename='.\\Images\\Edge2.jpg', flags=cv.IMREAD_GRAYSCALE)

# details of LoG filters
LoGs_details = [(0.8, 13), (1.2, 13), (1.6, 13), (2, 13)]
LoGs = [generate_filter(sigma=sigma, n=n) for sigma, n in LoGs_details]

#print(LoGs[0],'\n',LoGs[1])

images = []

# apply filters on image
for id, log in enumerate(LoGs):
    # convolving LOG
    log_img = convolution(img=img, filter=log)
    images.append(log_img)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    ax.set_title('Input Image')

    ax = plt.subplot(1, 2, 2)
    plt.imshow(log_img, cmap='gray')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    ax.set_title('LOG {}*{} sigma={}'.format(LoGs_details[id][1], LoGs_details[id][1], LoGs_details[id][0]))


fig = plt.figure()
for i in range(1, 5):
    ax = plt.subplot(2, 2, i)
    plt.imshow(images[i-1], cmap='gray')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    ax.set_title('LOG {}*{} sigma={}'.format(LoGs_details[i-1][1], LoGs_details[i-1][1], LoGs_details[i-1][0]))



plt.show()
