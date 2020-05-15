import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def correlation(img, filter):
    """
    Correlation an image by a filter
    :param img: input image, uint8
    :param filter: m*n filter with float values
    :return: result of correlation image
    """
    if filter.shape[0] % 2 == 0 or filter.shape[1] % 2 == 0:
        ValueError('Filter size should be odd')
    # size of filter
    m, n = filter.shape

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
            multi_neig = np.multiply(neighbour_hood, filter)

            # sum of products
            sum_neig = np.sum(np.sum(multi_neig))

            img_out[p, q] = sum_neig

            q = q + 1
        q = 0
        p = p + 1

    #return np.uint8(img_out)
    return img_out


#######################################
# part 1
#######################################
# read image
img = cv.imread(filename='.\\Images\\Scene.jpg', flags=cv.IMREAD_GRAYSCALE)

# read template
template = cv.imread(filename='.\\Images\\Template.png', flags=cv.IMREAD_GRAYSCALE)

templates = [template[0:-1, :], template[0:11, :], template[8:27, 22::], template[12:21, 12:21]]
thresholds = [162, 164, 160, 160]

# for different templates
for idx, img_t in enumerate(templates):
    # find template in image with correlation
    corr = correlation(img=img, filter=img_t)
    normalized_corr = np.zeros(corr.shape)
    normalized_corr = cv.normalize(corr, normalized_corr, 0, 255, cv.NORM_MINMAX)
    normalized_corr = np.uint8(normalized_corr)

    # threshold
    threshold_corr = np.where(normalized_corr >= thresholds[idx], 255, 0)
    threshold_corr = np.uint8(threshold_corr)

    # cups in image
    cups = np.multiply(np.double(threshold_corr)/255, img)
    cups = np.uint8(cups)

    if idx == 0:
        cv.imshow('Input Image', img)

    cv.imshow('Template- {}'.format(idx), img_t)
    cv.imshow('Correlation- {}'.format(idx), normalized_corr)
    cv.imshow('Threshold Correlation- {}'.format(idx), threshold_corr)
    cv.imshow('Cups In Image- {}'.format(idx), cups)
    cv.imwrite('.\\1\\Template- {}.jpg'.format(idx), img_t)
    cv.imwrite('.\\1\\Correlation- {}.jpg'.format(idx), normalized_corr)
    cv.imwrite('.\\1\\Threshold Correlation- {}.jpg'.format(idx), threshold_corr)
    cv.imwrite('.\\1\\Cups In Image- {}.jpg'.format(idx), cups)

cv.waitKey(0)
