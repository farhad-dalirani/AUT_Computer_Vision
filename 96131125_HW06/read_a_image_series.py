import cv2 as cv
import os, os.path
import numpy as np


def read_an_image_series(series='birds'):
    """
    Read a series of images
    :param series:
    :return:
    """
    # general path that contain all images of a series
    main_path = 'C:\\Users\\Home\\Dropbox\\codes\\CV\\96131125_HW06\\JPEGS_min\\JPEGS\\'

    # specific path of a series
    path_of_series = main_path + series + '\\'

    # number of images
    images_name = os.listdir(path_of_series)
    num_images = len(images_name)

    # images
    imgs = []

    # read all images of sequence
    for i in range(1, num_images+1):
        # read image
        name_of_image = 'frame_{}.jpg'.format(i)

        img = cv.imread(filename=path_of_series+name_of_image, flags=cv.IMREAD_GRAYSCALE)
        imgs.append(img)

    # return all images of sequence
    imgs = np.stack(imgs)
    return imgs


# test function
if __name__ == '__main__':

    # read a sequence of images
    imgs = read_an_image_series(series='birds')

    cv.imshow('birds 1', imgs[0])
    cv.imshow('birds 10', imgs[10])

    cv.waitKey(0)
    cv.destroyAllWindows()