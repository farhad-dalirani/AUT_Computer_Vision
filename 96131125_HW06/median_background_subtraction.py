import cv2 as cv
import numpy as np
from read_a_image_series import read_an_image_series


def median_background_subtraction_v1(img_seq, k=5):
    """
    Use median background subtraction for finding background in image sequence
    :param img_seq: sequence of images
    :param k: use last k image for finding background
    :return:
    """
    # median of k last images in sequence
    median_of_image = np.median(img_seq[0:k, :, :], axis=0)

    outs = []

    # find background in frame k+1 by using median of
    # k previous frame
    for i in range(k, img_seq.shape[0]):
        # calculate difference of frame k+1 and median of k
        # previous frames
        diff = np.float32(img_seq[i,:,:])-np.float32(median_of_image)
        diff = diff - np.min(diff)
        diff = cv.normalize(diff, diff, 255, 0, cv.NORM_MINMAX)

        # Otsu's thresholding after Gaussian filtering
        #blur = cv.GaussianBlur(diff, (3, 3), 0)
        ret, th = cv.threshold(np.uint8(diff), 100, 255, cv.THRESH_BINARY)
        diff = th

        #ret, th = cv.threshold(np.uint8(diff), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        #diff = th

        # median of k last images in sequence
        median_of_image = np.median(img_seq[i-k:i+1, :, :], axis=0)

        outs.append(diff)

    #cv.imshow('med', np.uint8(diff))
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    return outs


if __name__ == '__main__':

    # name of different sequence of image
    seq_names = ['birds', 'boats', 'bottle', 'cyclists', 'surf']

    # find background of each sequence
    for seq_name in seq_names:
        # read a sequence of images
        imgs = read_an_image_series(series=seq_name)

        # do background subtraction by median
        k=15
        out_seq = median_background_subtraction_v1(img_seq=imgs, k=k)
        for i in range(0, len(out_seq)):
            cv.imwrite('.\\median_output_v1\\{}\\frame_{}_out.jpg'.format(seq_name, i), np.hstack((out_seq[i], imgs[i+k,:,:])))

