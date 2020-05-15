import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import copy as cp

def convolution(input_img, filter):
    """
    Convolving an image by a filter
    :param input_img: input image, uint8
    :param filter: m*n filter with float values
    :return: result of convolving image, caution: output is float not uint8
    """
    if filter.shape[0] % 2 == 0 or filter.shape[1] % 2 == 0:
        ValueError('Filter size should be odd')
    # size of filter
    m, n = filter.shape

    # rotating filter 180 degree
    filter_90_r = np.array(list(zip(*filter[::-1])))
    filter_r = np.array(list(zip(*filter_90_r[::-1])))

    input_img = np.float32(input_img)

    # allocate an image for output
    img_out = np.zeros(shape=input_img.shape, dtype=np.float32)

    # pad image with zero
    img_pad = np.pad(array=input_img, pad_width=[(m//2,m//2),(n//2,n//2)], mode='constant', constant_values=0)

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


def threshold_img(input_img, thresh_value):
    """
    Threshold Image
    :param input_img:
    :param thresh_value:
    :return:
    """
    img_thresholded = np.where(input_img >= thresh_value, 255, 0)
    img_thresholded = np.uint8(img_thresholded)

    return img_thresholded


def next_point_in_8_neigh(b, c):
    """
    Next point of 8-neighbourhood b in clockwise order
    :param b: center of neighbour hood
    :param c: a point in 8 neighbour hood of b
    :return: next pixel in 8 neighbour hood of b after c
    """
    if c[0]-1 == b[0] and c[1]+1 == b[1]:
        return [b[0], b[1]-1]
    if c[0] == b[0] and c[1]+1 == b[1]:
        return [b[0]-1, b[1]-1]
    if c[0]+1 == b[0] and c[1]+1 == b[1]:
        return [b[0]-1, b[1]]
    if c[0]+1 == b[0] and c[1] == b[1]:
        return [b[0]-1, b[1]+1]
    if c[0]+1 == b[0] and c[1]-1 == b[1]:
        return [b[0], b[1]+1]
    if c[0] == b[0] and c[1]-1 == b[1]:
        return [b[0]+1, b[1]+1]
    if c[0]-1 == b[0] and c[1]-1 == b[1]:
        return [b[0]+1, b[1]]
    if c[0]-1 == b[0] and c[1] == b[1]:
        return [b[0]+1, b[1]-1]


# Boundary (border) following - Gonzales 3th edition page 796
def border_following(input_img):
    """
    Following border in clockwise order
    :param input_img: image with intensity 0 and 255, in only contains border
    of one object, there should be no discontinuously in border and thickness
    of border is one.
    :return: list of position of pixels on border [(x1,y1),(x2,y2),...]
    """
    output = []

    # find upper most-left most pixel of border as starting point
    start_point = None
    #print(input_img.shape)
    for i in range(0, input_img.shape[0]):
        for j in range(0, input_img.shape[1]):
            if input_img[i, j] == 255:
                start_point = [i, j]
                break
        if start_point is not None:
            break

    #print(start_point,' <')
    if start_point is None:
        raise ValueError('Image should have pixels with 255 intensity')

    # start point
    b0 = cp.copy(start_point)
    c0 = (start_point[0], start_point[1]-1)
    #print(input_img[b0[0], b0[1]])
    #print(input_img[c0[0], c0[1]])

    output.append(b0)

    b = cp.copy(b0)
    c = cp.copy(c0)
    first_time = True

    # traverse border in clockwise order
    while (b[0] != b0[0] or b[1] != b0[1]) or first_time == True:
        # next point on border
        while True:
            next_point = next_point_in_8_neigh(b=b, c=c)
            if input_img[next_point[0], next_point[1]] == 0:
                c = cp.copy(next_point)
            else:
                b = cp.copy(next_point)
                break

        output.append(b)
        first_time = False

    return output


if __name__ == '__main__':

    # read image
    img = cv.imread(filename='.\\Images\\Fourier.jpg', flags=cv.IMREAD_GRAYSCALE)

    # blur image
    kernel = np.ones(shape=(3,3), dtype=np.float32) / (3*3)
    img_blur = convolution(input_img=img, filter=kernel)
    img_blur = np.uint8(img_blur)

    # threshold image
    img_threshold = threshold_img(input_img=img_blur, thresh_value=5)

    # calculate edges
    # with this filter, thickness of edge is 1
    kernel = np.array([0, -1, 1], dtype=np.float32)
    kernel = np.reshape(kernel, newshape=(1, 3))
    img_edge_hor = convolution(input_img=img_threshold, filter=kernel)
    img_edge_hor = np.uint8(np.abs(img_edge_hor))
    img_edge_ver = convolution(input_img=img_threshold, filter=kernel.transpose())
    img_edge_ver = np.uint8(np.abs(img_edge_ver))
    img_edge = np.uint8(img_edge_ver + img_edge_hor)
    img_edge = threshold_img(input_img=img_edge, thresh_value=100)

    # travers boundary in clockwise order
    boundary_clockwise_order = border_following(input_img=img_edge)
    print('Boundary points, Clockwise order:\n', boundary_clockwise_order)

    # for different p
    for idx, p in enumerate([2, 4, 6, 8, 12, 18, 30, 60, 90, 140, 200, 300, 700, 930]):
        # convert (x,y) point on boundary to x+jy points
        complex_b_c_o  = []
        for point in boundary_clockwise_order:
            complex_b_c_o.append(complex(point[0], point[1]))
        complex_b_c_o = np.array(complex_b_c_o)

        complex_b_c_o = (complex_b_c_o)
        fourier_complex_b_c_o = np.fft.fft(complex_b_c_o)
        fourier_complex_b_c_o = np.fft.fftshift(fourier_complex_b_c_o)

        # zeroing f coefficient
        print(len(fourier_complex_b_c_o))
        for i in range(0, len(fourier_complex_b_c_o)):
           if abs(i-465) > p//2:
                fourier_complex_b_c_o[i] = 0

        fourier_complex_b_c_o = np.fft.ifftshift(fourier_complex_b_c_o)
        inverse_fourier_b_c_o = np.fft.ifft(fourier_complex_b_c_o)

        reconstructed_points_on_boundary = []
        for complex_number in inverse_fourier_b_c_o:
            reconstructed_points_on_boundary.append([int(round(np.real(complex_number))), int(round(np.imag(complex_number)))])

        print(reconstructed_points_on_boundary)
        point_on_reconstructed_boundary = np.zeros(shape=img_edge.shape, dtype=np.uint8)
        for point in reconstructed_points_on_boundary:
            point_on_reconstructed_boundary[point[0], point[1]] = 255

        if idx == 0:
            cv.imshow('Input Image', img)
            cv.imshow('Blur Image', img_blur)
            cv.imshow('Threshold Image', img_threshold)
            cv.imshow('Edge Image', img_edge)
            cv.imwrite('.\\3\\Input Image.jpg', img)
            cv.imwrite('.\\3\\Blur Image.jpg', img_blur)
            cv.imwrite('.\\3\\Threshold Image.jpg', img_threshold)
            cv.imwrite('.\\3\\Edge Image.jpg', img_edge)

        cv.imshow('Fourier Descriptor-p {}'.format(p), point_on_reconstructed_boundary)
        cv.imwrite('.\\3\\Fourier Descriptor-p {}.jpg'.format(p), point_on_reconstructed_boundary)

    cv.waitKey(0)

