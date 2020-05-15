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
        return (b[0], b[1]-1)
    if c[0] == b[0] and c[1]+1 == b[1]:
        return (b[0]-1, b[1]-1)
    if c[0]+1 == b[0] and c[1]+1 == b[1]:
        return (b[0]-1, b[1])
    if c[0]+1 == b[0] and c[1] == b[1]:
        return (b[0]-1, b[1]+1)
    if c[0]+1 == b[0] and c[1]-1 == b[1]:
        return (b[0], b[1]+1)
    if c[0] == b[0] and c[1]-1 == b[1]:
        return (b[0]+1, b[1]+1)
    if c[0]-1 == b[0] and c[1]-1 == b[1]:
        return (b[0]+1, b[1])
    if c[0]-1 == b[0] and c[1] == b[1]:
        return (b[0]+1, b[1]-1)


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
    print(input_img.shape)
    for i in range(0, input_img.shape[0]):
        for j in range(0, input_img.shape[1]):
            if input_img[i, j] == 255:
                start_point = (i, j)
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


def convert_boundary_to_grid(input_boundary_clockwise_order, space_pixels):
    """
    Get traversed clockwise boundary and put a grid on it
    and return new points in clockwise order
    :param input_boundary_clockwise_order:
    :param space_pixels: should be even, pixels between two pixels that are on grid
    :return: list of point in clockwise order
    """
    if space_pixels % 2 != 0:
        raise ValueError('space pixel should be even')

    output = []
    for point in input_boundary_clockwise_order:
        if point[0] % (space_pixels+1) == 0 and point[1] % (space_pixels+1) == 0:
            # if point on grid
            output.append(point)
        else:
            # if point isn't on grid find nearest point of grid
            x = point[0] // (space_pixels+1)
            y = point[1] // (space_pixels + 1)

            # for point aroud point on the grid
            neighbour_on_grid = [(x*(space_pixels+1), y*(space_pixels+1)),
                                 (x*(space_pixels+1), (y+1)*(space_pixels+1)),
                                 ((x+1) * (space_pixels + 1), y * (space_pixels + 1)),
                                 ((x+1)*(space_pixels+1), (y+1)*(space_pixels+1))]

            min_dis = np.inf
            nearest_neighb = None
            for neigh in neighbour_on_grid:
                distance = np.abs(point[0]-neigh[0])+np.abs(point[1]-neigh[1])
                if distance < min_dis:
                    min_dis = distance
                    nearest_neighb = cp.copy(neigh)

            if len(output) == 0:
                output.append(nearest_neighb)
            else:
                if output[-1][0] != nearest_neighb[0] or output[-1][1] != nearest_neighb[1]:
                    output.append(nearest_neighb)

    return output


def smallest_order(freeman_code_list):
    """
    This function finds smallest freeman code by rotation freeman code
    to right and examine for smallest order
    :param freemancode: should be a python list of integer number like [2,3,4, ...]
    :return:
    """
    extended_list = freeman_code_list + freeman_code_list
    for i in range(0, len(freeman_code_list)-1):
        smaller = True
        for j in range(i+1, len(freeman_code_list)):
            for k in range(0, len(freeman_code_list)):
                if extended_list[i+k] < extended_list[j+k]:
                    break
                elif extended_list[i+k] == extended_list[j+k]:
                    continue
                else:
                    smaller = False
                    break
            if smaller == False:
                break
        if smaller == True:
            best_index = i
            break
    smallest_seq = extended_list[best_index:best_index+len(freeman_code_list)]
    return smallest_seq


def freeman_code(point_on_grid_clockwise_order):
    """
    This function gets point on grid that are in clockwise order
    then it construct freeman code
    :param point_on_grid_clockwise_order:
    :return:
    """
    freeman_code_seq = []
    # compare position of each pixel to next pixel in sequence for finding associated
    # freeman code
    point = cp.copy(point_on_grid_clockwise_order)
    for i in range(0, len(point)-1):
        # find out where is next point located
        if point[i][0] == point[i+1][0] and point[i][1] < point[i+1][1]:
            freeman_code_seq.append(0)
            continue
        if point[i][0] > point[i+1][0] and point[i][1] < point[i+1][1]:
            freeman_code_seq.append(1)
            continue
        if point[i][0] > point[i+1][0] and point[i][1] == point[i+1][1]:
            freeman_code_seq.append(2)
            continue
        if point[i][0] > point[i+1][0] and point[i][1] > point[i+1][1]:
            freeman_code_seq.append(3)
            continue
        if point[i][0] == point[i+1][0] and point[i][1] > point[i+1][1]:
            freeman_code_seq.append(4)
            continue
        if point[i][0] < point[i+1][0] and point[i][1] > point[i+1][1]:
            freeman_code_seq.append(5)
            continue
        if point[i][0] < point[i+1][0] and point[i][1] == point[i+1][1]:
            freeman_code_seq.append(6)
            continue
        if point[i][0] < point[i+1][0] and point[i][1] < point[i+1][1]:
            freeman_code_seq.append(7)
            continue

    # find smallest order(independent of origin)
    freeman_code_seq_smallest = smallest_order(freeman_code_list=freeman_code_seq)

    # independent of rotation
    seq = [freeman_code_seq_smallest[-1]]+freeman_code_seq_smallest
    new_seq = []
    for i in range(1, len(seq)):
        diff = seq[i] - seq[i-1]
        if diff >= 0:
            new_seq.append(diff)
        else:
            new_seq.append(8+diff)

    # find smallest order(independent of origin)
    new_seq = smallest_order(freeman_code_list=new_seq)

    return new_seq


def draw_freeman_code(input_freeman_code, space_pixels, first_segment=2):
    """
    Get freeman code and draw it.
    :param freeman_code:
    :param space_pixels:
    :param first_segment:
    :return:
    """
    # initialize a big 2d array, it size will be reduced
    img = np.zeros(shape=(2000, 2000), dtype=np.uint8)

    cur_point = [1000, 1000]
    cum_sum = first_segment

    for i in range(0, len(input_freeman_code)):
        cum_sum = (cum_sum + input_freeman_code[i]) % 8
        pre_point = cp.copy(cur_point)
        if cum_sum == 0:
            cur_point[0] = cur_point[0]
            cur_point[1] = cur_point[1] + (space_pixels + 1)
        if cum_sum == 1:
            cur_point[0] = cur_point[0] - (space_pixels + 1)
            cur_point[1] = cur_point[1] + (space_pixels + 1)
        if cum_sum == 2:
            cur_point[0] = cur_point[0] - (space_pixels + 1)
            cur_point[1] = cur_point[1]
        if cum_sum == 3:
            cur_point[0] = cur_point[0] - (space_pixels + 1)
            cur_point[1] = cur_point[1] - (space_pixels + 1)
        if cum_sum == 4:
            cur_point[0] = cur_point[0]
            cur_point[1] = cur_point[1] - (space_pixels + 1)
        if cum_sum == 5:
            cur_point[0] = cur_point[0] + (space_pixels + 1)
            cur_point[1] = cur_point[1] - (space_pixels + 1)
        if cum_sum == 6:
            cur_point[0] = cur_point[0] + (space_pixels + 1)
            cur_point[1] = cur_point[1]
        if cum_sum == 7:
            cur_point[0] = cur_point[0] + (space_pixels + 1)
            cur_point[1] = cur_point[1] + (space_pixels + 1)

        cv.line(img, (pre_point[1],pre_point[0]), (cur_point[1], cur_point[0]), color=(255,255,255), thickness=1)

    # cut black area
    min_x = np.inf
    max_x = -1
    min_y = np.inf
    max_y = -1
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i, j] != 0:
                min_x = min([i, min_x])
                max_x = max([i, max_x])
                min_y = min([j, min_y])
                max_y = max([j, max_y])
    img = img[min_x-10:max_x+10, min_y-10:max_y+10]

    return img


if __name__ == '__main__':
    #######################################
    # part 1
    #######################################
    # read image
    img = cv.imread(filename='.\\Images\\Freeman.jpg', flags=cv.IMREAD_GRAYSCALE)
    img = img[10::, :]

    # blur image
    kernel = np.ones(shape=(5,5), dtype=np.float32) / (5*5)
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
    first_segment = [6,2,2]

    # for different grid size
    for idx, grid_size in enumerate([4, 10, 20]):
        point_on_grid_clockwise_order = convert_boundary_to_grid(input_boundary_clockwise_order=boundary_clockwise_order,
                                                                 space_pixels=grid_size)

        #print(point_on_grid_clockwise_order)

        # freeman code independent of origin and rotation
        freeman_code_of_img = freeman_code(point_on_grid_clockwise_order=point_on_grid_clockwise_order)
        print('(space between grid pixels is {}), Freeman code: \n'.format(grid_size), freeman_code_of_img)

        #######################################
        # part 2
        #######################################
        # draw freeman code
        freeman_img = draw_freeman_code(input_freeman_code=freeman_code_of_img, space_pixels=grid_size, first_segment=first_segment[idx])

        if idx == 0:
            cv.imshow('Input Image', img)
            cv.imshow('Blur Image', img_blur)
            cv.imshow('Threshold Image', img_threshold)
            cv.imshow('Edge Image', img_edge)
            cv.imwrite('.\\2\\Input Image.jpg', img)
            cv.imwrite('.\\2\\Blur Image.jpg', img_blur)
            cv.imwrite('.\\2\\Threshold Image.jpg', img_threshold)
            cv.imwrite('.\\2\\Edge Image.jpg', img_edge)

        point_on_grid = np.zeros(shape=img_edge.shape, dtype=np.uint8)
        for point in point_on_grid_clockwise_order:
            point_on_grid[point[0], point[1]] = 255
        cv.imshow('Points on grid- grid {}'.format(grid_size), point_on_grid)
        cv.imshow('Draw Freeman Code- grid {}'.format(grid_size), freeman_img)
        cv.imwrite('.\\2\\Points on grid- grid {}.jpg'.format(grid_size), point_on_grid)
        cv.imwrite('.\\2\\Draw Freeman Code- grid {}.jpg'.format(grid_size), freeman_img)

    cv.waitKey(0)

    #plt.imshow(img_edge)
    #plt.show()