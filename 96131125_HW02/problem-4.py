import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def Hough_circles(image, minDist, minRadius, maxRadius):
    """
    Hough Circle Transform
    :param image: input image
    :param minDist: minimum distance between center of circles
    :param minRadius: minimum acceptable radius
    :param maxRadius: maximum acceptable radius
    :return: circles [(x,y,r), (x,y,r), ...]
    """
    # Formula for circle: (x-c1)**2 + (x-c2)**2 = c3**2

    # detecting edges with Canny's algorithm
    edge_img = cv.Canny(image=img, threshold1=100, threshold2=250)
    edge_bin = cv.adaptiveThreshold(edge_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 0)
    #cv.imshow('edges', edge_bin)
    #cv.waitKey(0)
    print(edge_bin.shape)
    print(np.max(edge_bin))
    print(np.sum(np.sum(edge_bin))/255, ' - ', edge_bin.shape[0]*edge_bin.shape[1])

    # create accumulator cell
    A = np.zeros(shape=(image.shape[0], image.shape[1], maxRadius-minRadius+1))
    #print(A)

    # for each point on edges find circles that pass that point
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            print((i,j))

            # if point i,j isn't on edges go to next point
            if edge_bin[i,j] != 255:
                continue

            # assume circle with center (c1,c2) pass point (i,j) find radius (c3) of circle
            for c1 in range(0, image.shape[0]):
                for c2 in range(0, image.shape[1]):
                    # calculating radius of circle which is centered in (c1,c2) and pass point (i,j)
                    c3 = np.sqrt(((i-c1)**2 + (j-c2)**2))
                    # round c3 for putting it in accumulator cells
                    c3 = int(round(c3))

                    # if radius isn't in the valid range ignore it
                    if c3 > maxRadius or c3 < minRadius:
                        continue

                    # add one to accumulator cells c1,c2,c3
                    A[c1,c2, c3-minRadius] = A[c1,c2, c3-minRadius] + 1

    circles = []
    # keep circles that pass more that (2*pi*(r**2)) / 5 pixels
    for c1 in range(0, image.shape[0]):
        for c2 in range(0, image.shape[1]):
            for c3 in range(maxRadius-minRadius+1):
                if A[c1, c2, c3] >= 10:
                    circles.append((c1, c2, c3+minRadius))
    return circles

# read image
input_img = cv.imread(filename='.\\Images\\Circles.jpg')
factor = 0.3
input_img = cv.resize(input_img, (0,0), fx=factor, fy=factor)
output_img = input_img.copy()
img = cv.cvtColor(output_img, cv.COLOR_BGR2GRAY)

# call hough circles
circles = Hough_circles(image=img, minDist=int(80*factor), minRadius=int(10*factor), maxRadius=int(160*factor))
print(circles)

# display circles
if circles is None:
    print('There is no circles!')
else:
    for circle in circles:
         x, y, r = circle
         # Draw circles
         x, y, r = int(round(x)), int(round(y)), int(round(r))
         cv.circle(output_img, (x,y), r, (0, 255, 255), 1)
         cv.rectangle(output_img, (x - 2, y - 2), (x + 2, y + 2), (0, 0, 255), -1)

# show the output image
cv.imshow("output", output_img)
cv.imwrite('.\\5\\output.jpg', output_img)
cv.waitKey(0)
