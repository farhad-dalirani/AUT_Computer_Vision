import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


dp_md_mir_mar = ((1,80,9,350),(1,80,100,350),(1,80,9,50), (1,10,9,350), (1.1,80,9,350))

i = 0
for Dp, Md, Mir, Mar in dp_md_mir_mar:
    # read image
    input_img = cv.imread(filename='.\\Images\\Circles.jpg')
    output_img = input_img.copy()
    img = cv.cvtColor(output_img, cv.COLOR_BGR2GRAY)

    Dp = int(Dp)
    Md = int(Md)
    Mir = int(Mir)
    Mar = int(Mar)
    # detect circles
    circles = cv.HoughCircles(image=img, method=cv.HOUGH_GRADIENT, dp=Dp, minDist=Md, minRadius=Mir, maxRadius=Mar
                              , param1=195, param2=40)

    # display circles
    if circles is None:
        print('There is no circles!')
    else:
        for circle in circles:
            for x, y, r in circle:
                # Draw circles
                x, y, r = int(round(x)), int(round(y)), int(round(r))
                cv.circle(output_img, (x,y), r, (0, 255, 255), 3)
                cv.rectangle(output_img, (x - 4, y - 4), (x + 4, y + 4), (0, 0, 255), -1)

    # show the output image
    cv.imshow("output", output_img)
    cv.imwrite('.\\5\\output-{}.jpg'.format(i), output_img)
    i = i+1
cv.waitKey(0)
