import cv2
from matplotlib import pyplot as plt
import numpy as np

for i in ['00', '04', '05', '07', '08', '09']:
    img1 = cv2.imread('.\\v3\\trajectory-{}.png'.format(i), 1)         # queryImage
    img2 = cv2.imread('.\\v5\\trajectory-{}.png'.format(i), 1)         # trainImage

    #vis = np.concatenate((img1, img2), axis=1)
    dst = cv2.addWeighted(img1, 0.3, img2, 0.7, 0)

    cv2.imwrite('.\\v3-5\\comb-{}.png'.format(i), dst)


'''
for i in ['00', '01', '02', '03']:
    img1 = cv2.imread('.\\v4\\trajectory-{}.png'.format(i), 1)         # queryImage
    img2 = cv2.imread('.\\v6\\trajectory-{}.png'.format(i), 1)         # trainImage

    #vis = np.concatenate((img1, img2), axis=1)
    dst = cv2.addWeighted(img1, 0.3, img2, 0.7, 0)

    cv2.imwrite('.\\v4-6\\comb-{}.png'.format(i), dst)
'''