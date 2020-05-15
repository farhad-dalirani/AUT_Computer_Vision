import numpy as np
import cv2 as cv
import os
from monocular_visual_odometry_v4 import MonocularVisualOdometry, do_visual_odometry

# sequence '00', '04', '05', '07', '08', '09',...
seq = '02'

sequence_start_point = {'00': (300,250), '01': (10,600), '02': (100,100), '03': (100, 100), '04': (100, 100), '05': (300, 300),
                        '06': (300, 300), '07': (300, 300), '08': (400, 300), '09': (400, 300), '10': (50, 300)}
sequence_scale_point = {'00': 1.0, '01': 0.3, '02': 0.75, '03': 1.0, '04': 1.0, '05': 1.0,
                        '06': 1.0, '07': 1.0, '08': 0.8, '09': 0.8, '10': 1.0}


# path of ground truth
path_ground_truth = 'D:\\Vision\\KITTI Dataset\\data_odometry_poses\\dataset\\poses\\{}.txt'.format(seq)

# path of gray-scale images
path_images = 'D:\\Vision\\KITTI Dataset\\data_odometry_gray\\dataset\\sequences\\{}\\image_0\\'.format(seq)

# read first image for size information
img_first = cv.imread(path_images + str(0).zfill(6) + '.png', cv.IMREAD_GRAYSCALE)

# camera parameters
cam = {'width': img_first.shape[1], 'height': img_first.shape[0],
       'fx': 718.8560, 'fy': 718.8560,
       'cx': 607.1928, 'cy': 185.2157,
       'k1': 0.0, 'k2': 0.0, 'p1': 0.0,
       'p2': 0.0, 'k3': 0.0}

# number of images
images_name = os.listdir(path_images)
num_images = len(images_name)

# monocular visual odometry object
monocular_visual_odometry = MonocularVisualOdometry(cam, path_ground_truth)

# do visual odometry
do_visual_odometry(monocular_visual_odometry=monocular_visual_odometry, path_images=path_images, num_images=num_images,
                   sequence_start_point=sequence_start_point, sequence_scale_point=sequence_scale_point,
                   seq=seq)