import numpy as np
import cv2 as cv
from sklearn import linear_model
import copy as cp

def do_visual_odometry(monocular_visual_odometry, path_images, num_images, sequence_start_point, sequence_scale_point, seq):
    """
    use MonocularVisualOdometry class and its functions for finding path of
    moving camera
    :param monocular_visual_odometry:
    :param path_images:
    :param num_images:
    :param sequence_start_point:
    :param sequence_scale_point:
    :param seq:
    :return:
    """

    # trajectory image
    trajectory_image = np.zeros((800, 800, 3), dtype=np.uint8)

    # iterate all image in sequence
    for img_id in range(num_images):

        # read i'th image from file
        img_i_th = cv.imread(path_images + str(img_id).zfill(6) + '.png', cv.IMREAD_GRAYSCALE)

        # update rotation and translation matrix by i'th image
        monocular_visual_odometry.update_rotation_translation_matrix_of_camera(img_i_th, img_id)

        # draw current position of camera
        cur_t = monocular_visual_odometry.first_to_i_th_frame_T
        if img_id > 1:
            x, y, z = cur_t[0], cur_t[1], cur_t[2]
        else:
            x, y, z = 0.0, 0.0, 0.0

        # pad trajectory from corner of image
        pad_x, pad_y = sequence_start_point[seq][0], sequence_start_point[seq][1]
        scale = sequence_scale_point[seq]

        draw_x, draw_y = int(x * scale) + pad_x, int(z * scale) + pad_y
        true_x, true_y = int(monocular_visual_odometry.ground_truth_current_x * scale) + pad_x, \
                         int(monocular_visual_odometry.ground_truth_current_z * scale) + pad_y

        cv.circle(trajectory_image, (draw_x, draw_y), 1, (128, 255, 0), 1)
        cv.circle(trajectory_image, (true_x, true_y), 1, (255, 0, 255), 2)
        cv.putText(trajectory_image,
                   'Purple: Ground Truth - Green: Output - seq {} - scale of drawing {}'.format(seq, scale),
                   (30, 770), cv.FONT_HERSHEY_PLAIN, 1, (0, 128, 255), 1, 8)

        cv.imshow('camera', img_i_th)
        cv.imshow('trajectory-{}'.format(seq), trajectory_image)
        cv.waitKey(1)

    # save output as an image
    cv.imwrite('trajectory-{}.png'.format(seq), trajectory_image)

    # close open windows
    cv.destroyAllWindows()


def key_points_matching(des1, des2, kp1, kp2, coor1, coor2):
    """
    Finds key points of frame i in frame i+1
    :param prevImg:
    :param nextImg:
    :param prevPts:
    :return:
    """

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good.append([m])

    keypoint1_coor = []
    keypoint2_coor = []
    keypoint1_des = []
    keypoint2_des = []
    keypoint1_key = []
    keypoint2_key = []

    for instance in good:
        #print('---')
        #print(instance)
        #print(instance[0])
        #print(instance[0].queryIdx)
        #print(coor1[instance[0].queryIdx])
        #print(coor2[instance[0].trainIdx])
        #print('---')

        #print(instance[0].queryIdx)
        #print(coor1.shape)

        keypoint1_coor.append(coor1[instance[0].queryIdx])
        keypoint2_coor.append(coor2[instance[0].trainIdx])

        keypoint1_des.append(des1[instance[0].queryIdx])
        keypoint2_des.append(des2[instance[0].trainIdx])

        keypoint1_key.append(kp1[instance[0].queryIdx])
        keypoint2_key.append(kp2[instance[0].trainIdx])


    keypoint1_coor_np = keypoint1_coor[0]
    for i in range(1, len(keypoint1_coor)):
        keypoint1_coor_np = np.vstack((keypoint1_coor_np, keypoint1_coor[i]))

    keypoint2_coor_np = keypoint2_coor[0]
    for i in range(1, len(keypoint2_coor)):
        keypoint2_coor_np = np.vstack((keypoint2_coor_np, keypoint2_coor[i]))

    keypoint1_kp_np = keypoint1_key[0]
    for i in range(1, len(keypoint1_key)):
        keypoint1_kp_np = np.vstack((keypoint1_kp_np, keypoint1_key[i]))

    keypoint2_kp_np = keypoint2_key[0]
    for i in range(1, len(keypoint2_key)):
        keypoint2_kp_np = np.vstack((keypoint2_kp_np, keypoint2_key[i]))

    keypoint1_des_np = keypoint1_des[0]
    for i in range(1, len(keypoint1_des)):
        keypoint1_des_np = np.vstack((keypoint1_des_np, keypoint1_des[i]))

    keypoint2_des_np = keypoint2_des[0]
    for i in range(1, len(keypoint2_des)):
        keypoint2_des_np = np.vstack((keypoint2_des_np, keypoint2_des[i]))


    return keypoint1_coor_np, keypoint2_coor_np, keypoint1_kp_np, keypoint2_kp_np, keypoint1_des_np, keypoint2_des_np,


class MonocularVisualOdometry:
    def __init__(self, cam, ground_truth):
        """
        Initiates parameters, read ground truth and ...
        :param cam:
        :param ground_truth:
        """
        # camera parameters
        self.cam = cam

        # input frame
        self.new_frame = None

        # previous frame in sequence of input frame
        self.prev_frame = None

        # rotation and translation matrix from first image to i'th image
        self.first_to_i_th_frame_R = None
        self.first_to_i_th_frame_T = None

        # key points in precious and current frame
        self.ref_keypoints_coor = None
        self.tracked_keypoints_coor = None

        self.ref_keypoints_key = None
        self.tracked_keypoints_key = None

        self.ref_keypoints_des = None
        self.tracked_keypoints_des = None


        # Focal length in pixels.
        self.focal = cam['fx']

        # Optical center
        self.principal_point = (cam['cx'], cam['cy'])

        # ground truth current position of camera
        self.ground_truth_current_x, self.ground_truth_current_y, self.ground_truth_current_z = 0, 0, 0

        # feature detector cv2.xfeatures2d
        #self.feature_detector = cv.ORB_create(nfeatures=3500, nlevels=32)
        self.feature_detector = cv.xfeatures2d.SURF_create()

        # read ground truth poses (trajectory)
        with open(ground_truth) as f:
            self.ground_truth = f.readlines()

        # classifier for eliminating bad features(blob or corners)
        self.classifier = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)

    def get_absolute_scale(self, frame_id):
        """
        Calculates absolute scale
        :param frame_id:
        :return:
        """
        # open up ground truth for (i-1)'th frame
        opened_string = self.ground_truth[frame_id - 1].strip().split()
        # position of camera in (i-1)'th frame
        x_prev = float(opened_string[3])
        y_prev = float(opened_string[7])
        z_prev = float(opened_string[11])

        # open up ground truth for (i)'th frame
        opened_string = self.ground_truth[frame_id].strip().split()
        # position of camera in (i)'th frame
        x_cur = float(opened_string[3])
        y_cur = float(opened_string[7])
        z_cur = float(opened_string[11])

        # update correct and current position of camera
        self.ground_truth_current_x, self.ground_truth_current_y, self.ground_truth_current_z = x_cur, y_cur, z_cur

        # calculate absolute scale
        abs_scale = np.sqrt(((x_cur - x_prev)**2) + ((y_cur - y_prev)**2) + ((z_cur - z_prev)**2))
        return abs_scale

    def update_rotation_translation_matrix_of_camera(self, img, frame_id):
        """
        By receiving a i'th image, track features of (i-1)'th image in
        image i'th. then calculate Essential Matrix.
        By using essential matrix obtain position of camera.
        :param img:
        :param frame_id:
        :return:
        """
        # check size of camera model and input image are the same
        if not(img.ndim == 2 and img.shape[0] == self.cam['height'] and img.shape[1] == self.cam['width']):
            raise ValueError('Given frame is not gray-scale or size of frame is not equal camera model!')

        # update new frame by input image
        self.new_frame = img

        # if image is first frame of sequence
        if frame_id == 0:
            # for first image in sequence just calculate key-points.

            # calculate key-points
            self.ref_keypoints_key = self.feature_detector.detect(self.new_frame)

            # calculate descriptors
            kp, des = self.feature_detector.compute(img, self.ref_keypoints_key)

            self.ref_keypoints_key = kp
            self.ref_keypoints_des = des

            # coordinates of key-points
            self.ref_keypoints_coor = np.array([x.pt for x in self.ref_keypoints_key], dtype=np.float32)

        # if image is second frame of sequence
        elif frame_id == 1:
            # for second image in sequence, track key-points of first image, then calculate essential matrix
            # then obtain Rotation and Translation Matrix

            # calculate key-points
            self.tracked_keypoints_key = self.feature_detector.detect(self.new_frame)

            # calculate descriptors
            kp, des = self.feature_detector.compute(img, self.tracked_keypoints_key)

            self.tracked_keypoints_key = kp
            self.tracked_keypoints_des = des

            # coordinates of key-points
            self.tracked_keypoints_coor = np.array([x.pt for x in self.tracked_keypoints_key], dtype=np.float32)
            tracked_keypoints_coor_copy = np.copy(self.tracked_keypoints_coor)
            tracked_keypoints_kp_copy = np.copy(self.tracked_keypoints_key)
            tracked_keypoints_des_copy = np.copy(self.tracked_keypoints_des)
            #############################

            # find matches of feattures of previous frame in current frame
            self.ref_keypoints_coor, self.tracked_keypoints_coor ,\
            self.ref_keypoints_key, self.tracked_keypoints_key ,\
            self.ref_keypoints_des, self.tracked_keypoints_des = key_points_matching(des1=self.ref_keypoints_des, des2=self.tracked_keypoints_des,
                                                                                       coor1=self.ref_keypoints_coor, coor2=self.tracked_keypoints_coor,
                                                                                       kp1=self.ref_keypoints_key, kp2=self.tracked_keypoints_key)

            # calculate an essential matrix from the corresponding points in two images
            essential_matrix, mask_inliers = cv.findEssentialMat(points1=self.tracked_keypoints_coor, points2=self.ref_keypoints_coor,
                                                                 focal=self.focal, pp=self.principal_point,
                                                                 method=cv.RANSAC, prob=0.999, threshold=1.0)

            # recover relative camera rotation and translation from an estimated essential matrix
            # and the corresponding points in two images
            _, self.first_to_i_th_frame_R, self.first_to_i_th_frame_T, mask_inliers = \
                cv.recoverPose(E=essential_matrix, points1=self.tracked_keypoints_coor, points2=self.ref_keypoints_coor,
                               focal=self.focal, pp=self.principal_point)

            # store tracked key-points as keypoint of input image
            self.ref_keypoints_coor = tracked_keypoints_coor_copy
            self.ref_keypoints_des = tracked_keypoints_des_copy
            self.ref_keypoints_key = tracked_keypoints_kp_copy

        # if image is not first or second frame of sequence
        elif frame_id > 1:
            # for i'th image in sequence, track key-points of (i-1)'th image, then calculate essential matrix
            # then obtain Rotation and Translation Matrix

            # calculate key-points
            self.tracked_keypoints_key = self.feature_detector.detect(self.new_frame)

            # calculate descriptors
            kp, des = self.feature_detector.compute(img, self.tracked_keypoints_key)

            self.tracked_keypoints_key = kp
            self.tracked_keypoints_des = des

            # coordinates of key-points
            self.tracked_keypoints_coor = np.array([x.pt for x in self.tracked_keypoints_key], dtype=np.float32)

            #print(type(self.tracked_keypoints_key))
            #print(type(self.tracked_keypoints_des))
            #print(type(self.tracked_keypoints_coor))

            # use classifier
            if frame_id > 100 and self.tracked_keypoints_des.shape[0] > 2000:
                predicted_des = self.classifier.predict(self.tracked_keypoints_des)
                #print(predicted_des)
                #print(np.sum(predicted_des))
                #print(len(predicted_des)-np.sum(predicted_des))

                # delete bad features
                good_indexes = []
                for i in range(len(predicted_des)):
                    if predicted_des[i] == 1:
                        good_indexes.append(i)
                good_indexes = np.array(good_indexes)

                if np.sum(predicted_des) > 2000:
                    #print('=======')
                    #print(np.sum(predicted_des))
                    #print(len(predicted_des)-np.sum(predicted_des))
                    #print('=======')
                    self.tracked_keypoints_coor = self.tracked_keypoints_coor[good_indexes]
                    temp = []
                    for i in range(good_indexes.shape[0]):
                        temp.append(self.tracked_keypoints_key[good_indexes[i]])
                    self.tracked_keypoints_key = cp.copy(temp)
                    self.tracked_keypoints_des = self.tracked_keypoints_des[good_indexes]

            tracked_keypoints_coor_copy = np.copy(self.tracked_keypoints_coor)
            tracked_keypoints_kp_copy = np.copy(self.tracked_keypoints_key)
            tracked_keypoints_des_copy = np.copy(self.tracked_keypoints_des)
            #############################

            self.ref_keypoints_coor, self.tracked_keypoints_coor, \
            self.ref_keypoints_key, self.tracked_keypoints_key, \
            self.ref_keypoints_des, self.tracked_keypoints_des = key_points_matching(des1=self.ref_keypoints_des, des2=self.tracked_keypoints_des,
                                                                                       coor1=self.ref_keypoints_coor, coor2=self.tracked_keypoints_coor,
                                                                                       kp1=self.ref_keypoints_key, kp2=self.tracked_keypoints_key)

            # calculate an essential matrix from the corresponding points in two images
            essential_matrix, mask_inliers_ess = cv.findEssentialMat(points1=self.tracked_keypoints_coor, points2=self.ref_keypoints_coor,
                                                                 focal=self.focal, pp=self.principal_point,
                                                                 method=cv.RANSAC, prob=0.999, threshold=1.0)

            # recover relative camera rotation and translation from an estimated
            # essential matrix and the corresponding points in two images
            _, R, t, mask_inliers = cv.recoverPose(E=essential_matrix, points1=self.tracked_keypoints_coor, points2=self.ref_keypoints_coor,
                                                   focal=self.focal, pp=self.principal_point)

            #################################
            #print('mask shape', mask_inliers_ess.shape)
            #print('mask shape', np.sum(mask_inliers_ess))
            if mask_inliers_ess is not None and np.sum(mask_inliers_ess) > 10 and mask_inliers_ess.shape[0] > 15:
                good_indexes = []
                bad_indexes = []
                for i in range(mask_inliers_ess.shape[0]):
                    if mask_inliers_ess[i] == 1:
                        good_indexes.append(i)
                    else:
                        bad_indexes.append(i)
                good_indexes = np.array(good_indexes)
                bad_indexes = np.array(bad_indexes)

                #print('>',mask_inliers_ess.shape[0])
                #print('>>',good_indexes)
                #if bad_indexes.shape[0] > 100:
                #    bad_indexes = bad_indexes[0:50]

                # separate good and bad corners or blob
                good_corners_des = self.tracked_keypoints_des[good_indexes]
                bad_corners_des = self.tracked_keypoints_des[bad_indexes]

                #print('>>>>', self.tracked_keypoints_des.shape)
                #print('>>>>', good_corners_des.shape)

                # labels
                good_labels = np.ones(shape=(good_corners_des.shape[0],1))
                bad_labels = np.zeros(shape=(bad_corners_des.shape[0],1))

                labels = np.vstack((good_labels, bad_labels))
                labels = np.reshape(labels, newshape=(-1))

                all_samples = np.vstack((good_corners_des, bad_corners_des))

                #print(labels.shape)
                #print(all_samples.shape)

                # train classifier
                if frame_id == 2:
                    self.classifier.fit(X=all_samples, y=labels)
                else:
                    self.classifier.partial_fit(X=all_samples, y=labels)


                #print('number of ones', good_corners.shape)
                #print('number of zeros', bad_corners.shape)
                #print('current keypoints shape', len(self.tracked_keypoints_key))
                #print('previous keypoints shape', len(self.ref_keypoints_key))
                #print('===============================')

            # absolute scale for updating translation matrix
            absolute_scale = self.get_absolute_scale(frame_id)

            # if absolute scale is significant update rotation and translation matrix
            # otherwise it's probably is a noise in calculations
            if absolute_scale > 0.1:

                # update translation
                self.first_to_i_th_frame_T = self.first_to_i_th_frame_T + absolute_scale * self.first_to_i_th_frame_R.dot(t)

                # update rotation matrix
                self.first_to_i_th_frame_R = R.dot(self.first_to_i_th_frame_R)

            # if number of key-points drops under a treshold, ignore tracked
            # key-points from last image and do feature detection again
            # if self.ref_keypoints_coor.shape[0] < 1600:

                # calculate key-points
                #self.tracked_keypoints_coor = self.feature_detector.detect(self.new_frame)
                # coordinates of key-points
                #self.tracked_keypoints_coor = np.array([x.pt for x in self.tracked_keypoints_coor], dtype=np.float32)

            # store tracked key-points as keypoint of input image
            self.ref_keypoints_coor = tracked_keypoints_coor_copy
            self.ref_keypoints_des = tracked_keypoints_des_copy
            self.ref_keypoints_key = tracked_keypoints_kp_copy

        # after updating coordinate by input image, input image becomes previous image
        self.prev_frame = self.new_frame
