import numpy as np
import cv2 as cv


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


def key_points_tracking(prevImg, nextImg, prevPts):
    """
    Finds key points of frame i in frame i+1
    :param prevImg:
    :param nextImg:
    :param prevPts:
    :return:
    """

    # Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids.
    keupoints_next, st, err = cv.calcOpticalFlowPyrLK(prevImg=prevImg, nextImg=nextImg, prevPts=prevPts, nextPts=None, winSize=(21, 21),
                                            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

    # Eliminates key-points that a corresponding key-point wasn't found for them.
    st = st.reshape(st.shape[0])
    keypoints_pre = prevPts[st == 1]
    keupoints_next = keupoints_next[st == 1]

    return keypoints_pre, keupoints_next


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
        self.ref_keypoints = None
        self.tracked_keypoints = None

        # Focal length in pixels.
        self.focal = cam['fx']

        # Optical center
        self.principal_point = (cam['cx'], cam['cy'])

        # ground truth current position of camera
        self.ground_truth_current_x, self.ground_truth_current_y, self.ground_truth_current_z = 0, 0, 0

        # feature detector
        self.feature_detector = cv.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

        # read ground truth poses (trajectory)
        with open(ground_truth) as f:
            self.ground_truth = f.readlines()

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
            self.ref_keypoints = self.feature_detector.detect(self.new_frame)
            # coordinates of key-points
            self.ref_keypoints = np.array([x.pt for x in self.ref_keypoints], dtype=np.float32)

        # if image is second frame of sequence
        elif frame_id == 1:
            # for second image in sequence, track key-points of first image, then calculate essential matrix
            # then obtain Rotation and Translation Matrix

            # find matches of feattures of previous frame in current frame
            self.ref_keypoints, self.tracked_keypoints = key_points_tracking(prevImg=self.prev_frame, nextImg=self.new_frame,
                                                                             prevPts=self.ref_keypoints)

            # calculate an essential matrix from the corresponding points in two images
            essential_matrix, mask_inliers = cv.findEssentialMat(points1=self.tracked_keypoints, points2=self.ref_keypoints,
                                                                 focal=self.focal, pp=self.principal_point,
                                                                 method=cv.RANSAC, prob=0.999, threshold=1.0)

            # recover relative camera rotation and translation from an estimated essential matrix
            # and the corresponding points in two images
            _, self.first_to_i_th_frame_R, self.first_to_i_th_frame_T, mask_inliers = \
                cv.recoverPose(E=essential_matrix, points1=self.tracked_keypoints, points2=self.ref_keypoints,
                               focal=self.focal, pp=self.principal_point)

            # store tracked key-points as keypoint of input image
            self.ref_keypoints = self.tracked_keypoints

        # if image is not first or second frame of sequence
        elif frame_id > 1:
            # for i'th image in sequence, track key-points of (i-1)'th image, then calculate essential matrix
            # then obtain Rotation and Translation Matrix

            self.ref_keypoints, self.tracked_keypoints = key_points_tracking(prevImg=self.prev_frame, nextImg=self.new_frame,
                                                                             prevPts=self.ref_keypoints)

            # calculate an essential matrix from the corresponding points in two images
            essential_matrix, mask_inliers = cv.findEssentialMat(points1=self.tracked_keypoints, points2=self.ref_keypoints,
                                                                 focal=self.focal, pp=self.principal_point,
                                                                 method=cv.RANSAC, prob=0.999, threshold=1.0)

            # recover relative camera rotation and translation from an estimated
            # essential matrix and the corresponding points in two images
            _, R, t, mask_inliers = cv.recoverPose(E=essential_matrix, points1=self.tracked_keypoints, points2=self.ref_keypoints,
                                                   focal=self.focal, pp=self.principal_point)
            #print(mask_inliers.shape)
            #print(self.tracked_keypoints.shape)
            #print(self.ref_keypoints.shape)
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
            if self.ref_keypoints.shape[0] < 1600:

                # calculate key-points
                self.tracked_keypoints = self.feature_detector.detect(self.new_frame)
                # coordinates of key-points
                self.tracked_keypoints = np.array([x.pt for x in self.tracked_keypoints], dtype=np.float32)

            # store tracked key-points as keypoint of input image
            self.ref_keypoints = self.tracked_keypoints

        # after updating coordinate by input image, input image becomes previous image
        self.prev_frame = self.new_frame
