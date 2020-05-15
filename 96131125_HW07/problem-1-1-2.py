import numpy as np
import cv2 as cv


def featuers(img, type):
    """
    Give an image and type of feature then return coordinate of deatures
    :param img:
    :param type:
    :return:
    """
    if type == 'SIFT':
        # compute SIFT Features
        feature_detector = cv.xfeatures2d.SIFT_create()
        key_p = feature_detector.detect(img, None)
        key_p = [[kp.pt] for kp in key_p]
        coordinate_of_features = np.array(key_p, dtype=np.float32)

    elif type == 'SURF':
        # compute SURF Features
        feature_detector = cv.xfeatures2d.SURF_create()
        key_p = feature_detector.detect(img, None)
        key_p = [[kp.pt] for kp in key_p]
        coordinate_of_features = np.array(key_p, dtype=np.float32)

    elif type == 'ORB':
        # compute ORB Features
        feature_detector = cv.ORB_create(nfeatures=500, nlevels=16)
        key_p = feature_detector.detect(img, None)
        key_p = [[kp.pt] for kp in key_p]
        coordinate_of_features = np.array(key_p, dtype=np.float32)

    elif type == 'ShiTomasi':
        # parameters for Shi-Tomasi
        params_of_shiTomasi = {'maxCorners': 200, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7}
        coordinate_of_features = cv.goodFeaturesToTrack(img, mask=None, **params_of_shiTomasi)

    elif type == 'Grid':
        # grid
        gap = 22
        coordinate_of_features = []
        for i in range(gap, img.shape[0]-gap+1, gap):
            for j in range(gap, img.shape[1]-gap+1, gap):
                coordinate_of_features.append([[j, i]])
        coordinate_of_features = np.array(coordinate_of_features, dtype=np.float32)

    else:
        raise ValueError('Type of feature is not correct!')

    return coordinate_of_features


# folder names
folder_names = ['Army', 'Backyard', 'Basketball', 'Dumptruck',
                'Evergreen', 'Grove', 'Mequon', 'Schefflera',
                'Teddy', 'Urban', 'Wooden', 'Yosemite']

# Parameters of lucas kanade optical flow
parameters_lucas_kanade = dict(winSize=(15, 15),
                               maxLevel=3,
                               criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 11, 0.04))


# for different feature detector
for feature_method in ['Grid', 'ShiTomasi']:
    # for different image sequences
    for folder_name in folder_names:
        # read a sequence of images
        cap = cv.VideoCapture('D:\\Vision\\HW07_dataset\\Question1\\eval-data\\{}\\frame%02d.png'.format(folder_name))

        # First Frame
        _, old_frame = cap.read()

        # if image are not 3 slot, add 2 other slot
        if len(old_frame.shape) != 3 or old_frame.shape[2] != 3:
            old_frame = cv.cvtColor(old_frame, cv.COLOR_GRAY2RGB)

        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

        # coordinate of features in previous frame
        coord_features_prev = featuers(img=old_gray, type=feature_method)

        # Create random colors
        colors = np.random.randint(0, 255, (coord_features_prev.shape[0], 3))

        track_image = np.zeros_like(old_frame)

        while True:
            _,frame = cap.read()

            # if there is no more image in sequence, save result and exit
            if frame is None:
                cv.imwrite(filename='.\\problem-1-1-2\\{}-{}.png'.format(feature_method, folder_name), img=img)
                break

            # if image are not 3 slot, add 2 other slot
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)

            # convert to gray for feeding to Lucas-Kanade
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # calculate sparse optical flow
            coord_features_nex, status, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, coord_features_prev, None, **parameters_lucas_kanade)

            # Select points that a match are found for them
            point_nex = coord_features_nex[status == 1]
            point_prev = coord_features_prev[status == 1]

            # draw trajectory
            for i, (new, old) in enumerate(zip(point_nex, point_prev)):
                p1_x, p1_y = new.ravel()
                p2_x, p2_y = old.ravel()
                track_image = cv.line(track_image, (p1_x, p1_y), (p2_x, p2_y), colors[i].tolist(), 2)
                frame = cv.circle(frame, (p1_x, p1_y), 5, colors[i].tolist(), -1)
            img = cv.add(frame, track_image)

            cv.imshow('frame',img)
            k = cv.waitKey(300) & 0xff
            if k == 27:
                break

            # replace prevous keypoints by next keypoints
            old_gray = frame_gray.copy()
            coord_features_prev = point_nex.reshape(-1, 1, 2)

        cv.destroyAllWindows()
        cap.release()