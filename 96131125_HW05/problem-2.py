import cv2 as cv
import numpy as np

def match_and_draw_key_points(img1=None, name1='', img2=None, name2='', feature_detector_name='sift',
                              feature_descriptor_name='sift', top_match=50, order=0):
    """

    :param img1:
    :param name1:
    :param img2:
    :param name2:
    :param detector:
    :param top_match:
    :return:
    """
    descriptor_img1 = None
    descriptor_img2 = None
    keypoints_img1 = None
    keypoints_img2 = None

    if feature_detector_name is None or feature_descriptor_name is None:
        raise ValueError('Feature Detector or Descriptor can\t be none!')

    # feature detector
    feature_detector = None
    if feature_detector_name == 'sift':
        feature_detector = cv.xfeatures2d.SIFT_create()
    elif feature_detector_name == 'surf':
        feature_detector = cv.xfeatures2d.SURF_create()
    if feature_detector is None:
        raise ValueError('Feature Detector can\'t be none')

    # feature descriptor
    feature_descriptor = None
    if feature_descriptor_name == 'sift':
        feature_descriptor = cv.xfeatures2d.SIFT_create()
    elif feature_descriptor_name == 'surf':
        feature_descriptor = cv.xfeatures2d.SURF_create()
    elif feature_descriptor_name == 'freak':
        feature_descriptor = cv.xfeatures2d.FREAK_create()

    if feature_descriptor is None:
        raise ValueError('Feature Descriptor can\'t be none')

    # Detector
    if feature_descriptor_name == 'surf':
        # keypoints and descriptors
        keypoints_img1, descriptor_img1 = feature_descriptor.detectAndCompute(img1, None)
        keypoints_img2, descriptor_img2 = feature_descriptor.detectAndCompute(img2, None)

    if feature_descriptor_name == 'freak':
        kp1 = feature_detector.detect(img1, None)
        keypoints_img1, descriptor_img1 = feature_descriptor.compute(img1, kp1)

        kp2 = feature_detector.detect(img2, None)
        keypoints_img2, descriptor_img2 = feature_descriptor.compute(img2, kp2)

    #print(descriptor_img1.shape)
    #print(descriptor_img2.shape)

    # Brute Force Matcher
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptor_img1, descriptor_img2, k=2)

    # Ratio Test for good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1, keypoints_img1, img2, keypoints_img2, good_matches, None, flags=2)

    plt.figure()
    plt.imshow(img3)
    plt.title('{}, i\'th building & j\'th image {}, p\'th building and q\'th image {}.({})'.format(feature_detector_name+'_'+feature_descriptor_name, name1, name2, len(good_matches)))
    cv.imwrite('.\\2\\{}-{}, i\'th building & j\'th image {}, p\'th building and q\'th image {}.({}).jpg'.format(order, feature_detector_name+'_'+feature_descriptor_name, name1, name2, len(good_matches)), img3)


if __name__ == '__main__':
    from read_dataset import read_dataset
    import matplotlib.pyplot as plt

    # read dataset from file
    images, img_family, images_train, training_family, images_test, test_family = read_dataset()

    # len
    print("dataset: ", len(images), len(img_family), len(images_train),
          len(training_family), len(images_test), len(test_family))

    order = 0

    # surf
    for i in [0, 5, 10]:
        for j in range(1, 5):
            match_and_draw_key_points(img1=images[i], name1='{}-{}'.format(i,0), img2=images[i+j],
                                      name2='{}-{}'.format(i,j), feature_detector_name='surf',
                              feature_descriptor_name='surf', order=order)
            order = order + 1

    # ith building with other
    match_and_draw_key_points(img1=images[0], name1='{}-{}'.format(0, 0), img2=images[6],
                                  name2='{}-{}'.format(1, 1), feature_detector_name='surf',
                              feature_descriptor_name='surf', order = order)
    order = order + 1

    match_and_draw_key_points(img1=images[0], name1='{}-{}'.format(0, 0), img2=images[14],
                              name2='{}-{}'.format(2, 5), feature_detector_name='surf',
                              feature_descriptor_name='surf', order = order)
    order = order + 1

    match_and_draw_key_points(img1=images[5], name1='{}-{}'.format(1, 0), img2=images[2],
                              name2='{}-{}'.format(0, 3), feature_detector_name='surf',
                              feature_descriptor_name='surf',order = order)
    order = order + 1

    match_and_draw_key_points(img1=images[5], name1='{}-{}'.format(1, 0), img2=images[12],
                              name2='{}-{}'.format(3, 3), feature_detector_name='surf',
                              feature_descriptor_name='surf',order = order)
    order = order + 1

    match_and_draw_key_points(img1=images[10], name1='{}-{}'.format(2, 0), img2=images[2],
                              name2='{}-{}'.format(0, 3), feature_detector_name='surf',
                              feature_descriptor_name='surf',order = order)
    order = order + 1

    match_and_draw_key_points(img1=images[10], name1='{}-{}'.format(2, 0), img2=images[8],
                              name2='{}-{}'.format(1, 4), feature_detector_name='surf',
                              feature_descriptor_name='surf',order = order)
    order = order + 1

    # freak
    for i in [0, 5, 10]:
        for j in range(1, 5):
            match_and_draw_key_points(img1=images[i], name1='{}-{}'.format(i,0), img2=images[i+j],
                                      name2='{}-{}'.format(i,j), feature_detector_name='surf',
                              feature_descriptor_name='freak',order = order)
            order = order + 1

    # ith building with other
    match_and_draw_key_points(img1=images[0], name1='{}-{}'.format(0, 0), img2=images[6],
                              name2='{}-{}'.format(1, 1),  feature_detector_name='surf',
                              feature_descriptor_name='freak',order = order)
    order = order + 1
    match_and_draw_key_points(img1=images[0], name1='{}-{}'.format(0, 0), img2=images[14],
                              name2='{}-{}'.format(2, 5),  feature_detector_name='surf',
                              feature_descriptor_name='freak',order = order)
    order = order + 1
    match_and_draw_key_points(img1=images[5], name1='{}-{}'.format(1, 0), img2=images[2],
                              name2='{}-{}'.format(0, 3),  feature_detector_name='surf',
                              feature_descriptor_name='freak',order = order)
    order = order + 1
    match_and_draw_key_points(img1=images[5], name1='{}-{}'.format(1, 0), img2=images[12],
                              name2='{}-{}'.format(3, 3),  feature_detector_name='surf',
                              feature_descriptor_name='freak',order = order)
    order = order + 1
    match_and_draw_key_points(img1=images[10], name1='{}-{}'.format(2, 0), img2=images[2],
                              name2='{}-{}'.format(0, 3),  feature_detector_name='surf',
                              feature_descriptor_name='freak',order = order)
    order = order + 1
    match_and_draw_key_points(img1=images[10], name1='{}-{}'.format(2, 0), img2=images[8],
                              name2='{}-{}'.format(1, 4),  feature_detector_name='surf',
                              feature_descriptor_name='freak',order = order)

    plt.show()

