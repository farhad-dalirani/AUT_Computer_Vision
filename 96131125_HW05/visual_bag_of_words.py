import cv2 as cv
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def visual_bag_of_words(images_train, feature_detector_name='sift', feature_descriptor_name='sift', dictionary_size=1000):
    """
    Visual bag of words
    :param images_train:
    :param feature_detector_name:
    :param feature_descriptor_name:
    :param dictionary_size: number of cluster in kmeans
    :return:
    """
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


    all_dsc = None
    # find feature for each training image
    for idx, image in enumerate(images_train):
        if feature_descriptor_name == 'sift' or feature_detector_name == 'surf':
            kp, dsc = feature_detector.detectAndCompute(image, None)

            if idx == 0:
                all_dsc = np.copy(dsc)
            else:
                all_dsc = np.vstack((all_dsc,dsc))

        if feature_descriptor_name == 'freak':
            kp = feature_detector.detect(image, None)
            kp, dsc = feature_descriptor.compute(image, kp)

            if idx == 0:
                all_dsc = np.copy(dsc)
            else:
                all_dsc = np.vstack((all_dsc, dsc))

    # batch kmeans
    kmeans_bow = MiniBatchKMeans(n_clusters=dictionary_size, random_state = 0).fit(all_dsc)

    # return batch kmeans
    return kmeans_bow


def keypoints_to_features(image, kmeans_bow, dictionary_size=1000, feature_detector_name='sift', feature_descriptor_name = 'sift'):
    """
    this function gets an image then extracts sift features of the image and return its equivalent in
    bag of visual words normalized feature vector
    :param image:
    :param dictionary_size: number of cluster in kmeans
    :param feature_detector_name:
    :return:
    """
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

    # extract features from image
    if feature_descriptor_name == 'sift' or feature_descriptor_name == 'surf':
        kp, dsc = feature_detector.detectAndCompute(image, None)
    if feature_descriptor_name == 'freak':
        kp = feature_detector.detect(image, None)
        kp, dsc = feature_descriptor.compute(image, kp)

    # find equivalent of each feature in bag of words
    equivalent_in_bow = kmeans_bow.predict(dsc)

    # count number of occurrence of each feature
    a = equivalent_in_bow.tolist()
    a_sum = max(1, sum(a))
    vector_feature = [a.count(i)/a_sum for i in range(0, dictionary_size)]

    return vector_feature


if __name__ == '__main__':
    import cv2 as cv
    from read_dataset import read_dataset

    # read dataset from file
    images, img_family, images_train, training_family, images_test, test_family = read_dataset()
    # len
    print("dataset: ", len(images), len(img_family), len(images_train), len(training_family), len(images_test), len(test_family))

    ###################################
    # bag of words
    bowDiction = visual_bag_of_words(images_train=images_train, feature_detector_name='surf', feature_descriptor_name='freak', dictionary_size=100)
    print(bowDiction.cluster_centers_)

    # feature vector for an image
    feature = keypoints_to_features(image=images_train[2], kmeans_bow=bowDiction, dictionary_size=100,
                                    feature_detector_name='surf', feature_descriptor_name='freak')
    print(feature)

    ###################################
    # bag of words
    bowDiction = visual_bag_of_words(images_train=images_train, feature_detector_name='surf', dictionary_size=100)
    print(bowDiction.cluster_centers_)

    # feature vector for an image
    feature = keypoints_to_features(image=images_train[2], kmeans_bow=bowDiction, dictionary_size=100,
                                    feature_detector_name='surf')
    print(feature)

    ###################################
    # bag of words
    bowDiction = visual_bag_of_words(images_train=images_train, feature_detector_name='sift', dictionary_size=100)
    print(bowDiction.cluster_centers_)

    # feature vector for an image
    feature = keypoints_to_features(image=images_train[2], kmeans_bow=bowDiction, dictionary_size=100,
                                    feature_detector_name='sift')
    print(feature)