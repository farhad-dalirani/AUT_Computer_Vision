from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import itertools
import numpy as np
import matplotlib.pyplot as plt
from classifier import *
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

if __name__ == '__main__':
    #######################################################
    #               SIFT FEATURE
    #######################################################

    import cv2 as cv
    from visual_bag_of_words import *
    from read_dataset import read_dataset

    # read dataset from file
    images, img_family, images_train, training_family, images_test, test_family = read_dataset()
    # len
    print("Dataset: ", len(images), len(img_family), len(images_train), len(training_family), len(images_test),
          len(test_family))

    # bag of words
    bowDiction = visual_bag_of_words(images_train=images_train, feature_detector_name='sift', dictionary_size=100)
    # print(bowDiction.cluster_centers_)

    # feature vector for training images
    features_training = []
    for image in images_train:
        feature = sifts_to_features(image=image, kmeans_bow=bowDiction, dictionary_size=100,
                                    feature_detector_name='sift')
        features_training.append(feature)
    # features_training = np.array(features_training)

    # feature vector for test images
    features_test = []
    for image in images_test:
        feature = sifts_to_features(image=image, kmeans_bow=bowDiction, dictionary_size=100,
                                    feature_detector_name='sift')
        features_test.append(feature)
    # features_test = np.array(features_test)

    # print(features_training.shape, ' <')
    # print(features_test.shape, ' <')

    # train classifier
    print('taining_X', len(features_training))
    print(features_training)
    print('training_y ', len(training_family))
    print(training_family)
    clf = train_classifer(training_x=features_training, training_y=training_family)

    # predict train label
    pre_train = predict(test_x=features_training, clf=clf)
    print('Train Accuracy: ', accuracy_score(training_family, pre_train))
    #print('Train Confusion:\n', confusion_matrix(training_family, pre_train))
    print('Train F1-score: ', f1_score(training_family, pre_train, average='macro'))
    print('Train Precision-score: ', precision_score(training_family, pre_train, average='micro'))
    print('Train Recall-score: ', recall_score(training_family, pre_train, average='micro'))

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix(training_family, pre_train), classes=[i for i in range(0, 49)],
                          title='SIFT,Confusion matrix Train, without normalization')

    # predict test label
    pre_test = predict(test_x=features_test, clf=clf)
    print('Test Accuracy: ', accuracy_score(test_family, pre_test))
    #print('Test Confusion:\n', confusion_matrix(test_family, pre_test))
    print('Test F1-score: ', f1_score(test_family, pre_test, average='macro'))
    print('Test Precision-score: ', precision_score(test_family, pre_test, average='micro'))
    print('Test Recall-score: ', recall_score(test_family, pre_test, average='micro'))

    plt.figure()
    plot_confusion_matrix(confusion_matrix(test_family, pre_test), classes=[i for i in range(0, 49)],
                          title='SIFT,Confusion matrix Test, without normalization')

    #plt.show()

    print('===========================')
    #######################################################
    #               HoG FEATURE
    #######################################################

    import cv2 as cv
    from visual_bag_of_words import *
    from read_dataset import read_dataset

    # read dataset from file
    images, img_family, images_train, training_family, images_test, test_family = read_dataset()
    # len
    print("dataset: ", len(images), len(img_family), len(images_train), len(training_family), len(images_test),
          len(test_family))

    # HOG
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    #hog = cv.HOGDescriptor()

    # feature vector for training images
    features_training = []
    for image in images_train:
        #feature = hog.compute(image)
        feature, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualise=True, feature_vector=True)
        #print(feature.shape, ' <')
        #feature = feature.T
        #print(feature.shape, ' <<')
        #feature = feature.tolist()[0]
        #print(len(feature), ' <<<')
        features_training.append(feature)
    features_training = np.array(features_training)
    print('feature training: ', features_training.shape)

    print('type of feature',type(features_training))
    # pca
    pca = PCA(n_components=200)
    pca.fit(features_training)
    print('PCA is done.')
    features_training = pca.transform(features_training)
    print('feature training after pca: ', features_training.shape)
    print('type of feature',type(features_training))

    # feature vector for test images
    features_test = []
    for image in images_test:
        #feature = hog.compute(image)
        feature, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                                 cells_per_block=(2, 2), visualise=True, feature_vector=True)
        #feature = feature.T
        #feature = feature.tolist()[0]
        features_test.append(feature)
    features_test = np.array(features_test)
    print('feature test: ', features_test.shape)

    features_test = pca.transform(features_test)
    print('feature test after pca: ', features_test.shape)

    features_training = features_training.tolist()
    features_test = features_test.tolist()

    # train classifier
    print('training_X', len(features_training),len(features_training[0]))
    #print(features_training)
    print('training_y ', len(training_family))
    #print(training_family)
    clf = train_classifer(training_x=features_training, training_y=training_family)

    # predict train label
    pre_train = predict(test_x=features_training, clf=clf)
    print('Train Accuracy: ', accuracy_score(training_family, pre_train))
    #print('Train Confusion:\n', confusion_matrix(training_family, pre_train))
    print('Train F1-score: ', f1_score(training_family, pre_train, average='macro'))
    print('Train Precision-score: ', precision_score(training_family, pre_train, average='micro'))
    print('Train Recall-score: ', recall_score(training_family, pre_train, average='micro'))

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix(training_family, pre_train), classes=[i for i in range(0, 49)],
                          title='HoG, Confusion matrix Train, without normalization')

    # predict test label
    pre_test = predict(test_x=features_test, clf=clf)
    print('Test Accuracy: ', accuracy_score(test_family, pre_test))
    #print('Test Confusion:\n', confusion_matrix(test_family, pre_test))
    print('Test F1-score: ', f1_score(test_family, pre_test, average='macro'))
    print('Test Precision-score: ', precision_score(test_family, pre_test, average='micro'))
    print('Test Recall-score: ', recall_score(test_family, pre_test, average='micro'))

    plt.figure()
    plot_confusion_matrix(confusion_matrix(test_family, pre_test), classes=[i for i in range(0, 49)],
                          title='HoG, Confusion matrix Test, without normalization')

    plt.show()