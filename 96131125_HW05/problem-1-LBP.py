from skimage import feature as LBP
import numpy as np
import cv2 as cv
from visual_bag_of_words import *
from read_dataset import read_dataset
from classifier import *
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


class local_binary_patterns_class:
    """
    LBP
    """
    def __init__(self, numPoints, radius):
        # the number of points
        self.numPoints = numPoints
        # radius
        self.radius = radius

    def compute(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = LBP.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        #hist = np.array(hist, dtype=np.float)
        #hist = np.reshape(hist, newshape=(1, len(hist)))
        # return the histogram of Local Binary Patterns
        return hist


#######################################################
#               LBG FEATURE
#######################################################
if __name__ == '__main__':

    # read dataset from file
    images, img_family, images_train, training_family, images_test, test_family = read_dataset()
    # len
    print("dataset: ", len(images), len(img_family), len(images_train), len(training_family), len(images_test),
          len(test_family))

    # LBP Descriptor
    radius = 3
    desc = local_binary_patterns_class(numPoints=radius*3, radius=radius)

    # feature vector for training images
    features_training = []
    for image in images_train:
        #feature = hog.compute(image)
        feature = desc.compute(image=image)
        #print(feature.shape, ' <')
        #print(feature)
        #feature = feature.T
        #print(feature.shape, ' <<')
        #feature = feature.tolist()[0]
        #print(len(feature), ' <<<')
        features_training.append(feature)
    features_training = np.array(features_training)
    print('feature training: ', features_training.shape)
    print('type of feature',type(features_training))

    # feature vector for test images
    features_test = []
    for image in images_test:
        #feature = hog.compute(image)
        feature = desc.compute(image=image)
        #feature = feature.T
        #feature = feature.tolist()[0]
        features_test.append(feature)
    features_test = np.array(features_test)
    print('feature test: ', features_test.shape)

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
                          title='LBP, Confusion matrix Train, without normalization')

    # predict test label
    pre_test = predict(test_x=features_test, clf=clf)
    print('Test Accuracy: ', accuracy_score(test_family, pre_test))
    #print('Test Confusion:\n', confusion_matrix(test_family, pre_test))
    print('Test F1-score: ', f1_score(test_family, pre_test, average='macro'))
    print('Test Precision-score: ', precision_score(test_family, pre_test, average='micro'))
    print('Test Recall-score: ', recall_score(test_family, pre_test, average='micro'))

    plt.figure()
    plot_confusion_matrix(confusion_matrix(test_family, pre_test), classes=[i for i in range(0, 49)],
                          title='LBP, Confusion matrix Test, without normalization')

    plt.show()