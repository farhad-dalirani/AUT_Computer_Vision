from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def train_classifer(training_x, training_y):
    """
    Train svm classifier
    :param training_x:
    :param training_y:
    :return:
    """

    #print('shape of x & y: ', training_x.shape, len(training_y))
    clf = svm.SVC()
    clf.fit(training_x, training_y)

    return clf


def predict(test_x, clf):
    """
    return Label of an image
    :param test_x: feature vector of image in bag of visual word
    :param clf:
    :return:
    """
    pre_test = clf.predict(test_x)
    return pre_test


if __name__ == '__main__':
    import cv2 as cv
    from visual_bag_of_words import *
    from read_dataset import read_dataset

    # read dataset from file
    images, img_family, images_train, training_family, images_test, test_family = read_dataset()
    # len
    print("dataset: ", len(images), len(img_family), len(images_train), len(training_family), len(images_test), len(test_family))

    # bag of words
    bowDiction = visual_bag_of_words(images_train=images_train, feature_detector_name='sift', dictionary_size=100)
    #print(bowDiction.cluster_centers_)

    # feature vector for training images
    features_training = []
    for image in images_train:
        feature = sifts_to_features(image=image, kmeans_bow=bowDiction, dictionary_size=100, feature_detector_name='sift')
        features_training.append(feature)
    #features_training = np.array(features_training)

    # feature vector for test images
    features_test = []
    for image in images_test:
        feature = sifts_to_features(image=image, kmeans_bow=bowDiction, dictionary_size=100,
                                    feature_detector_name='sift')
        features_test.append(feature)
    #features_test = np.array(features_test)

    #print(features_training.shape, ' <')
    #print(features_test.shape, ' <')

    # train classifier
    print('taining_X', len(features_training))
    print(features_training)
    print('training_y ', len(training_family))
    print(training_family)
    clf = train_classifer(training_x=features_training, training_y=training_family)

    # predict train label
    pre_train = predict(test_x=features_training, clf=clf)
    print('Train Accuracy: ', accuracy_score(training_family, pre_train))
    print('Train Confusion:\n', confusion_matrix(training_family, pre_train))
    print('Train F1-score: ', f1_score(training_family, pre_train, average='macro'))

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(confusion_matrix(training_family, pre_train), classes=[i for i in range(0,49)],
                          title='Confusion matrix Train, without normalization')

    # predict test label
    pre_test = predict(test_x=features_test, clf=clf)
    print('Test Accuracy: ', accuracy_score(test_family, pre_test))
    print('Test Confusion:\n', confusion_matrix(test_family, pre_test))
    print('Test F1-score: ', f1_score(test_family, pre_test, average='macro'))
    plt.figure()
    plot_confusion_matrix(confusion_matrix(test_family, pre_test), classes=[i for i in range(0, 49)],
                          title='Confusion matrix Test, without normalization')

    plt.show()