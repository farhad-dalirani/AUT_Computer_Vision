import cv2 as cv

def read_dataset(path_of_folder='D:\Vision\HW04_dataset\caltech-buildings'):
    """
    Read caltech-building dataset
    :param path:
    :return:
    """
    list_path= 'D:\\Vision\\HW04_dataset\\caltech-buildings\\lists\\caltech-list.txt'
    family_path= 'D:\\Vision\\HW04_dataset\\caltech-buildings\\lists\\caltech-family.txt'
    image_path='D:\\Vision\\HW04_dataset\\caltech-buildings\\images\\'

    # read name of images from file
    with open(list_path) as f:
        content = f.readlines()
    img_names = [x.strip() for x in content]

    # read family of images from file
    with open(family_path) as f:
        content = f.readlines()
    img_family = [x.strip() for x in content]
    img_family = [int(family)-1 for family in img_family]

    #print(img_names, len(img_names))
    #print(img_family)

    images = []

    # read images
    for img_name in img_names:
        img = cv.imread(filename=image_path+img_name[7::], flags=cv.IMREAD_GRAYSCALE)
        img = cv.resize(img,(int(800),int(600)))
        images.append(img)

    images_train = []
    training_family = []
    images_test = []
    test_family = []

    # split test and train
    for i in range(len(images)):
        if i % 5 == 0 or i % 5 == 1 or i % 5 == 2:
            images_train.append(images[i])
            training_family.append(img_family[i])
        else:
            images_test.append(images[i])
            test_family.append(img_family[i])

    return images, img_family, images_train, training_family, images_test, test_family


if __name__ == '__main__':
    import cv2 as cv

    # read dataset from file
    images, img_family, images_train, training_family, images_test, test_family = read_dataset()

    # len
    print(len(images), len(img_family), len(images_train), len(training_family), len(images_test), len(test_family))

    print(img_family)
    print(training_family)
    print(test_family)

    #cv.imshow('all 6 {}'.format(img_family[5]), images[5])
    cv.namedWindow('train 3 {}'.format(training_family[9]), cv.WINDOW_NORMAL)
    cv.imshow('train 3 {}'.format(training_family[9]), images_train[9])
    cv.namedWindow('train 4 {}'.format(training_family[10]), cv.WINDOW_NORMAL)
    cv.imshow('train 4 {}'.format(training_family[10]), images_train[10])
    cv.namedWindow('train 5 {}'.format(training_family[11]), cv.WINDOW_NORMAL)
    cv.imshow('train 5 {}'.format(training_family[11]), images_train[11])
    #cv.imshow('test 1 {}'.format(test_family[4]), images_test[4])
    #cv.imshow('test 2 {}'.format(test_family[5]), images_test[5])

    cv.waitKey(0)






