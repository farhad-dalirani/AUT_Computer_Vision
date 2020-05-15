import matplotlib.pyplot as plt
from skimage import data, exposure
from skimage.feature import hog
from read_dataset import read_dataset
import numpy as np
# read dataset from file


def hog_for_point(image, point):
    """
    Hog of a point
    :param image:
    :param point:
    :return:
    """
    img = np.pad(image, ((16, 16), (16, 16)), 'edge')
    point_1 = [0,0]
    point_1[0] = min(point[0], 600) + 16
    point_1[1] = min(point[1], 800) + 16
    image_area = img[point_1[0] - 8:point_1[0] + 8, point_1[1] - 8:point_1[1] + 8]

    fd, hog_image = hog(image_area, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True, feature_vector=False)

    #print(fd[0,0,:,:,:])
    #print(fd.shape)
    #print(len(fd))
    #print(fd[0].shape)
    #print(image[point[0]-8:point[0]+8, point[1]-8:point[1]+8].shape)

    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    #ax1.axis('off')
    #ax1.imshow(image[point[0]-8:point[0]+8, point[1]-8:point[1]+8], cmap=plt.cm.gray)
    #ax1.set_title('Input image')

    # Rescale histogram for better display
    #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    #ax2.axis('off')
    #ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    #ax2.set_title('Histogram of Oriented Gradients')
    #plt.show()
    #print(len(fd.shape),'<<')
    #x = fd.shape
    if fd.shape[0] != 0:
        #print(fd[0, 0, 0, 0, :], ' <<<<')
        return fd[0, 0, 0, 0, :]
    else:
        #print(' ******')
        return [0,0,0,0,0,0,0,0]


def hog_for_keypoint_of_image(image, keypoints):
    """
    for each keypoint find hog descriptor
    :param image:
    :param keypoints:
    :return:
    """
    all_dst = []
    for keypoint in keypoints:
        dst = hog_for_point(image=image, point=(int(round(keypoint.pt[0])), int(round(keypoint.pt[1]))))
        if dst is not None:
            all_dst.append(dst)
    all_dst = np.array(all_dst, dtype=np.float32)

    return all_dst


if __name__ == '__main__':
    from read_dataset import read_dataset

    # read dataset from file
    images, img_family, images_train, training_family, images_test, test_family = read_dataset()

    fd = hog_for_point(images[0], [10, 10])

    print(fd)
