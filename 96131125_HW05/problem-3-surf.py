import cv2 as cv
import time


# sliding window
def sliding_window(image, stepSize, windowSize):
    """
    slide a window across the image
    :param image:
    :param stepSize:
    :param windowSize:
    :return:
    """
    # slide a window across the image
    for x in range(0, image.shape[0], stepSize):
        for y in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[x:x + windowSize[0], y:y + windowSize[1]])


if __name__ == '__main__':

    treshold_match_point = 1

    # read template
    template = cv.imread('Template.png', cv.IMREAD_GRAYSCALE)
    template_shape = template.shape

    # zoomed template
    enlargement = 4
    template_big = cv.resize(template, dsize=(template_shape[0]*enlargement, template_shape[1]*enlargement))

    # read image
    image = cv.imread('Scene.jpg', cv.IMREAD_GRAYSCALE)
    image_copy_rec = image.copy()
    image_copy_point = image.copy()

    # SURF
    surf = cv.xfeatures2d.SURF_create()

    # SURF key point for template
    temp_kp, temp_desc = surf.detectAndCompute(template_big, None)
    print('descriptor of template: ', temp_desc.shape)

    # check different sliding window for existence of template
    for x, y, sub_img in sliding_window(image=image, stepSize=template_shape[1]//6, windowSize=(template_shape[0]+5, template_shape[1]+5)):

        # if sliding window is out of image
        if sub_img.shape[0] != template_shape[0]+5 or sub_img.shape[1] != template_shape[1]+5:
            continue

        # surf descriptor for sliding window
        slide_kp, slide_desc = surf.detectAndCompute(sub_img, None)
        if slide_desc is None:
            continue

        print('descriptor of slide: ', slide_desc.shape)

        if slide_desc.shape[0] <= 2:
            continue

        # Brute Force Matcher
        bf = cv.BFMatcher()
        matches = bf.knnMatch(temp_desc, slide_desc, k=2)

        # Ratio Test for good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])

        if len(good_matches) >= treshold_match_point:
            cv.rectangle(image_copy_rec, (y, x), (y + template_shape[1], x + template_shape[0]), (0, 255, 255), 1)
            cv.imshow("Window_rec", image_copy_rec)
            cv.waitKey(1)
            time.sleep(0.025)
            cv.circle(img=image_copy_point, center=(y + template_shape[1]//2, x + template_shape[0]//2), radius=1,
                         color=(0, 255, 255), thickness=1)
            cv.imshow("Window_pint", image_copy_point)
            cv.waitKey(1)
            time.sleep(0.025)

    cv.imshow('Input Image', image)
    cv.imshow('Template', template)
    cv.imshow('Enlarge Template', template_big)
    cv.imwrite('.\\3\\1-surf.jpg', image_copy_rec)
    cv.imwrite('.\\3\\2-surf.jpg', image_copy_point)
    cv.waitKey(0)
