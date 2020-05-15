import cv2
import numpy as np

folders = ['Adirondack-perfect', 'Backpack-perfect',
           'Couch-perfect', 'Sword2-perfect']

# for different sequences, find depth image
for folder in folders:
    # path of images
    path = 'D:\\Vision\\HW07_dataset\\Question2\\{}\\{}\\'.format(folder, folder)
    # left image
    path_img0 = path + 'im0.png'
    # right image
    path_img1 = path + 'im1.png'

    img_l = cv2.imread(path_img0)
    img_r = cv2.imread(path_img1)

    # SGBM Parameters
    window_size = 3
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Convenience method to set up the matcher for computing the
    # right-view disparity map that is required in case of filtering with confidence.
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # filter parameters
    lambda_param = 80000
    sigma_param = 1.2
    visual_multiplier = 1.0

    # Convenience factory method that creates an instance of DisparityWLSFilter and sets up all
    # the relevant filter parameters automatically based on the matcher instance. Currently supports
    # only StereoBM and StereoSGBM.
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lambda_param)
    wls_filter.setSigmaColor(sigma_param)

    displ = left_matcher.compute(img_l, img_r)
    dispr = right_matcher.compute(img_r, img_l)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, img_l, None, dispr)

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    heat_img = cv2.applyColorMap(filteredImg, cv2.COLORMAP_JET)

    cv2.imshow('Disparity Map {}'.format(folder), cv2.resize(heat_img, (heat_img.shape[1]//4, heat_img.shape[0]//4)))

    # write to file
    cv2.imwrite(filename='.\\problem-1-2\\heatmap-{}.png'.format(folder), img=heat_img)

cv2.waitKey(0)
cv2.destroyAllWindows()