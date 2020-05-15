import cv2 as cv
import numpy as np
from read_a_image_series import read_an_image_series

if __name__ == '__main__':

    # determine how much fast results are shown
    delay = 50

    # apply mixture of gaussian on different sequence of images
    for folder_name in ['birds', 'boats', 'bottle', 'cyclists', 'surf']:

        # read a sequence of image
        cap = cv.VideoCapture('C:\\Users\\Home\\Dropbox\\codes\\CV\\96131125_HW06\\JPEGS_min_v2\\JPEGS\\{}\\frame_%02d.jpg'.format(folder_name))

        # Define the codec and create VideoWriter object
        fourcc = cv.VideoWriter_fourcc(*'DIVX')
        out = cv.VideoWriter('.\\mog_output\\output-{}.avi'.format(folder_name), fourcc, 10.0, (int(cap.get(3)),int(cap.get(4))))

        # create MOG
        mog = cv.bgsegm.createBackgroundSubtractorMOG()
        while True:
            ret, frame = cap.read()
            # apply MOG
            fgmask = mog.apply(frame)
            if fgmask is None:
                break
            cv.imshow('frame', fgmask)
            fgmask_rbg = cv.cvtColor(fgmask,cv.COLOR_GRAY2BGR)
            out.write(fgmask_rbg)
            k = cv.waitKey(delay) & 0xff
            if k == 27:
                break
        cap.release()
        out.release()
        cv.destroyAllWindows()
