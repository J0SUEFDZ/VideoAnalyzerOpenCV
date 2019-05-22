import cv2
import numpy as np
from time import sleep


class Tracker:
    def __init__(self, fileInput):
        self.videoPath = fileInput
        self.heigth = 0
        self.width = 0
        self.size = 3

    def getBlobParams(self):
        params = cv2.SimpleBlobDetector_Params()
        # params.filterByCircularity = True
        # params.minCircularity = 0.6

        # params.filterByConvexity = True
        # params.minConvexity = 0.2

        # params.filterByArea = True
        # params.minArea = 100
        # params.maxArea = 99

        params.filterByColor = True
        params.blobColor = 10

        return params

    def resize(self, frame):
        new_heigth = int(self.heigth/self.size)
        new_width = int(self.width/self.size)
        return cv2.resize(frame, (new_width, new_heigth))

    def tracking(self):
        cap = cv2.VideoCapture(self.videoPath)

        # Resize image since is too big
        self.heigth = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        detector = cv2.SimpleBlobDetector_create(self.getBlobParams())
        fgbg = cv2.createBackgroundSubtractorMOG2()
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame = self.resize(frame)
            #  1 - Aplying blur and thresold.
            # frame = self.resize(frame)
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

            #  2 - Aplying a mask per color.
            # lower_red = np.array([0, 130, 75])
            # upper_red = np.array([255, 255, 255])
            # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # mask = cv2.inRange(hsv, lower_red, upper_red)
            # contours, _ = cv2.findContours(
            #     mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            # )
            # red = cv2.bitwise_and(frame, frame, mask=mask)

            # cnt = cv2.drawContours(frame.copy(), contours, -1, (255, 0, 0))
            # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
            # keypoints = detector.detect(gray)
            # im_with_keypoints = cv2.drawKeypoints(
            #     frame, keypoints,
            #     np.array([]), (255, 0, 0),
            #     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            # )
            # if keypoints:
            #     sleep(1)
            # cv2.imshow('Recording', gray)
            # cv2.imshow('Recording', im_with_keypoints)

            fgmask = fgbg.apply(frame)
            fgmask[fgmask == 127] = 0
            cv2.imshow('Background Sustractor', fgmask)
            keypoints = detector.detect(fgmask)
            im_with_keypoints = cv2.drawKeypoints(
                frame, keypoints,
                np.array([]), (255, 0, 0),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            if keypoints:
                sleep(1)
            cv2.imshow('Recording', im_with_keypoints)

            # Show data
            # cv2.imshow("thresh", thresh)
            # cv2.imshow("red", red)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Q Key pressed
                break
            elif key == ord('s'):  # S Key pressed
                sleep(5)
            elif key == ord('t'):  # T Key pressed
                input('Any key to continue')

        cap.release()  # Dont forget to release your video!


if __name__ == "__main__":
    videoPath = 'assets/vid3.mp4'

    tracker = Tracker(videoPath)
    tracker.tracking()

    cv2.destroyAllWindows()  # Close all the opened windows
