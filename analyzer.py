import cv2
import numpy as np
from time import sleep


class Tracker:
    def __init__(self, fileInput):
        self.videoPath = fileInput
        self.tracker = None
        self.heigth = 0
        self.width = 0
        self.size = 3

    def getBlobParams(self):
        params = cv2.SimpleBlobDetector_Params()
        # params.filterByCircularity = True
        # params.minCircularity = 0.8

        # params.filterByConvexity = True
        # params.minConvexity = 0.5

        # params.filterByInertia = False
        # params.minInertiaRatio = 0.01

        params.filterByArea = True
        params.minArea = 9999
        # params.maxArea = 999

        # params.filterByColor = True
        # params.blobColor = 200

        return params

    # def fromBlobToBbox(self, blob):
    #     x, y, size = blob  # Separate the values on the blob
    #     x = (x - size/self.size)
    #     y = int(y - size/self.size)
    #     return x, y, size, size  # Since is a circle

    def resize(self, frame):
        new_heigth = int(self.heigth/self.size)
        new_width = int(self.width/self.size)
        return cv2.resize(frame, (new_width, new_heigth))

    def getTrackerByName(self, tracker_name='KCF'):
        tracker_name = tracker_name.upper()
        trackers = {
            'BOOSTING': cv2.TrackerBoosting_create(),
            'MIL': cv2.TrackerMIL_create(),
            'KCF': cv2.TrackerKCF_create(),
            'TLD': cv2.TrackerTLD_create(),
            'MEDIANFLOW': cv2.TrackerMedianFlow_create(),
            'GOTURN': cv2.TrackerGOTURN_create(),
            'MOSSE': cv2.TrackerMOSSE_create(),
            'CSRT': cv2.TrackerCSRT_create()
        }
        try:
            self.tracker = trackers[tracker_name]
        except KeyError:
            raise KeyError('Wrong Tracker name')
        except Exception:
            raise

    def tracking(self):
        cap = cv2.VideoCapture(self.videoPath)

        # Resize image since is too big
        self.heigth = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # _, first_frame = cap.read()
        # first_frame = self.resize(first_frame.copy())
        # first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        detector = cv2.SimpleBlobDetector_create(self.getBlobParams())
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            #  1 - Aplying blur and thresold.
            frame = self.resize(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

            #  2 - Aplying a mask per color.
            lower_red = np.array([0, 130, 75])
            upper_red = np.array([255, 255, 255])
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_red, upper_red)
            # ret, thresh = cv2.threshold(frame, 127, 255, 0)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            red = cv2.bitwise_and(frame, frame, mask=mask)

            # cnt = cv2.drawContours(frame.copy(), contours, -1, (255, 0, 0))
            # keypoints = detector.detect(hsv)
            # im_with_keypoints = cv2.drawKeypoints(
            #     frame, keypoints,
            #     np.array([]), (255, 0, 0),
            #     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            # )

            # cv2.imshow("frame", frame)
            # cv2.imshow("gray", gray)
            # cv2.imshow("blurred", blurred)
            cv2.imshow("thresh", thresh)
            # cv2.imshow("Keypoints", im_with_keypoints)
            # cv2.imshow("Contours", cnt)
            # cv2.imshow("Mask", mask)
            cv2.imshow("red", red)


            # success, bbox = self.tracker.update(frame)
            # if success:
            #     point1 = (int(bbox[0]), int(bbox[1]))
            #     point2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
            #     cv2.rectangle(frame, point1, point2, (255, 0, 0), 2, 1)
            # else:
            #     cv2.putText(
            #         frame, "Failed.", (100, 80),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.75, (0, 0, 255), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Q Key pressed
                break
            elif key == ord('s'):  # S Key pressed
                time.sleep(5)
            elif key == ord('t'):  # T Key pressed
                input('Any key to continue')

            # cv2.imshow("Ball Rolling", frame)

        cap.release()  # Dont forget to release your video!

if __name__ == "__main__":
    videoPath = 'assets/vid1.mp4'

    tracker = Tracker(videoPath)
    tracker.tracking()

    cv2.destroyAllWindows()  # Close all the opened windows
