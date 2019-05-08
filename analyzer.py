import cv2
import numpy as np


class Tracker:
    def __init__(self, fileInput):
        self.videoPath = fileInput
        self.tracker = None
        self.heigth = 0
        self.width = 0

    def getBlobParams(self):
        params = cv2.SimpleBlobDetector_Params()

        params.filterByInertia = True
        params.minInertiaRatio = 0.7  # At least .7, that should be a circle, right?

        params.filterByConvexity = True
        params.minConvexity = 0.7  # Round enough

        return params

    def detector(self, frame):
        detector = cv2.SimpleBlobDetector_create(self.getBlobParams())
        keypoints = detector.detect(frame)
        return keypoints

    def fromBlobToBbox(self, blob):
        x, y, size = blob  # Separate the values on the blob
        x = (x - size/2)
        y = int(y - size/2)
        return x, y, size, size  # Since is a circle, the width-height should be the same

    def resize(self, frame):
        new_heigth = int(self.heigth/2)
        new_width = int(self.width/2)
        resized_frame = cv2.resize(frame, (new_width, new_heigth))
        return resized_frame

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
        self.getTrackerByName()

        success, frame = cap.read()
        frame = self.resize(frame)

        if not success:
            raise ValueError('File not found.')

        # Select bounding box with ROI
        # bbox = cv2.selectROI(frame, False)
        blob = self.detector(frame)[0].pt, self.detector(frame)[0].size
        bbox = self.fromBlobToBbox(blob)
        success = self.tracker.init(frame, bbox)

        while cap.isOpened():
            success, frame = cap.read()
            frame = self.resize(frame)
            if not success:
                break

            success, bbox = self.tracker.update(frame)
            if success:
                point1 = (int(bbox[0]), int(bbox[1]))
                point2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
                cv2.rectangle(frame, point1, point2, (255, 0, 0), 2, 1)
            else:
                cv2.putText(
                    frame, "Failed.", (100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Q Key pressed
                break
            elif key == ord('s'):  # S Key pressed
                time.sleep(5)
            elif key == ord('t'):  # T Key pressed
                input('Any key to continue')

            cv2.imshow("Ball Rolling", frame)

        cv2.release()  # Dont forget to release your video!
        cv2.destroyAllWindows()  # Close all the opened windows

if __name__ == "__main__":
    videoPath = 'assets/vid2.mp4'

    tracker = Tracker(videoPath)
    tracker.tracking()
