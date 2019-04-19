import cv2
import numpy as np


class Tracker:
    def __init__(self, fileInput, ):
        self.videoPath = fileInput
        self.tracker = None

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
            print('Wrong Tracker name')
        except Exception:
            raise

    def trackMario(self):
        cap = cv2.VideoCapture(self.videoPath)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            cv2.imshow("Mario", frame)

if __name__ == "__main__":
    videoPath = 'assets/video.mp4'

    tracker = Tracker(videoPath)
    tracker.getTrackerByName()
    tracker.trackMario()
