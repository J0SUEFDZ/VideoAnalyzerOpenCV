import cv2
import numpy as np
from time import sleep


class Detector:
    """
    Source Video: https://www.youtube.com/watch?v=heds_qDFqsA
    """
    def __init__(self, fileInput):
        self.height = 0
        self.width = 0

    def getBlobParams(self):
        params = cv2.SimpleBlobDetector_Params()

        params.filterByColor = True
        params.blobColor = 255

        return params

    def video_writer(self, file_to_write):
        out = cv2.VideoWriter(
            "assets/"+file_to_write+'.avi',
            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
            30, (int(self.width), int(self.height))
        )
        return out

    def detect_movement(self):
        cap = cv2.VideoCapture('assets/vid_test.mp4')
        if not cap.isOpened():
            raise ValueError('The path specified is not valid.')

        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        _, first_frame = cap.read()

        detector = cv2.SimpleBlobDetector_create(self.getBlobParams())
        red_video = self.video_writer('red')
        diff_video = self.video_writer('diff')
        keypoints_video = self.video_writer('keypoints')
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # 1 - Applying a mask per color.
            lower_red = np.array([0, 130, 75])
            upper_red = np.array([255, 255, 255])
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_red, upper_red)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            red = cv2.bitwise_and(frame, frame, mask=mask)
            cv2.imshow("1- Only Red", red)
            red_video.write(red)

            # 2- Difference between first frame and the rest.
            diff = cv2.absdiff(first_frame, frame)
            cv2.imshow("2- Difference", diff)
            diff_video.write(diff)

            # 3- Gey coordinates of the ball.
            keypoints = detector.detect(frame)
            im_with_keypoints = frame.copy()
            if keypoints:
                x_pt, y_pt = int(keypoints[0].pt[0]), int(keypoints[0].pt[1])
                ball_size = round(keypoints[0].size/2)
                cv2.circle(
                    im_with_keypoints,  # Image output
                    (x_pt, y_pt),  # X, Y coordiantes
                    ball_size,  # Radius of the ball.
                    (255, 0, 0),  # Color, BGR.
                    3  # Thickness of the line.
                )
            cv2.imshow("3- Keypoints", im_with_keypoints)
            keypoints_video.write(im_with_keypoints)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Q Key pressed, quit everything.
                break
            elif key == ord('s'):  # S Key pressed, sleep for 5 seconds
                sleep(5)
            elif key == ord('t'):  # T Key pressed, stop detection.
                input('Any key to continue')

        cap.release()  # Dont forget to release your video!


if __name__ == "__main__":
    detector = Detector()
    detector.detect_movement()
    cv2.destroyAllWindows()  # Close all the opened windows
