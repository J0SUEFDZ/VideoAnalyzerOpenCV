# Video Analyzer with OpenCV

## What it does

This project analyzes videos using common OpenCV techniques: **color masking** (e.g. isolate red/orange objects), **frame differencing** (movement vs. a reference frame), and **blob detection** (locate simple objects like a ball and get their coordinates). It reads a video from AWS S3, runs these three analyses in parallel, and writes three annotated output videos back to S3 (mask-only, difference, and keypoints with a circle around the detected object). It can run locally or as an **AWS Lambda** function triggered when a new video is uploaded to S3, with optional transcoding via Elastic Transcoder.

## Context

Part of a **startup project at 4Geeks** for image identification and video analysis. The implementation and concepts were documented in the 4Geeks blog: [Analyzing Videos With Python](https://blog.4geeks.io/analyzing-videos-with-python/).

## Use cases

Video analysis is used in security, sports, cinema, transport, health, and home automation. This code focuses on:

- **Movement detection** — What changed between the first frame and the rest (frame differencing).
- **Color-based filtering** — Isolate objects by color (HSV mask) and ignore the rest.
- **Object detection and position** — Find blob-like objects (e.g. a ball), get (x, y) and size, and draw them on the video.

Defining *what* you want to analyze (detect vs. track, how many objects, entry/exit rules) drives *how* you implement it; this project demonstrates a simple pipeline for detection and masking.

## How it was implemented

- **Mask**: Convert each frame to HSV, apply `cv2.inRange` with lower/upper red (e.g. `[0, 130, 75]`–`[255, 255, 255]`), then `cv2.bitwise_and` so only the colored region is visible.
- **Frame difference**: Store the first frame, then `cv2.absdiff(first_frame, frame)` per frame to highlight what moved (e.g. ball vs. static background).
- **Blob detection**: `cv2.SimpleBlobDetector` with color filter (e.g. blobColor 255 for dark blobs). Detector returns keypoints (position and size); the code draws a circle around the first keypoint and writes a “keypoints” video.
- **Pipeline**: One video is read from an S3 URL; three `VideoWriter`s produce `red_*`, `diff_*`, and `keypoints_*` outputs. These are uploaded to S3 and optionally sent to AWS Elastic Transcoder for transcoding.

## How to run it

**Local**

- Create a virtual environment and install dependencies:  
  `pip install -r requirements.txt`
- AWS credentials: set `ACCESS_KEY_ID` and `SECRET_ACCESS_KEY` in the environment, or use a local `settings` module with those variables.
- Run the analyzer (example):  
  `python analyzer.py`  
  The default in `__main__` uses a fixed S3 bucket and key (`blog4geeks`, `original/vid_test.mp4`); change those in `analyzer.py` for your bucket and object key.

**AWS Lambda**

- Package: `./make.sh` (or `zip analyzer.zip analyzer.py lambda_function.py`).
- Deploy the zip as a Lambda function and set the handler to `lambda_function.lambda_handler`.
- Configure an S3 trigger so that when a video is uploaded to the bucket, the event passes `bucket` and `object key` to the handler; the handler creates a `Detector` and runs `detect_movement()`.

**Dependencies**: Python 3, NumPy, OpenCV (`opencv-python`, `opencv-contrib-python`). For Lambda, include these in the deployment package (or use a layer). For AWS features: `boto3` (add to `requirements.txt` if missing).

## Technologies used

Python, OpenCV (cv2), NumPy, AWS (S3, Lambda, Elastic Transcoder), boto3.

---

*Startup project at 4Geeks for image identification and video analysis. See [Analyzing Videos With Python](https://blog.4geeks.io/analyzing-videos-with-python/) for the accompanying blog post.*
