# -*- coding: UTF-8 -*-

import cv2
import numpy as np
import dlib
import os
from faceemo.utils import *


if __name__ == '__main__':
    # init the FeatureExtractor object
    predictor_file = os.path.abspath("data/support/shape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor(predictor_file)
    detector = FaceDetector(predictor)
    aligner = FaceAligner()

    # open webcam
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        # capture
        ret, frame = cap.read()

        # convert to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # mark landmarks
        landmarks = detector.get_landmarks(gray)
        if landmarks is not None:
            # circle_landmarks(frame, landmarks)
            # cv2.imshow('Image Captured with Landmarks', frame)
            aligned = aligner.align_image(gray, landmarks)
            landmarks_af = aligner.align_landmarks(landmarks, round=True)
            circle_landmarks(aligned, landmarks_af)
            cv2.imshow('Aligned face', aligned)

        # exit when user presses `q`
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release web cam handle and clean up
    cap.release()
    cv2.destroyAllWindows()
