# -*- coding: UTF-8 -*-

import cv2
import numpy as np
import dlib
import curses
import os
from faceemo.utils import FeatureExtractor, FaceAligner


if __name__ == '__main__':
    # init the FeatureExtractor object
    predictor_file = os.path.abspath("data/support/shape_predictor_68_face_landmarks.dat")
    aligner = FaceAligner()
    extractor = FeatureExtractor(predictor_file, aligner)

    # open webcam
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        # capture
        ret, frame = cap.read()

        # convert to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # mark landmarks
        landmarks = extractor.get_landmarks(gray)
        extractor.circle_landmarks(frame, landmarks)
        cv2.imshow('Image Captured with Landmarks', frame)

        # show aligned face with landmarks
        if landmarks is not None:
            extractor.circle_landmarks(gray, landmarks)
            aligned = extractor.get_aligned_face(gray, landmarks)
            cv2.imshow('Aligned face', aligned)

        # exit when user presses `q`
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# release web cam handle and clean up
cap.release()
cv2.destroyAllWindows()
