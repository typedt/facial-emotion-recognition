# -*- coding: UTF-8 -*-

import cv2
import numpy as np
import dlib
import os
import joblib
from faceemo.utils import *
from faceemo.feature import *


CLF_FILE = os.path.abspath('data/classify/classifier.joblib')
CLF_FILE = os.path.abspath('data/classify/classifier_neighbors.joblib')
PCA_PARAM_FILE = os.path.abspath('data/classify/pca_params.joblib')
PREDICTOR_FILE = os.path.abspath("data/support/shape_predictor_68_face_landmarks.dat")


def apply_clf(landmarks, clf):
    features = features_by_neighbors(landmarks)
    return clf.predict([features])


if __name__ == '__main__':
    predictor = dlib.shape_predictor(PREDICTOR_FILE)
    detector = FaceDetector(predictor)
    aligner = FaceAligner()
    clf = joblib.load(CLF_FILE)
    pca_params = joblib.load(PCA_PARAM_FILE)

    # open webcam
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        # capture
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = detector.get_landmarks(gray)
        if landmarks is not None:
            aligned = aligner.align_image(gray, landmarks)
            landmarks_af = aligner.align_landmarks(landmarks, round=True)
            prediction = apply_clf(landmarks_af, clf)
            circle_landmarks(aligned, landmarks_af)
            cv2.imshow('Aligned face', aligned)
            print(prediction[0])

        # exit when user presses `q`
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release web cam handle and clean up
    cap.release()
    cv2.destroyAllWindows()
