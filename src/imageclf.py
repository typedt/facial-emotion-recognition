# -*- coding: UTF-8 -*-

import sys
import dlib
import os
import cv2
import numpy as np
import joblib
from faceemo.utils import *
from faceemo.feature import *


CLF_FILE = os.path.abspath('data/classify/classifier_neighbors.joblib')
PCA_PARAM_FILE = os.path.abspath('data/classify/pca_params.joblib')
PREDICTOR_FILE = os.path.abspath("data/support/shape_predictor_68_face_landmarks.dat")


def apply_clf(landmarks, clf):
    features = features_by_neighbors(landmarks)
    return clf.predict([features])


if __name__ == '__main__':
    imgfile = input("Input image path:")
    if not imgfile:
        sys.exit(0)

    img = cv2.imread(imgfile)
    if img is None:
        raise RuntimeError("Can no open or can not read img file.")

    predictor = dlib.shape_predictor(PREDICTOR_FILE)
    detector = FaceDetector(predictor)
    aligner = FaceAligner()
    clf = joblib.load(CLF_FILE)
    pca_params = joblib.load(PCA_PARAM_FILE)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    landmarks = detector.get_landmarks(gray)
    img_aligned = aligner.align_image(gray, landmarks)
    cv2.imshow('Aligned face', img_aligned)

    landmarks_aligned = aligner.align_landmarks(landmarks, round=True)
    prediction = apply_clf(landmarks_aligned, clf)
    print("Predicted emotion: " + prediction[0])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
