# -*- coding: UTF-8 -*-

import cv2
import numpy as np
import dlib


class FeatureExtractor:
    def __init__(self, predictor_file):
        self.predictor = self._get_predictor(predictor_file)

    def _get_predictor(self, predictor_file):
        return dlib.shape_predictor(predictor_file)

    def _get_biggest_face(self, faces):
        maxarea = 0
        face = None
        for f in faces:
            if f.area() > maxarea:
                maxarea = f.area()
                face = f
        return face

    def get_landmarks(self, img):
        '''
        Get 68 facial landmarks from a gray scale image.
        If there are multiple faces detected, return the landmarks
        of the biggest face.

        Args:
        img: A gray scale image

        Returns:
        A (68,2) np array of all landmarks, each row contains the x, and y
        coordinates of a landmark.
        '''
        detector = dlib.get_frontal_face_detector()
        faces = detector(img)
        face = self._get_biggest_face(faces)
        if not face:
            return None
        shape = self.predictor(img, face)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        return np.asarray(landmarks)

    def circle_landmarks(self, img, landmarks):
        if landmarks is None:
            return
        if isinstance(landmarks, np.ndarray):
            for (x, y) in landmarks:
                cv2.circle(img, (x, y), 1, (0,0,255), thickness=5)
        else:
            raise ValueError("landmarks should be ndarray")

    def get_features(self, landmarks):
        pass
