# -*- coding: UTF-8 -*-

from collections import OrderedDict
import cv2
import numpy as np
import dlib



FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])


class FeatureExtractor:
    def __init__(self, predictor_file, aligner):
        '''
        Initialize a FeatureExtractor.

        Args:
        predictor_file: a front face shape predictor file
        aligner: a FaceAligner object that normalizes a face shape
        '''
        self.predictor = self._get_predictor(predictor_file)
        self.aligner = aligner

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

    def get_aligned_face(self, img, landmarks):
        return self.aligner.align(img, landmarks)

    def get_features(self, landmarks):
        pass


class FaceAligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=256, desiredFaceHeight=256):
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

    def align(self, img, landmarks):
        # get left and right eye coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
        leftEyePts = landmarks[lStart:lEnd]
        rightEyePts = landmarks[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        dist = np.linalg.norm(leftEyeCenter - rightEyeCenter)
        desiredDist = 1.0 - 2 * self.desiredLeftEye[0]
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # center of two eyes
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # rotation matrix
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        # TODO: figure this out
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(img, M, (w, h),flags=cv2.INTER_CUBIC)

        return output
