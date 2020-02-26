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


def circle_landmarks(img, landmarks, color=(0,0,255), thickness=2):
    if landmarks is None:
        return
    if isinstance(landmarks, np.ndarray):
        for (x, y) in landmarks:
            cv2.circle(img, (x, y), 1, color, thickness=thickness)


class FaceDetector:

    def __init__(self, predictor):
        '''
        Initialize a FeatureExtractor.

        Args:
        predictor: a dlib shape predictor
        '''
        self.predictor = predictor

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


class FaceAligner:

    def __init__(self, desiredLeftEye=(0.35, 0.35),
                 desiredWidth=256, desiredHeight=256):
        '''
        Initialize a FaceAligner.

        Args:
        desiredLeftEye: the desired location of the center of the left eye,
                        compared with the entire output image of the aligner.
                        The biggner the numbers are, the more zoomed-out the
                        face will be.
        desiredWidth: width of the output image.
        desiredHeight: height of the output image.
        '''
        self.desiredLeftEye = desiredLeftEye
        self.desiredWidth = desiredWidth
        self.desiredHeight = desiredHeight

    def _getAffineMatrix(self, landmarks):
        '''
        Get affine matrix from landmark coordinates.
        '''
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
        desiredDist *= self.desiredWidth
        scale = desiredDist / dist

        # center of two eyes
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # rotation matrix
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        # recenter the output image to be center of eyes
        tX = self.desiredWidth * 0.5
        tY = self.desiredHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        return M

    def align_image(self, img, landmarks):
        '''
        Do affine transformation of the original image. The output image is
        scaled to certain size, and rotated and translated, so that the two
        eyes of the face are horizontally aligned.

        Returns:
        The output image.
        '''
        M = self._getAffineMatrix(landmarks)

        # apply the affine transformation
        (w, h) = (self.desiredWidth, self.desiredHeight)
        output = cv2.warpAffine(img, M, (w, h),flags=cv2.INTER_CUBIC)

        return output

    def align_landmarks(self, landmarks, round=False):
        '''
        Do affine transformation of the landmarks.
        https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/warp_affine/warp_affine.html
        '''
        M = self._getAffineMatrix(landmarks)
        A = M[:, :2]; B = M[:, 2]
        landmarks_ltfm = np.dot(A, landmarks.T)
        landmarks_af = np.add(landmarks_ltfm.T, B)
        if round:
            return np.rint(landmarks_af).astype(int)
        else:
            return landmarks_af
