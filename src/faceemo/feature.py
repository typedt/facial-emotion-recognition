# -*- coding: UTF-8 -*-

import numpy as np
import math
from faceemo.utils import FACIAL_LANDMARKS_68_IDXS


def _distance_x(l1, l2):
    return (l2[0] - l1[0])


def _distance_y(l1, l2):
    return (l2[1] - l1[1])


def _distance_allpairs_xy(landmarks):
    allpairsX = []
    allpairsY = []
    for i in range(len(landmarks)):
        this = landmarks[i]
        allpairsX.extend([_distance_x(this, that) for that in landmarks[i+1:]])
        allpairsY.extend([_distance_y(this, that) for that in landmarks[i+1:]])
    return allpairsX, allpairsY


def _distance_neighbors_xy(landmarks):
    X = landmarks[:,0]
    Y = landmarks[:,1]
    Xneighbors = zip(X[1:], X[:-1])
    Yneighbors = zip(Y[1:], Y[:-1])
    f = lambda x: x[0] - x[1]
    Xdistance = list(map(f, Xneighbors))
    Ydistance = list(map(f, Yneighbors))
    return Xdistance, Ydistance


def _angle(landmark):
    x, y = landmark
    return (math.atan2(y, x)*360)/(2*math.pi)


def _angle_two_pts(p1, p2):
    dx, dy = p2 - p1
    return (math.atan2(dy, dx)*360)/(2*math.pi)


def inner_mouth_angles(landmarks):
    startupper, endupper = FACIAL_LANDMARKS_68_IDXS.get("inner_mouth_right_upper")
    im_landmarks_upper = landmarks[startupper: endupper]
    pairs_upper = list(zip(im_landmarks_upper[:-1], im_landmarks_upper[1:]))
    upper = [_angle_two_pts(p1, p2) for (p1, p2) in pairs_upper]
    im_landmarks_lower = [landmarks[60], landmarks[67], landmarks[66]]
    pairs_lower = list(zip(im_landmarks_lower[:-1], im_landmarks_lower[1:]))
    lower = [_angle_two_pts(p1, p2) for (p1, p2) in pairs_lower]
    return np.asarray(upper + lower)


def features_by_neighbors_and_inner_mouth_angles(landmarks):
    '''
    A combinations of features, from both `features_by_neighbors`
    and `inner_mouth_angles`
    '''
    neighbors = features_by_neighbors(landmarks)
    angles1 = inner_mouth_angles(landmarks)
    return np.concatenate([neighbors, angles1])


def features_by_neighbors(landmarks):
    '''
    Substract two neighbor landmarks.

    Returns:
    features = [
        (x1 - x0), (x2 - x1), ..., (x67 - x66),
        (y1 - y0), (y2 - y1), ..., (y67 - y66)
    ]
    '''
    Xfeatures, Yfeatures = _distance_neighbors_xy(landmarks)
    return np.asarray(Xfeatures + Yfeatures)


def features_by_neighbors_extended(landmarks, mouth=True):
    '''
    Get all features from the `features_by_neighbors` function plus,
    all pairs of landmarks in the mouth.
    '''
    if mouth:
        mouth_start, mounth_end = FACIAL_LANDMARKS_68_IDXS.get("mouth")
        mouth_landmarks = landmarks[mouth_start: mounth_end]
        mX, mY = _distance_allpairs_xy(mouth_landmarks)
        other_landmarks = landmarks[:mouth_start]
        Xdistance, Ydistance = _distance_neighbors_xy(other_landmarks)
        return np.asarray(Xdistance + mX + Ydistance + mY)
    else:
        return features_by_neighbors(landmarks)


def features_from_center(landmarks):
    '''
    Get center (mean) point (Xmean, Ymean) of all landmarks,
    substract the center point from all other landmarks.

    Return:
    features = [
        (x0 - xmean), (x1 - xmean), ..., (x67 - xmean),
        (y0 - ymean), (y1 - ymean), ..., (y67 - ymean)
    ]
    '''
    X = landmarks[:,0]
    Y = landmarks[:,1]
    Xmean = np.mean(X)
    Ymean = np.mean(Y)
    return np.concatenate([X - Xmean, Y - Ymean])


def features_from_center_with_angle(landmarks):
    '''
    Get center (mean) point (Xmean, Ymean) of all landmarks,
    calculate distance to the center point and angle

    Returns:
    features = [
        norm(landmark0 - center), angle0, norm(landmark1 - center), andle1,
        ..., norm(landmark67 - center), angle67
    ]
    '''
    center = np.mean(landmarks, axis=0)
    distances = [np.linalg.norm(l - center) for l in landmarks]
    angles = [_angle(l) for l in landmarks]
    return np.asarray(list(zip(list(distances), list(angles))))
