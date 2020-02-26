# -*- coding: UTF-8 -*-

'''
This script generates training data.
It reads in images with emotions and extracts normalized
landmarks, and then saves them with the label, to a csv file.

Local installation of the faceemo module is required to run
the script. To do so, from command line:
    $cd src
    $pip install .

To use the script, the images should be stored under the same
folder, and each of the emotions has a corresponding folder.
For example,
    root_folder/
      happy/
        img1.tiff
        img2.tiff
        ..
      sad/
        ..
      neutral/
        ..

When invoking the script, the root_folder needs to be provided
at prompt.
'''


import os
from os.path import isfile, join
import csv
import cv2
from warnings import warn
from faceemo.utils import *


LABELS = ('happy', 'sad', 'neutral')


def extract_landmarks(imgfile, detector, aligner):
    # try:
        img = cv2.imread(imgfile)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        landmarks = detector.get_landmarks(gray)
        if landmarks is not None:
            return aligner.align_landmarks(landmarks, round=True)
        else:
            warn(imgfile + " does not have valid features.")
            return None
    # except:
    #     warn(imgfile + ": something went wrong")
    #     return None


def convert_landmarks_to_row(landmarks, label):
    row = np.append(landmarks.flatten(), label)
    return row


if __name__ == '__main__':
    # get root folder
    root_folder = input("Root folder of images:")

    # prepare for feature extraction
    predictor_file = os.path.abspath("../data/support/shape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor(predictor_file)
    detector = FaceDetector(predictor)
    aligner = FaceAligner()

    # open csv file as output file
    with open('train.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quoting=csv.QUOTE_MINIMAL)

        # walk through each of the label sub folders
        for label in LABELS:
            dir = join(root_folder, label)
            files = [join(dir, f) for f in os.listdir(dir) if isfile(join(dir, f))
                     and not f.startswith(".")]
            rows = []
            for f in files:
                landmarks = extract_landmarks(f, detector, aligner)
                print(landmarks.T)
                row = convert_landmarks_to_row(landmarks, label)
                rows.append(row)
            writer.writerows(rows)
