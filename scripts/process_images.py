# -*- coding: UTF-8 -*-

'''
This script generates training data.
It reads in images with emotions and extracts normalized
landmarks, and then saves them with the label, to a csv file.
At the same time if a feature extract function is selected,
the features will be extracted from landmarks and saved to
a csv file.

Local installation of the faceemo module is required to run
the script. To do so, from command line:
    $cd src/faceemo
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
at prompt. The output file names of landmarks and features are
optional at prompt. By default, they will be written to
landmarks.csv and features.csv.
'''


import sys
import os
from os.path import isfile, join, basename, isdir
import csv
import cv2
from warnings import warn
from faceemo.utils import *
from faceemo.feature import *


WRITE_IMAGE = False
SHOW_PROGRESS = True
FEATURE_EXTRACT_FUNC = features_by_neighbors
LABELS = ('happy', 'sad', 'neutral')


def dflt_input(default, prompt):
    '''
    Get user input from prompt, if none provided, then use default value.
    '''
    result = input(prompt)
    if not result:
        result = default
    return result


def list_files(dir):
    '''
    Only list out files, no directories or hidden files.
    '''
    return [join(dir, f) for f in os.listdir(dir) if \
             isfile(join(dir, f)) and not f.startswith(".")]


def create_output_dir_if_not_exist(rootdir, label):
    outputdir = join(rootdir, label)
    if not isdir(outputdir):
        os.makedir(outputdir)
    return outputdir


def features_to_row(features, label):
    row = np.append(features.flatten(), label)
    return row


if __name__ == '__main__':
    root_folder = input("Root folder of images:")
    if not root_folder:
        print("Root folder of images must be provided.")
        sys.exit(1)
    if WRITE_IMAGE:
        output_root_dir = dflt_input('.', "Output image root folder:")
    landmark_csv = dflt_input('landmarks.csv', 'Landmark output CSV file name:')
    feature_csv = dflt_input('features.csv', 'Feature output CSV file name:')

    predictor_file = os.path.abspath("../data/support/shape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor(predictor_file)
    detector = FaceDetector(predictor)
    aligner = FaceAligner(desiredLeftEye=(0.35, 0.4),
                          desiredWidth=256, desiredHeight=256)

    # CSV file rows to store landmarks and labels
    landmarks_label_pairs = []

    for label in LABELS:
        print("Processing images labeled with %s ..." % label)
        imgdir = join(root_folder, label)
        files = list_files(imgdir)
        if WRITE_IMAGE:
            outputdir = create_output_dir_if_not_exist(output_root_dir, label)
        for f in files:
            if SHOW_PROGRESS:
                print("  processing image %s ..." % f)
            img = cv2.imread(f)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            landmarks = detector.get_landmarks(gray)
            if landmarks is not None:
                landmarks_af = aligner.align_landmarks(landmarks)
                landmarks_label_pairs.append((landmarks_af, label))
                if WRITE_IMAGE:
                    image_af = aligner.align_image(gray, landmarks)
                    fout = join(outputdir, basename(f))
                    cv2.imwrite(fout, image_af)
            else:
                warn(f + " does not have valid features.")
                continue

    print("Writing landmarks to csv file %s ..." % landmark_csv)
    with open(landmark_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quoting=csv.QUOTE_MINIMAL)
        for pair in landmarks_label_pairs:
            row = features_to_row(pair[0], pair[1])
            writer.writerow(row)

    print("Writing features to csv file %s ..." % feature_csv)
    with open(feature_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quoting=csv.QUOTE_MINIMAL)
        # extract feature
        for pair in landmarks_label_pairs:
            features = FEATURE_EXTRACT_FUNC(pair[0])
            row = features_to_row(features, pair[1])
            writer.writerow(row)
