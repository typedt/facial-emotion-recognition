# -*- coding: UTF-8 -*-

'''
This script takes a pre stored aligned landmarks from a csv file,
and extract features from landmarks using a selected feature
extraction function, and store results in a csv file with
corresponding labels.
'''


import csv
import numpy as np
from faceemo.feature import *


FEATURE_EXTRACT_FUNC = features_by_neighbors


def dflt_input(default, prompt):
    '''
    Get user input from prompt, if none provided, then use default value.
    '''
    result = input(prompt)
    if not result:
        result = default
    return result


if __name__ == '__main__':
    landmark_csv = input("Landmark csv file:")
    if not landmark_csv:
        print("Landmark csv file must be provided.")
        sys.exit(1)
    output_csv = dflt_input('features_from_landmark.csv', "Output file name:")

    allrows = np.loadtxt(landmark_csv, delimiter=',', dtype='str')
    X = allrows[:, :-1].astype('float')
    T = allrows[:, -1]

    with open(output_csv, 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',',
                            quoting=csv.QUOTE_MINIMAL)
        for i in range(len(X)):
            landmarks = X[i]
            landmarks = np.array(landmarks).reshape(68, 2)
            label = T[i]
            features = FEATURE_EXTRACT_FUNC(landmarks)
            row = np.append(features.flatten(), label)
            writer.writerow(row)
