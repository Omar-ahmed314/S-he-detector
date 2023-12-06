import argparse
import preprocessing
import ExtracteFeatures
import os
import cv2
import timeit
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("testPath", help="path")
parser.add_argument("outPath", help="path")

args = parser.parse_args()
testPath = args.testPath
outPath = args.outPath

clf = pickle.load(open("final models/85.13/Classification_Model.pkl", "rb"))

files = os.listdir(testPath)
print(len(files))
file1 = open(outPath + "/results.txt", "w")
file1.truncate(0)

file2 = open(outPath + "/times.txt", "w")
file2.truncate(0)

i = 1
for file in files:
    try:
        img = cv2.imread(testPath + "/" + file)

        start = timeit.default_timer()

        cropped_img = preprocessing.cropTxtOnly(img)

        preprocessed_img = preprocessing.preprocessing(cropped_img)

        X_test = ExtracteFeatures.extract(preprocessed_img)

        Y_Predicted = clf.predict([X_test])

    except Exception as err:
        print("Exception error", err)
        Y_Predicted = -1

    end = timeit.default_timer() - start

    if i == len(files):
        file1.write(str(int(Y_Predicted)))
        if round(end, 3) == 0.0:
            end = 0.001
            file2.write(str(end))
        else:
            file2.write(str(round(end, 2)))
    else:
        file1.write(str(int(Y_Predicted)) + "\n")
        if round(end, 3) == 0.0:
            end = 0.001
            file2.write(str(end) + "\n")
        else:
            file2.write(str(round(end, 2)) + "\n")
    i += 1

file1.close()
file2.close()
