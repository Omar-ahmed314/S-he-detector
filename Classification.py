from skimage.feature import hog
from skimage import io,color
from skimage.transform import resize
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics
import os
from  LBP_descriptor import LocalBinaryPatterns
import commonfunctions as cf
import cv2
# import pandas as pd
import csv
from skimage.feature import greycomatrix, greycoprops
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import timeit



def RandomForestClassification(X_train,Y_train):
    clf=RandomForestClassifier(n_estimators=1000)
    clf.fit(X_train,Y_train)
    return clf
    