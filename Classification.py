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
    
def linearSVCclassifier(X_train,Y_train):
    clf=LinearSVC(C=300.0)
    clf.fit(X_train,Y_train )
    return clf

def SVM_SVC_classifier(X_train,Y_train ):
    clf=svm.SVC(C=300.0)
    clf.fit(X_train,Y_train )
    return clf

def AdaBoostClassification(X_train,Y_train):
    clf=AdaBoostClassifier(n_estimators=400)
    clf.fit(X_train,Y_train)
    return clf

def Gradient_Boost_Classifier(X_train,Y_train):
    clf=GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=8,random_state=0)
    clf.fit(X_train,Y_train)
    return clf

def MLP(X_train,Y_train):
    clf= MLPClassifier(random_state=0, max_iter=100, hidden_layer_sizes=2)
    clf.fit(X_train,Y_train)
    return clf







