from skimage.feature import hog
from skimage.transform import resize
import numpy as np 
import os
from  LBP_descriptor import LocalBinaryPatterns
import commonfunctions as cf
import cv2
import csv
from skimage.feature import greycomatrix, greycoprops
from cold_feature import cold_feature


# ICDAR LABELS 
labels_ICDAR=[]
with open("our dataset/train_answers.csv", 'r') as file:
    csvreader = csv.reader(file, delimiter=',')
    rows= np.array(list(csvreader))[1:].astype(float).astype(int)
for row in rows:
    labels_ICDAR.append(row[1])
    
def get_label_ICDAR(img):
    if img[0]=='0':
        if img[1]=='0': 
            return labels_ICDAR[int(img[2])-1]
        else: 
            return labels_ICDAR[int(img[1:3])-1]
    else: 
         return labels_ICDAR[int(img[0:3])-1]
        
def read_labels(path): 
    y=[]
    files = os.listdir(path)

    for file in files:
        if file[0]=='F':
            y.append(0)
        elif file[0]=='M':
            y.append(1)
        else: # ICDAR dataset
            y.append(get_label_ICDAR(file[1:4]))
    y=np.array(y).astype(float)
    return y 

def HOG(img):
    img = np.array(resize(img,(128,64))) 
    feature_vector, hog_image = hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(3,3), visualize=True)
    return feature_vector,hog_image

def lbp():
    return LocalBinaryPatterns(24, 8)

def GLCM (image):

    # convert image to gray
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    featureVector=[]

    glcm = greycomatrix(image, distances=[5], angles=[0], levels=256,
                            symmetric=True, normed=True)
    featureVector.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    featureVector.append(greycoprops(glcm, 'correlation')[0, 0])
    featureVector.append(greycoprops(glcm, 'contrast')[0, 0])
    featureVector.append(greycoprops(glcm, 'homogeneity')[0, 0])
    return featureVector

def extract(image , file):
    HOG_test=[]
    LBP_test=[]
    GLCM_test=[]
    cold = cold_feature(10, 10)

    #------------------- HOG feature------------------------
    feature_vector,hog_image=HOG(image)
    HOG_test.append(feature_vector)
    #--------------------------------------------------------

    # #------------------- LBP feature------------------------
    image = cf.downSize(image , 0.5)
    hist = lbp().describe(image)
    LBP_test.append(hist)
    #--------------------------------------------------------
    
    #------------------- GLCM feature------------------------
    img = cv2.imread("Test_data_evaluation/"+file )  
    GLCM_test=GLCM(img)
    #-------------------- Cold feature ------------------------
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #-------------------- Cold feature ------------------------
     # threshold the grayscale image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cold_feature_vector = cold.getFeatureVectors(thresh)
    #--------------------------------------------------------

    # concatenate all the features in X_train
    feature_test_temp=(np.hstack((HOG_test,LBP_test))).tolist()
    feature_test_temp1=(np.hstack((feature_test_temp[0],cold_feature_vector))).tolist()
    feature_test_temp2=(np.hstack((feature_test_temp1[0],GLCM_test))).tolist()
    
    return feature_test_temp2
 
