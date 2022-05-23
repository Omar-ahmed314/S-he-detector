
from  LBP_descriptor import LocalBinaryPatterns
import commonfunctions as cf
from sklearn.svm import LinearSVC
from sklearn import metrics
# from imutils import paths
import argparse
import cv2
import os
#--------------------------------------load training data----------------------------
#Females
maleData=cf.load_images("our dataset/train/Female")
femaleData=cf.load_images("our dataset/train/Male")
trainingData=[]
labels=[]   
scale=0.5
print("loading is done")
for img in femaleData:
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    trainingData.append(img)
    labels.append(1)
for img in maleData:
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    trainingData.append(img)
    labels.append(0)
print("reading is done")
#---------------------------------- train the classifier using LBP ----------------------
desc = LocalBinaryPatterns(35, 8)
LBP_features=[]
for img in trainingData:
    print("hist")
    hist = desc.describe(img)
    LBP_features.append(hist)
print(LBP_features[0])    
# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42)
model.fit(LBP_features, labels)
# # test sample
imgT = cv2.imread("our dataset/test/F108.jpg" )
test_img= cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY)
#desc_test = LocalBinaryPatterns(1, 8)
test_feature=desc.describe(test_img)
print("**********")
print(test_feature)
y_pred = model.predict([test_feature])    
print(y_pred)
    
#---------------------------------- test data --------------------
maleTestData=cf.load_images("our dataset/test/Female")
femaleTestData=cf.load_images("our dataset/test/Male")
LBP_Test_features=[]

# for imgT in femaleTestData:
#     imgT= cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY)
#     print("hist test")
#     width = int(imgT.shape[1] * scale)
#     height = int(imgT.shape[0] * scale)
#     dim = (width, height)    
#     cv2.resize(imgT, dim, interpolation = cv2.INTER_AREA)
#     hist = desc.describe(imgT)
#     LBP_Test_features.append(hist)
# # for imgT in maleTestData:
# #     imgT= cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY)
# #     print("hist test")
# #     hist = desc.describe(imgT)
# #     LBP_Test_features.append(hist)
# #Predict the response for test dataset
# y_pred = model.predict(LBP_Test_features)    
# print(y_pred)
# print("ACCURACY")
# # Model Accuracy: how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(labels, y_pred))    