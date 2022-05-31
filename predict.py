import argparse
import preprocessing
import ExtracteFeatures
import os
import cv2
import timeit
import numpy as np
import Classification
from cold_feature import cold_feature

parser = argparse.ArgumentParser()
parser.add_argument("testPath", help="path")
parser.add_argument("outPath", help="path")

args = parser.parse_args()
testPath = args.testPath
outPath = args.outPath
# target = "Test_data_evaluation/"
#-------------------delete---------------------------------------------
Y_train= ExtracteFeatures.read_labels("Training_data/")

#write train labels in external file
with open('final models/85.13/final_train_labels.npy', 'wb') as f:
    np.save(f, Y_train)
f.close()  
#------------------------delete-------------------------------------------
# then uncomment this:
# # Read feature vector of train data from the npy file 
# with open('final models/85.13/final_train_labels.npy', 'rb') as f:
#     Y_train = np.load(f,allow_pickle=True)
#f.close() 
#-------------------------------------------------------------------

# Read feature vector of train data from the npy file 
with open('final models/85.13/ALL_features.npy', 'rb') as f:
    X_train = np.load(f,allow_pickle=True)
f.close() 
# print(len(X_train[0]))

# files=os.listdir(testPath)
# for file in files:
#     img = cv2.imread("Training_data/"+file )  
#     X_train = ExtracteFeatures.extract(img,file)
# #write feature vector of each image in external file
# with open('HOG_LBP_GLCM_COLD_train.npy', 'wb') as f:
#     np.save(f, X_train)
# f.close()

clf = Classification.Gradient_Boost_Classifier(X_train,Y_train)

# with open('clf.npy', 'wb') as f:
#     np.save(f, clf)
# f.close()

# with open('clf.npy', 'rb') as f:
#     clf = np.load(f,allow_pickle=True)
# f.close() 

cold = cold_feature()
# print("hi")
files=os.listdir(testPath)
file1 = open(outPath+'/results.txt', 'a')
file1.truncate(0)

file2 = open(outPath+'/times.txt', 'a')
file2.truncate(0)

i = 1
for file in files:

    try:
        img = cv2.imread(testPath+'/'+ file )

        start = timeit.default_timer() 

        cropped_img = preprocessing.cropTxtOnly(img)

        preprocessed_img = preprocessing.preprocessing(cropped_img)

        cv2.imwrite("testing/test.jpg", preprocessed_img)

        X_test = ExtracteFeatures.extract(cold)

        Y_Predicted=clf.predict([X_test])

    except: 
        Y_Predicted=-1
    
    end= timeit.default_timer()  - start
   
    
    if( i == len(files)):
        file1.write(str(int(Y_Predicted)))
        if(round(end,3) == 0.0):
            end =  0.001
            file2.write(str(end))
        else: 
            file2.write(str(round(end,2)))
    else:
        file1.write(str(int(Y_Predicted))+ "\n")
        if(round(end,3) == 0.0):
            end =  0.001
            file2.write(str(end)+ "\n")
        else: 
            file2.write(str(round(end,2))+ "\n")
    i+=1
     
file1.close() 
file2.close() 