import argparse
import preprocessing
import ExtracteFeatures
import os
import cv2
import timeit
import numpy as np
import Classification

parser = argparse.ArgumentParser()
parser.add_argument("testPath", help="path")
parser.add_argument("outPath", help="path")

args = parser.parse_args()
testPath = args.testPath
outPath = args.outPath
# target = "Test_data_evaluation/"
Y_train= ExtracteFeatures.read_labels("Training_data/")

# Read feature vector of train data from the npy file 
with open('training_features.npy', 'rb') as f:
    X_train = np.load(f,allow_pickle=True)
f.close() 

# files=os.listdir(testPath)
# for file in files:
#     img = cv2.imread("Training_data/"+file )  
#     X_train = ExtracteFeatures.extract(img,file)
# #write feature vector of each image in external file
# with open('HOG_LBP_GLCM_COLD_train.npy', 'wb') as f:
#     np.save(f, X_train)
# f.close()

clf = Classification.RandomForestClassification(X_train,Y_train)

files=os.listdir(testPath)
file1 = open(outPath+'/results.txt', 'a')
file1.truncate(0)

file2 = open(outPath+'/times.txt', 'a')
file2.truncate(0)

i = 1
for file in files:
    
    img = cv2.imread(testPath+'/'+ file )

    start = timeit.default_timer() 

    cropped_img = preprocessing.cropTxtOnly(img)

    preprocessed_img = preprocessing.preprocessing(cropped_img)

    X_test = ExtracteFeatures.extract(preprocessed_img,file)

    Y_Predicted=clf.predict([X_test])

    end= timeit.default_timer()  - start

    if(end == 0):
        end =  0.001
    if( i == len(files)):
        file1.write(str(int(Y_Predicted)))
        file2.write(str(round(end,2)))
    else:
        file1.write(str(int(Y_Predicted))+ "\n")
        file2.write(str(round(end,2))+ "\n")
    i+=1
     
file1.close() 
file2.close() 