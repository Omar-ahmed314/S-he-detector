from xml.sax.handler import feature_validation
import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops
from skimage import data

from skimage import io
import cv2
from sklearn.metrics import homogeneity_score
import commonfunctions as cf

# to test
image = cv2.imread("M160.jpg")  

 
def GLCM (image):

    # convert image to gray
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # features will be extracted from GLC matrix
    # dissimilarity 0
    #correlation 1
    #contrast 2
    #homogeneity 3
    featureVector=[]

    glcm = graycomatrix(image, distances=[5], angles=[0], levels=256,
                            symmetric=True, normed=True)
    featureVector.append(graycoprops(glcm, 'dissimilarity')[0, 0])

    featureVector.append(graycoprops(glcm, 'correlation')[0, 0])
   
    featureVector.append(graycoprops(glcm, 'contrast')[0, 0])
    
    featureVector.append(graycoprops(glcm, 'homogeneity')[0, 0])
    print(featureVector)
    return featureVector

# to test
GLCM(image)    