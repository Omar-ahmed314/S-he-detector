import cv2
import numpy as np

def cropTxtOnly(img):
    
    img = img[30:-450,10:-10] # Perform pre-cropping
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for i in range(90,1000,10):

        gray = 255*(gray < i).astype(np.uint8) # To invert the text to white

        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8)) # Perform noise filtering

        coords = cv2.findNonZero(gray) # Find all non-zero points (text)
        x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
        rect = img[y:y+h, x:x+w] # Crop the original image

        if (rect.size > 10000): 
            return rect
            
    return None



def preprocessing(img):
    
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # threshold the grayscale image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # use morphology erode to blur horizontally
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    # use morphology open to remove thin lines from dotted lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 17))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    # find contours
    cntrs = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

    return thresh
        