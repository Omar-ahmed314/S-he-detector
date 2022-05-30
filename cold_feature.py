import cv2
import numpy as np
import math

class cold_feature:
    def __init__(self, numOfRhos = 7, numOfAngles = 12, poly_approx_method_accuracy = 0.01):
        self.numOfRhos = numOfRhos
        self.numOfAngles = numOfAngles
        self.poly_approx_method_accuracy = poly_approx_method_accuracy
    
    # def set_image(self, image):
    #     """
    #     Set the image to be used in the feature vectors calculations

    #     key arguments:
    #     image -- The image to be extracted
    #     """
    #     self.image = image

    def PolyApproxMethod(self, image):
        """
        This function calculates the approximated shape to the image
        by representing it with minimum points w.r.t an accuracy
        given to the constructor.
        This function is built on Polygon approximation method

        key arguments:
        image -- The needed image
        """
        approximated_images = []
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            percentage = self.poly_approx_method_accuracy * cv2.arcLength(contour, True)
            approximated_shape = cv2.approxPolyDP(contour, percentage, False)
            approximated_shape.shape = (len(approximated_shape), 2)
            approximated_images.append(approximated_shape)
        return approximated_images

    def cold_ang_mag(self, x1, y1, x2, y2):
        """
        This function calculates the euclidean distance for two points
        and their tangent angles

        key arguments:
        x1 -- x coordinate of the first point
        y1 -- y coordinate of the first point
        x2 -- x coordinate of the second point
        y1 -- y coordinate of the second point
        """
        r = abs(np.sqrt(np.power(y2 - y1, 2) + np.power(x2 - x1, 2)))
        if(x2 - x1 == 0):
            theta = (y2 - y1) / (abs(y2 - y1)) * np.pi / 2
        elif(x2 - x1 < 0):
            theta = (math.atan((y2 - y1)/(x2 - x1)) + np.pi)
        else:
            theta = math.atan((y2 - y1) / (x2 - x1)) 
        if(theta < 0):
            theta = theta + 2 * np.pi 
        if(theta * 180 / np.pi  > 180.0):
            theta = theta - 2 * np.pi
        return r, theta
    
    def distribution(self, arr):
        """
        This function calculates the R, theta distribution for contour
        in the image

        key arguments: 
        arr -- The array of the r and thetas
        """
        distrib =[]
        for shape in arr:
            for i in range(len(shape) - 1):
                distrib.append(list(self.cold_ang_mag(shape[i][0], shape[i][1], shape[i+1][0], shape[i+1][1])))
        return np.asarray(distrib)

    def convertPolarToSpatial(self, polarCoordArr):
        """
        This function convert the polar coordinates into 
        the spatial coordinates

        key arguments:
        polarCoordArr -- The array of r and thetas points
        """
        spatialCoord = [[i[0] * math.cos(i[1]), i[0] * math.sin(i[1])] for i in polarCoordArr]
        return np.asarray(spatialCoord)
    
    def calculateFeatureVector(self, distribution):
        """
        This function calculates the feature vector from the 
        array of the rho and thetas.
        as it uses 2D histogram to map the data then converts it into 
        vector used as a feature vector

        key arguments:
        distribution -- The array of rhos and thetas
        """
        hist = np.zeros((self.numOfRhos, self.numOfAngles), dtype=int)
        angle_bin_size = 360 / self.numOfAngles
        r_inner = 5
        r_outter = 35
        rhos = np.log10(distribution[:, 0])
        thetas = np.degrees(np.asarray(distribution[:, 1]))
        rho_levels = np.log10(np.linspace(r_inner, r_outter, self.numOfRhos))
        quantized_values = np.zeros(rhos.shape, dtype = int)
        for i in range(self.numOfRhos):
            quantized_values += (rhos < rho_levels[i])
        for i, r_bin in enumerate(quantized_values):
            theta_bin = int(thetas[i] // angle_bin_size) % self.numOfAngles
            hist[r_bin - 1, theta_bin] += 1
        normalized_hist = hist / hist.sum()
        feature_vector = normalized_hist.flatten()
        return feature_vector

    def getFeatureVectors(self, image):
        """
        This function implements the whole process to 
        extract the feature vector for the image provided 
        by the constructor
        """
        poly_approximated_image = self.PolyApproxMethod(image)
        distribution = self.distribution(poly_approximated_image)
        feature_vector = self.calculateFeatureVector(distribution)
        return np.asarray(feature_vector)