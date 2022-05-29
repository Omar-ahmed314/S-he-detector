import cv2
import numpy as np
import math

class cold_feature:
    def __init___(self, images, histogram_shape = (7, 12), poly_approx_method_accuracy = 0.01):
        self.numOfRhos = histogram_shape[0]
        self.numOfAngles = histogram_shape[1]
        self.accuracy = poly_approx_method_accuracy
        self.images = images
    
    def PolyApproxMethod(self, image):
        approximated_images = []
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            percentage = self.accuracy * cv2.arcLength(contour, True)
            approximated_shape = cv2.approxPolyDP(contour, percentage, False)
            approximated_shape.shape = (len(approximated_shape), 2)
            approximated_images.append(approximated_shape)
        return approximated_images

    def cold_ang_mag(self, x1, y1, x2, y2):
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
        distrib =[]
        for shape in arr:
            for i in range(len(shape) - 1):
                distrib.append(list(self.cold_ang_mag(shape[i][0], shape[i][1], shape[i+1][0], shape[i+1][1])))
        return np.asarray(distrib)

    def convertPolarToSpatial(self, polarCoordArr):
        spatialCoord = [[i[0] * math.cos(i[1]), i[0] * math.sin(i[1])] for i in polarCoordArr]
        return np.asarray(spatialCoord)
    
    def calculateFeatureVector(self, distribution):
        hist = np.zeros((self.numOfRhos, self.numOfAngels), dtype=int)
        angle_bin_size = 360 / self.numOfAngels
        r_inner = 5
        r_outter = 35
        rhos = np.log10(distribution[:, 0])
        thetas = np.asarray(distribution[:, 1])
        rho_levels = np.log10(np.linspace(r_inner, r_outter, self.numOfRhos))
        quantized_values = np.zeros(rhos.shape, dtype = int)
        for i in range(self.numOfRhos):
            quantized_values += (rhos < rho_levels[i])
        for i, r_bin in enumerate(quantized_values):
            theta_bin = int(thetas[i] // angle_bin_size) % self.numOfAngels
            hist[r_bin - 1, theta_bin] += 1
        normalized_hist = hist / hist.sum()
        feature_vector = normalized_hist.flatten()
        return feature_vector

    def getFeatureVectors(self):
        feature_vectors = []
        for image in self.images:
            poly_approx_method = self.PolyApproxMethod(image)
            distribution = self.distribution(poly_approx_method)
            feature_vector = self.calculateFeatureVector(distribution)
            feature_vectors.append(feature_vector)
        return np.asarray(feature_vectors)