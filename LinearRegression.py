#!/usr/bin/env python3
import numpy

class LinearRegression(object):

    # Constructor. Initailize Constants.
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.__M = 0
        self.__theta0 = 2
        self.__theta1 = 2

        # defining Alpha I.E length of steps in gradient descent Or learning Rate.
        self.__alpha = 0.01

    def predict(self,x):
        return (self.__theta0 + x * self.__theta1)
    
    # Cost Function fot theta0.
    def __cost_theta0(self,X,Y):
        
        sqrerror = 0.0
        for i in range(0,X.__len__()):
            sqrerror += (self.predict(X[i]) - Y[i])
        return (1/self.__M * sqrerror)
    
    # Cost Function fot theta1.
    def __cost_theta1(self,X,Y):
        sqrerror = 0.0
        for i in range(0,X.__len__()):
            sqrerror += (self.predict(X[i]) - Y[i]) * X[i]
        return (1/self.__M * sqrerror)


    # training Data :
    # Finding Best __theta0 and __theta1 for data such that the Squared  Error is Minimized.
    def train(self,features,target):
        
        # Validate Data
        if not features.__len__() == target.__len__():
            raise Exception("features and target should be of same length")

        # Initailize M with Size of X and Y
        self.__M = features.__len__()
        
        # gradient descent
        prevt0, prevt1 = self.__theta0 , self.__theta1
        
        while True:
            tmp0 = self.__theta0 - self.__alpha * (self.__cost_theta0(features,target))
            tmp1 = self.__theta1 - self.__alpha * (self.__cost_theta1(features,target))
           
            self.__theta0, self.__theta1 = tmp0, tmp1

            print("theta0(b) :", self.__theta0)
            print("theta1(m) :", self.__theta1)
            
            if "0:o.5f".format(prevt0) == "0:o.5f".format(self.__theta0) and "0:o.5f".format(prevt1) == "0:o.5f".format(self.__theta1):
                break
            
            prevt0, prevt1 = self.__theta0 , self.__theta1


        # Training Completed.
        # log __theta0 __theta1
        print("theta0(b) :", self.__theta0)
        print("theta1(m) :", self.__theta1)
