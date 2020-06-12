"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Author(s):

===================================================
"""

import pandas as pd
import pickle
from classifier import final_pre_proc
from strong  import WeakTeam 
from binarysearch import binarysearch
from adaboost import AdaBoost 
from ex4_tools import DecisionStump

class FlightPredictor:
    def __init__(self, path_to_weather=''):
        """
        Initialize an object from this class.
        @param path_to_weather: The path to a csv file containing weather data.
        """
        #raise NotImplementedError
        self.mod = pickle.load(open("./BinAgent" , "rb"))  

    def predict(self, x):
        """
        Recieves a pandas DataFrame of shape (m, 15) with m flight features, and predicts their
        delay at arrival and the main factor for the delay.
        @param x: A pandas DataFrame with shape (m, 15)
        @return: A pandas DataFrame with shape (m, 2) with your prediction
        """

        _dataset, y, x_factor,y_factor = final_pre_proc(x)
        self.mod.predict(x)
