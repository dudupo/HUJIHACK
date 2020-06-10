from models import abcModel, DecisionTree
from adaboost import AdaBoost, AdaBoostList
from itertools import combinations
import numpy as np
import pandas as pd

from datetime import date
# class WeakFactory:
#     class WeakLernerByFeature(abcModel):

#         def __init__(self, _model):
#             self.mod = _model

#         def fit(self, X, y):
#             self.mod.fit(X,y)

#         def predict(self, X):
#             return self.mod.predict()

    
#     def __init__ (self):
#         pass
    
#     @staticmethod
#     def CreateWeaks(self):

#         return { 

#                 "DayOfWeek" : WeakFactory ( DecisionTree(max_depth=2) ),
#                 "FlightDate" : WeakFactory ( DecisionTree(max_depth=2) ),
#                 "Reporting_Airline" : WeakFactory ( DecisionTree(max_depth=2) ),
#                 "Tail_Number" : WeakFactory ( DecisionTree(max_depth=2) ),
#                 "Flight_Number_Reporting_Airline" : WeakFactory ( DecisionTree(max_depth=2) ),
#                 "Origin" : WeakFactory ( DecisionTree(max_depth=2) ),
#                 "OriginCityName" : WeakFactory ( DecisionTree(max_depth=2) ),
#                 "OriginState" : WeakFactory ( DecisionTree(max_depth=2) ),
#                 "Dest" : WeakFactory ( DecisionTree(max_depth=2) ),
#                 "DestCityName" : WeakFactory ( DecisionTree(max_depth=2) ),
#                 "DestState" : WeakFactory ( DecisionTree(max_depth=2) ),
#                 "CRSDepTime" : WeakFactory ( DecisionTree(max_depth=2) ),
#                 "CRSArrTime" : WeakFactory ( DecisionTree(max_depth=2) ),
#                 "CRSElapsedTime" : WeakFactory ( DecisionTree(max_depth=2) ),
#                 "Distance" : WeakFactory ( DecisionTree(max_depth=2) ),
#                 "ArrDelay" : WeakFactory ( DecisionTree(max_depth=2) ),
#                 "DelayFacto" : WeakFactory ( DecisionTree(max_depth=2) )
#         }

def generateTeamClass(featuers):
    class WeakTeam(DecisionTree):

        def __init__(self):
            DecisionTree.__init__(self, max_depth = 3)
            self.featuers = featuers

        def filterX( self, X ):
            
            return [ X[featuer] for featuer in self.featuers ]

        def train(self, X, y):
            super().fit(
                np.array(self.filterX(X)),  y)
        def predict(self, X):
            return super().predict( self.filterX(X) )
        
    return WeakTeam

featuers =   ["DayOfWeek",
            "FlightDate",
            "Reporting_Airline",
            "Tail_Number",
            "Flight_Number_Reporting_Airline",
            "Origin",
            "OriginCityName",
            "OriginState",
            "Dest",
            "DestCityName",
            "DestState",
            "CRSDepTime",
            "CRSArrTime",
            "CRSElapsedTime",
            "Distance"]
            
            # "ArrDelay",
            # "DelayFactor"]



def learn(_dataframe, y):
    agents = 42
    group_size = 4
    subgroups = [ generateTeamClass(team) for team in combinations(featuers, group_size) ]

    def calc_error(model, _dataframe, y):
        _error = 0
        for _bool in model.predict(_dataframe) != y: 
            _error += {  False : 0 , True : 1  }[ _bool ]
        return _error / len( y ) 

    strongGroups = []
    for weak in subgroups:
        _model = AdaBoost(weak, agents)
        _model.train(_dataframe, y)
        strongGroups.append( _model )

    return strongGroups[ np.argmin( [  calc_error( _model, _dataframe, y ) for _model in strongGroups] )] 

if __name__ == "__main__" :
    _dataset = pd.read_csv("~/data/train_data.csv", nrows=100)

    # categorical = ['view' , 'waterfront', 'bedrooms', 'grade', 'floors', 'condition', 'bathrooms']
    # cat = pd.DataFrame ( { 'id' : _dataset_prepoc['id'] } , pd.get_dummies(_dataset_prepoc[categorical].astype('category') ))
    
    
    print(_dataset)
    _dataset.drop( ["FlightDate"] )
    y = []
    for _bool in _dataset["ArrDelay"] > 0 :
        y.append( {  False : 0 , True : 1  }[ _bool ] )
    print(np.array(y))

    _mod = learn(_dataset, np.array(y))














# class StrongClassifer:
    
#     def __init__ (self, features):
#         pass

#     def predict(self, X):
#         pass
    
#     def train(self, X, y):
#         pass

    

