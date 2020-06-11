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
            ret = np.array( pd.DataFrame( { featuer:  X[featuer] for featuer in self.featuers  }))
            return ret

        def train(self, X, y):
            print(f"[@] train on features : { self.featuers}")
            super().fit(
                np.array(self.filterX(X)),  y)
        def predict(self, X):
            return super().predict( self.filterX(X) )
        
    return WeakTeam





def learn(_dataframe, y, featuers ):
    agents = 3
    group_size = 1
    subgroups = [ generateTeamClass(team) for team in combinations(featuers, group_size) ]

    def calc_error(model, _dataframe, y):
        _error = 0
        for _bool in (model.predict(_dataframe, max_t=1 ) != y).flatten(): 
            _error += {  False : 0 , True : 1  }[ _bool ]
        return _error / len( y ) 

    strongGroups = []
    for weak in subgroups:
        _model = AdaBoost(weak, agents)
        _model.train(_dataframe, y)
        strongGroups.append( _model )

    return strongGroups[ np.argmin( [  calc_error( _model, _dataframe, y ) for _model in strongGroups] )]

def pre_proc(_dataset, droped_fe, categorical ):

    def generateY(_dataset):
        y = []
        for _bool in _dataset["ArrDelay"] > 0 :
            y.append( {  False : [0] , True : [1]  }[ _bool ] )
        return np.array( y )

    # def remove_end_cases(_frame):
    #     return _frame[  _frame['price'] > 0  ]
    y = generateY(_dataset)

    cat = pd.DataFrame(  pd.get_dummies(_dataset[categorical].astype('category') ))
    _dataset = _dataset.drop( droped_fe + categorical, axis=1)
    _dataset_prepoc = pd.concat([ _dataset.reset_index(drop=True), cat.reset_index(drop=True)], axis=1)
    return _dataset_prepoc, y


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

droped_fe = [ 'Flight_Number_Reporting_Airline',
              'Tail_Number',
              'FlightDate',
              'ArrDelay' ,
              'DelayFactor' ]

categorical = [
    'OriginCityName'
    , 'OriginState'
    , 'Origin'
    , 'Dest'
    , 'DestCityName'
    , 'DestState'
    , 'Reporting_Airline']



if __name__ == "__main__" :
    _dataset = pd.read_csv("~/data/train_data.csv", nrows=100)

    # categorical = ['view' , 'waterfront', 'bedrooms', 'grade', 'floors', 'condition', 'bathrooms']
    # cat = pd.DataFrame ( { 'id' : _dataset_prepoc['id'] } , pd.get_dummies(_dataset_prepoc[categorical].astype('category') ))
    
    





            # "ArrDelay",
            # "DelayFactor"]

    print("[#] before pre processing")
    print(_dataset)
    _dataset, y  = pre_proc(_dataset, droped_fe, categorical )
    print("[#] after pre processing")
    print(_dataset)
    print("[#] y' vector ")
    print(y)
    
    print( _dataset.keys())
    _mod = learn(_dataset, y, _dataset.keys() )
    print("[#] best featuers:")
    print(_mod.h[0].featuers)













# class StrongClassifer:
    
#     def __init__ (self, features):
#         pass

#     def predict(self, X):
#         pass
    
#     def train(self, X, y):
#         pass

    

