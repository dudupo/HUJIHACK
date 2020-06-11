from models import abcModel, DecisionTree, Logistic, SVM, DecisionStumpWarper
from adaboost import AdaBoost, AdaBoostList
from itertools import combinations
import numpy as np
import pandas as pd
from binarysearch import binarysearch
from datetime import date
from random import shuffle
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
    def generateWeakClass(Type):
        class WeakTeam(Type):

            def __init__(self):
                Type.__init__(self)
                self.featuers = featuers

            def filterX( self, X ):
                ret = np.array( pd.DataFrame( { featuer:  X[featuer] for featuer in self.featuers  }))
                return ret

            def train(self, X, y, D=None):
                print(f"[@] train on features : { self.featuers}")
                
                if D is None:
                    super().fit(
                        np.array(self.filterX(X)),  y)
                else:
                    super().fit(
                        np.array(self.filterX(X)), y, D)
            
            def predict(self, X):
                return super().predict( self.filterX(X) )
        
        return WeakTeam
    #return generateWeakClass( DecisionTree ) 
    # return generateWeakClass( Logistic )

    return generateWeakClass(DecisionStumpWarper)


def calc_error(model, _dataframe, y, agents):
    _error = 0
    z = (y.flatten() - 0.5 * np.ones(len(y)))*2
    for _bool in (model.predict(_dataframe, max_t=agents ) != z ): 
        _error += {  False : 0 , True : 1  }[ _bool ]
    return _error / len( y ) 


def learn(_dataframe, y, featuers ):
    agents = 50
    group_size = 2
    subgroups = [ generateTeamClass(team) for team in combinations(featuers, group_size) ]

    strongGroups = []
    for weak in subgroups:
        _model = AdaBoost(weak, agents, support_wights=True)
        _model.train(_dataframe, y)
        strongGroups.append( _model )
    
    return strongGroups[ np.argmin(
         [  calc_error( _model, _dataframe, y, agents ) for _model in strongGroups] )]


def generateY(_dataset, treshold=0):
        y = []
        for _bool in _dataset["ArrDelay"] > treshold:
            y.append( {  False : [0] , True : [1]  }[ _bool ] )
        return np.array( y )

def pre_proc(_dataset, droped_fe, categorical ):

    # def remove_end_cases(_frame):
    #     return _frame[  _frame['price'] > 0  ]
    y = generateY(_dataset)

    if len(categorical) > 0 :
        cat = pd.DataFrame(  pd.get_dummies(_dataset[categorical].astype('category') ))


    _dataset = _dataset.drop( droped_fe + categorical, axis=1)
    if len(categorical) > 0 :
        _dataset_prepoc = pd.concat([ _dataset.reset_index(drop=True), cat.reset_index(drop=True)], axis=1)
    else :
        _dataset_prepoc = _dataset 
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
              'DayOfWeek',
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
    original_dataset = pd.read_csv("~/data/train_data.csv", nrows=100)

    print("[#] before pre processing")
    print(original_dataset)
    # _dataset, y  = pre_proc(_dataset, droped_fe, categorical )
    print("[#] after pre processing")
    print("[#] y' vector ")
    
    # _mods = learn(_dataset, y, _dataset.keys() )
    # print("[#] best featuers:")
    # print(_mod.h[0].featuers)

    '''
        just to check compiletion.  
    '''
    _mods = {}  
    _maxrange = 30
    _dataset, y = pre_proc(original_dataset, droped_fe + categorical, [] )
    start_range , end_range = np.zeros(len(y)) , np.ones(len(y)) * _maxrange
    for i, time in enumerate( np.arange(0, _maxrange,  _maxrange/ 2**5 ) ):
        _mods[time] = learn(_dataset,
                        generateY(original_dataset, time),
                            _dataset.keys() )  

    # print( i, time)
    #times_tersholds = { time : _mods[i] for i, time in enumerate( range(0, 1, 2**-4) ) }

    Bagent = binarysearch( _mods ) 
    _middles = Bagent.predict( _dataset , start_range , end_range)
    print(_middles)


    original_dataset = pd.read_csv("~/data/train_data.csv", nrows=10000 )[100:]
    _dataset, y = pre_proc(original_dataset, droped_fe + categorical, [] )
    _middles = Bagent.mods[0.0].predict( _dataset )
    _middles[ _middles > 0 ] = 1
    t = sum( (_middles- y.flatten())**2/len(y))
    print (  f"[error]: {t}"  )    
    #print( )













# class StrongClassifer:
    
#     def __init__ (self, features):
#         pass

#     def predict(self, X):
#         pass
    
#     def train(self, X, y):
#         pass

    

