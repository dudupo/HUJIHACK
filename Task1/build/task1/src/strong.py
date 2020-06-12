from models import abcModel, DecisionTree, Logistic, SVM, DecisionStumpWarper
from adaboost import AdaBoost, AdaBoostList
from itertools import combinations
import numpy as np
import pandas as pd
from binarysearch import binarysearch
from datetime import date
from random import shuffle
import pickle

from classifier import final_pre_proc, final_pre_proc_test
from sklearn.model_selection import train_test_split


class WeakTeam(DecisionStumpWarper):

    def __init__(self, featuers=True):
        DecisionStumpWarper.__init__(self)
        self.featuers = featuers

    def filterX( self, X ):
        ret = np.array( pd.DataFrame( { featuer:  X[featuer] for featuer in self.featuers }))
        return ret

    def train(self, X, y, D=None):
        #print(f"[@] train on features : { self.featuers}")
        
        if D is None:
            super().fit(
                np.array(self.filterX(X)),  y)
        else:
            super().fit(
                np.array(self.filterX(X)), y, D)
    
    def predict(self, X):
        c = X.select_dtypes(np.number).columns
        X[c] = X[c].fillna(0)
        return super().predict( self.filterX(X) )

def generateTeamClass(featuers):
    return (WeakTeam, featuers)


def calc_error(model, _dataframe, y, agents):
    _error = 0
    z = (y.flatten() - 0.5 * np.ones(len(y))) * 2
    for _bool in (model.predict(_dataframe, max_t=agents) != z):
        _error += {False: 0, True: 1}[_bool]
    return _error / len(y)


import heapq


def hash_strings(featuers, _list):
    ret = 0
    base = 1
    for featuer in featuers:
        if featuer in _list:
            ret += base
        base *= 10
    return ret


def learn(_dataframe, y, featuers, teams=set(), depth=4, orignal=[],
          _hased=set(),):
    if len(teams) == 0:
        teams = [[featuer] for featuer in featuers]

    agents = 12
    _hashed = set()
    new_team = []

    def create_subgroups():
        subgroups = []
        for featuer in featuers:
            for team in teams:
                if featuer not in team:
                    _hash = hash_strings(orignal, team + [featuer])
                    if _hash not in _hashed:
                        subgroups.append(generateTeamClass(team + [featuer]))
                        _hashed.add(_hash)
        return subgroups

    strongGroups = []
    heap = []

    for weak in create_subgroups():
        _model = AdaBoost(weak, agents, support_wights=True)
        _model.train(_dataframe, y)
        strongGroups.append(_model)
        heapq.heappush(heap, (
            -calc_error(_model, _dataframe, y, agents), _model.h[0].featuers,
            _model))


        if len(heap) > 3:
            heapq.heappop( heap )

    if depth == 0:
        while len(heap) > 1:
            train_error, _featuers, _model =  heapq.heappop( heap )
        train_error, _featuers, _model =  heapq.heappop( heap )
        return train_error, _featuers, _model

    else:
        teams = []
        while len(heap) > 1:

            heapq.heappop( heap )
            train_error, _featuers, _model =  heapq.heappop( heap )
            teams.append( _featuers  )
        return learn(_dataframe, y, featuers, teams, depth - 1,
                     orignal=orignal)


def generateY(_dataset, treshold=0):
    y = []
    for _bool in _dataset["ArrDelay"] > treshold:
        y.append({False: [0], True: [1]}[_bool])
    return np.array(y)


def generateYbyString(_dataset, s, treshold=0):
    y = []
    for _bool in _dataset[s] > treshold:
        y.append({False: [0], True: [1]}[_bool])
    return np.array(y)


def pairs():
    df1 = df_copy.loc[np.logical_or(df_copy["DelayFactor"] == 0,
                                    df_copy["DelayFactor"] == 1)]
    df0 = df1.drop(columns=["DelayFactor"])
    _, _, classifier1 = learn(df0, df1["DelayFactor"].as_matrix(), df0.keys(),
                              orignal=df0.keys())

    df2 = df_copy.loc[np.logical_or(df_copy["DelayFactor"] == 1,
                                    df_copy["DelayFactor"] == 2)]
    df0 = df2.drop(columns=["DelayFactor"])
    _, _, classifier2 = learn(df0, df2["DelayFactor"].as_matrix() - 1,
                              df0.keys(),
                              orignal=df0.keys())
    print(df2["DelayFactor"].as_matrix() - 1)

    df3 = df_copy.loc[np.logical_or(df_copy["DelayFactor"] == 2,
                                    df_copy["DelayFactor"] == 3)]
    df0 = df3.drop(columns=["DelayFactor"])
    _, _, classifier3 = learn(df0, df3["DelayFactor"].as_matrix() - 2,
                              df0.keys(),
                              orignal=df0.keys())

    df4 = df_copy.loc[np.logical_or(df_copy["DelayFactor"] == 3,
                                    df_copy["DelayFactor"] == 1)]
    df0 = df4.drop(columns=["DelayFactor"])
    s = df4["DelayFactor"].as_matrix()
    s[s == 3] = 0
    df4 = df4.drop(columns=["DelayFactor"])
    df4["DelayFactor"] = s
    _, _, classifier4 = learn(df0, df4["DelayFactor"].as_matrix(), df0.keys(),
                              orignal=df0.keys())

    df5 = df_copy.loc[np.logical_or(df_copy["DelayFactor"] == 2,
                                    df_copy["DelayFactor"] == 0)]
    df0 = df5.drop(columns=["DelayFactor"])
    s = df5["DelayFactor"].as_matrix()
    s[s == 2] = 1
    df5 = df5.drop(columns=["DelayFactor"])
    df5["DelayFactor"] = s
    _, _, classifier5 = learn(df0, df5["DelayFactor"].as_matrix(), df0.keys(),
                              orignal=df0.keys())

    df6 = df_copy.loc[np.logical_or(df_copy["DelayFactor"] == 3,
                                    df_copy["DelayFactor"] == 0)]
    df0 = df6.drop(columns=["DelayFactor"])
    s = df6["DelayFactor"].as_matrix()
    s[s == 3] = 1
    df6 = df6.drop(columns=["DelayFactor"])
    df6["DelayFactor"] = s
    _, _, classifier6 = learn(df0, df6["DelayFactor"].as_matrix(), df0.keys(),
                              orignal=df0.keys())
    return [(classifier1, df1.index), (classifier2, df2.index),
            (classifier3, df3.index), (classifier4, df4.index),
            (classifier5, df5.index), (classifier6, df6.index)]


def predict_pairs(X, list_classifiers):

    def convrt( yy):
        return (yy + np.ones(len(yy))) /2

    y0 = convrt(list_classifiers[0][0].predict(X))
    y1 = convrt(list_classifiers[1][0].predict(X))
    y1[y1 == 0] = 1
    y1[y1 == 1] = 2
    y2 = convrt(list_classifiers[2][0].predict(X))
    y2[y2 == 0] = 2
    y2[y2 == 1] = 3
    y3 = convrt(list_classifiers[3][0].predict(X))
    y3[y3 == 0] = 3
    y4 = convrt(list_classifiers[4][0].predict(X))
    y4[y4 == 1] = 2
    y5 = convrt(list_classifiers[5][0].predict(X))
    y5[y5 == 1] = 3

    final_result = np.concatenate((y0, y1, y2, y3, y4, y5), axis=0).reshape(
        (6, y0.shape[0])).T
    counts = np.apply_along_axis(np.bincount, 1, final_result.astype(np.int))
    final_result = np.apply_along_axis(np.argmax, 1, counts).astype(np.str)
    final_result[final_result == '0'] = "CarrierDelay"
    final_result[final_result == '1'] = "WeatherDelay"
    final_result[final_result == '2'] = "NASDelay"
    final_result[final_result == '3'] = "LateAircraftDelay"
    return final_result


def pre_proc(_dataset, droped_fe, categorical ):
    y = generateY(_dataset)

    if len(categorical) > 0:
        cat = pd.DataFrame(
            pd.get_dummies(_dataset[categorical].astype('category')))

    _dataset = _dataset.drop(droped_fe + categorical, axis=1)
    if len(categorical) > 0:
        _dataset_prepoc = pd.concat(
            [_dataset.reset_index(drop=True), cat.reset_index(drop=True)],
            axis=1)
    else:
        _dataset_prepoc = _dataset

    print(_dataset_prepoc)
    return _dataset_prepoc, y


featuers = ["DayOfWeek",
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

droped_fe = ['Flight_Number_Reporting_Airline',
             'Tail_Number',
             'FlightDate',
             'ArrDelay',
             'DelayFactor']

categorical = [
    'OriginCityName'
    , 'OriginState'
    , 'Origin'
    , 'Dest'
    , 'DestCityName'
    , 'DestState'
    , 'Reporting_Airline']



if __name__ == "__main__" :
    original_dataset = pd.read_csv("~/data/train_data.csv", nrows=200)

    print("[#] before pre processing")
    print(original_dataset)
    _mods = {}  
    _minrange, _maxrange = -30, 30
    _dataset, y, x_factor,y_factor = final_pre_proc(original_dataset)

    print( _dataset)

    start_range , end_range = np.ones(len(y)) * _minrange , np.ones(len(y)) * _maxrange
    for i, time in enumerate( np.arange(_minrange, _maxrange,  (_maxrange-_minrange)/ 32 ) ):
        train_error, _featuers, _mods[time] = learn(_dataset,
                                                    generateY(original_dataset,
                                                              time),
                                                    _dataset.keys(),  orignal=_dataset.keys())
        print(f"{time} : {_featuers} : {train_error}")

    Bagent = binarysearch( _mods ) 

    with open("./BinAgent", "wb") as f:
        pickle.dump(Bagent, f)




# class StrongClassifer:

#     def __init__ (self, features):
#         pass

#     def predict(self, X):
#         pass

#     def train(self, X, y):
#         pass
