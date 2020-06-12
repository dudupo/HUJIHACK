from models import abcModel, DecisionTree, Logistic, SVM, DecisionStumpWarper
from adaboost import AdaBoost, AdaBoostList
from itertools import combinations
import numpy as np
import pandas as pd
from binarysearch import binarysearch
from datetime import date
from random import shuffle

from classifier import final_pre_proc
from sklearn.model_selection import train_test_split


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

            def filterX(self, X):
                ret = np.array(pd.DataFrame(
                    {featuer: X[featuer] for featuer in self.featuers}))
                return ret

            def train(self, X, y, D=None):
                # print(f"[@] train on features : { self.featuers}")

                if D is None:
                    super().fit(
                        np.array(self.filterX(X)), y)
                else:
                    super().fit(
                        np.array(self.filterX(X)), y, D)

            def predict(self, X):
                return super().predict(self.filterX(X))

        return WeakTeam

    # return generateWeakClass( DecisionTree )
    # return generateWeakClass( Logistic )

    return generateWeakClass(DecisionStumpWarper)


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


def learn(_dataframe, y, featuers, teams=set(), depth=5, orignal=[],
          _hased=set()):
    if len(teams) == 0:
        teams = [[featuer] for featuer in featuers]

    agents = 42
    group_size = 2
    _hashed = set()
    new_team = []

    def create_subgroups():
        subgroups = []
        for featuer in featuers:
            for team in teams:
                if featuer not in team:
                    _hash = hash_strings(orignal, team + [featuer])
                    if _hash not in _hashed:
                        # print(f"i was here {team + [featuer]}, _hash : {_hash}")
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

        if len(heap) > 30000:
            heapq.heappop(heap)

    if depth == 0:
        while len(heap) > 1:
            train_error, _featuers, _model = heapq.heappop(heap)
            # print(train_error)
        train_error, _featuers, _model = heapq.heappop(heap)
        return train_error, _featuers, _model

    else:
        teams = []
        while len(heap) > 1:
            heapq.heappop(heap)
            train_error, _featuers, _model = heapq.heappop(heap)
            teams.append(_featuers)
        # print(teams)
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
    df2 = df1.drop(columns=["DelayFactor"])
    learn(df2, df1["DelayFactor"].as_matrix(), df2.keys(), orignal=df2.keys())

    df1 = df_copy.loc[np.logical_or(df_copy["DelayFactor"] == 1,
                                    df_copy["DelayFactor"] == 2)]
    df2 = df1.drop(columns=["DelayFactor"])
    learn(df2, df1["DelayFactor"].as_matrix() - 1, df2.keys(),
          orignal=df2.keys())

    df1 = df_copy.loc[np.logical_or(df_copy["DelayFactor"] == 2,
                                    df_copy["DelayFactor"] == 3)]
    df2 = df1.drop(columns=["DelayFactor"])
    learn(df2, df1["DelayFactor"].as_matrix() - 2, df2.keys(),
          orignal=df2.keys())

    df1 = df_copy.loc[np.logical_or(df_copy["DelayFactor"] == 3,
                                    df_copy["DelayFactor"] == 1)]
    df2 = df1.drop(columns=["DelayFactor"])
    s = df1["DelayFactor"].as_matrix()
    s[s == 3] = 0
    df1 = df1.drop(columns=["DelayFactor"])
    df1["DelayFactor"] = s
    learn(df2, df1["DelayFactor"].as_matrix(), df2.keys(),
          orignal=df2.keys())


    df1 = df_copy.loc[np.logical_or(df_copy["DelayFactor"] == 2,
                                    df_copy["DelayFactor"] == 0)]
    df2 = df1.drop(columns=["DelayFactor"])
    s = df1["DelayFactor"].as_matrix()
    s[s == 2] = 1
    df1 = df1.drop(columns=["DelayFactor"])
    df1["DelayFactor"] = s
    learn(df2, df1["DelayFactor"].as_matrix(), df2.keys(),
          orignal=df2.keys())


    df1 = df_copy.loc[np.logical_or(df_copy["DelayFactor"] == 3,
                                    df_copy["DelayFactor"] == 0)]
    df2 = df1.drop(columns=["DelayFactor"])
    s = df1["DelayFactor"].as_matrix()
    s[s == 3] = 1
    df1 = df1.drop(columns=["DelayFactor"])
    df1["DelayFactor"] = s
    learn(df2, df1["DelayFactor"].as_matrix(), df2.keys(),
          orignal=df2.keys())

    return


def pre_proc(_dataset, droped_fe, categorical):
    # def remove_end_cases(_frame):
    #     return _frame[  _frame['price'] > 0  ]
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
             #   'DayOfWeek',
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

if __name__ == "__main__":
    original_dataset = pd.read_csv("~/data/train_data.csv", nrows=7000)

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
    _minrange, _maxrange = -30, 30
    # _dataset, y = pre_proc(original_dataset, droped_fe , categorical )
    _dataset, y, x_factor, y_factor = final_pre_proc(original_dataset)

    print(_dataset)
    print(y)

    print("-------------------------------------------------------")
    print(y)

    df_copy = _dataset.copy()
    df_copy["DelayFactor"] = y_factor
    pairs()

    start_range, end_range = np.ones(len(y)) * _minrange, np.ones(
        len(y)) * _maxrange
    for i, time in enumerate(
            np.arange(_minrange, _maxrange, _maxrange / 2 ** 5)):
        train_error, _featuers, _mods[time] = learn(_dataset,
                                                    generateY(original_dataset,
                                                              time),
                                                    _dataset.keys(), teams=[
                ['CRSDepTime', 'DayOfWeek']], orignal=_dataset.keys())
        print(f"{time} : {_featuers} : {train_error}")

    # print( i, time)
    # times_tersholds = { time : _mods[i] for i, time in enumerate( range(0, 1, 2**-4) ) }

    Bagent = binarysearch(_mods)
    _middles = Bagent.predict(_dataset, start_range, end_range)
    print(_middles)

    _dataset, y, x_factor, y_factor = final_pre_proc(
        pd.read_csv("~/data/train_data.csv", nrows=10000)[9800:])
    # _dataset, y = pre_proc(original_dataset, droped_fe , categorical )
    _middles = Bagent.mods[0.0].predict(_dataset)
    _middles[_middles > 0] = 1
    t = sum((_middles - y.flatten()) ** 2 / len(y))
    print(f"[error]: {t}")
    # print( )

# class StrongClassifer:

#     def __init__ (self, features):
#         pass

#     def predict(self, X):
#         pass

#     def train(self, X, y):
#         pass
