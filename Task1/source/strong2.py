from HUJIHACK.Task1.source.models import abcModel, DecisionTree
from HUJIHACK.Task1.source.adaboost import AdaBoost, AdaBoostList
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from datetime import date
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
import seaborn as sns
import math

from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


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
            DecisionTree.__init__(self, max_depth=3)
            self.featuers = featuers

        def filterX(self, X):
            ret = np.array(pd.DataFrame({featuer: X[featuer] for featuer in self.featuers}))
            return ret

        def train(self, X, y):
            print(f"[@] train on features : {self.featuers}")
            super().fit(
                np.array(self.filterX(X)), y)

        def predict(self, X):
            return super().predict(self.filterX(X))

    return WeakTeam


def pre_proc_new(_dataset, droped_fe, categorical):
    y_del = []
    y_factor = []
    for delay, factor in zip(_dataset["ArrDelay"], _dataset["DelayFactor"]):
        y_del.append(delay)
        y_factor.append(factor)
    y_del = np.array(y_del)
    y_factor = np.array(y_factor)
    for index, row in _dataset.iterrows():
        _dataset.loc[index, 'CRSElapsedTime'] = math.floor(row['CRSElapsedTime'] / 10)
        _dataset.loc[index, 'CRSArrTime'] = math.floor(row['CRSArrTime'] / 100)
        _dataset.loc[index, 'CRSDepTime'] = math.floor(row['CRSDepTime'] / 100)
    cat = pd.DataFrame(pd.get_dummies(_dataset[categorical].astype('category')))
    _dataset = _dataset.drop(droped_fe + categorical, axis=1)
    _dataset_prepoc = pd.concat([_dataset.reset_index(drop=True), cat.reset_index(drop=True)], axis=1)
    return _dataset_prepoc, y_del, y_factor


def ridge_reg(x, y):
    # ridge = Ridge()
    # parameters = {'alpha': [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 1000]}
    # ridge_regressor = GridSearchCV(ridge, parameters, cv=5)
    # ridge_regressor.fit(x, y)
    # b = ridge_regressor.predict(x)
    # print(ridge_regressor.best_params_)
    # print(ridge_regressor.best_score_)
    # a = range(1,10001)
    # plt.scatter(a, y, c='b', s=1, alpha=0.6, label='Training data')
    # plt.scatter(a, b, c='red', s=1, label='prediction')
    # # plt.plot(a, y, 'k', label='True function')
    # plt.legend()
    # plt.show()
    # rr = Ridge(alpha=5)
    rr = LogisticRegression(solver='liblinear')
    rr.fit(x, y)
    pred_train_rr = rr.predict(x)
    print(rr.score(x, y))
    # print(np.sqrt(mean_squared_error(y, pred_train_rr)))
    # print(r2_score(y, pred_train_rr))

    # pred_test_rr = rr.predict(X_test)
    # print(np.sqrt(mean_squared_error(y_test, pred_test_rr)))
    # print(r2_score(y_test, pred_test_rr))


def identify_corelated_features(df):
    corr = df.corr()
    print(sns.heatmap(corr))
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= 0.9:
                if columns[j]:
                    columns[j] = False
    selected_columns = df.columns[columns]
    df = df[selected_columns]
    return df


def learn(_dataframe, y, featuers):
    agents = 3
    group_size = 1
    subgroups = [generateTeamClass(team) for team in combinations(featuers, group_size)]

    def calc_error(model, _dataframe, y):
        _error = 0
        for _bool in (model.predict(_dataframe, max_t=1) != y).flatten():
            _error += {False: 0, True: 1}[_bool]
        return _error / len(y)

    strongGroups = []
    for weak in subgroups:
        _model = AdaBoost(weak, agents)
        _model.train(_dataframe, y)
        strongGroups.append(_model)

    return strongGroups[np.argmin([calc_error(_model, _dataframe, y) for _model in strongGroups])]


if __name__ == "__main__":
    _dataset = pd.read_csv("../data/train_data.csv", nrows=10000)


    # categorical = ['view' , 'waterfront', 'bedrooms', 'grade', 'floors', 'condition', 'bathrooms']
    # cat = pd.DataFrame ( { 'id' : _dataset_prepoc['id'] } , pd.get_dummies(_dataset_prepoc[categorical].astype('category') ))

    def pre_proc(_dataset, droped_fe, categorical):

        def generateY(_dataset):
            y = []
            for _bool in _dataset["ArrDelay"] > 0:
                y.append({False: [0], True: [1]}[_bool])
            return np.array(y)

        # def remove_end_cases(_frame):
        #     return _frame[  _frame['price'] > 0  ]
        y = generateY(_dataset)

        cat = pd.DataFrame(pd.get_dummies(_dataset[categorical].astype('category')))
        _dataset = _dataset.drop(droped_fe + categorical, axis=1)
        _dataset_prepoc = pd.concat([_dataset.reset_index(drop=True), cat.reset_index(drop=True)], axis=1)
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

    droped_fe_new = ['Flight_Number_Reporting_Airline',
                     'Tail_Number',
                     'FlightDate',
                     'ArrDelay',
                     'DelayFactor',
                     "OriginCityName",
                     "OriginState",
                     "DestCityName",
                     "DestState", ]

    categorical_new = ['Reporting_Airline', 'Origin', 'Dest']

    categorical = [
        'OriginCityName'
        , 'OriginState'
        , 'Origin'
        , 'Dest'
        , 'DestCityName'
        , 'DestState'
        , 'Reporting_Airline']

    # "ArrDelay",
    # "DelayFactor"]

    # print("[#] before pre processing")
    # print(_dataset)
    # _dataset, y = pre_proc(_dataset, droped_fe, categorical)
    # print("[#] after pre processing")
    # print(_dataset)
    # print("[#] y' vector ")
    # print(y)
    #
    # print(_dataset.keys())
    # _mod = learn(_dataset, y, _dataset.keys())
    # print("[#] best featuers:")
    # print(_mod.h[0].featuers)
    x, y_del, y_factor = pre_proc_new(_dataset, droped_fe_new, categorical_new)
    print(x)
    x = identify_corelated_features(x)
    print(x)
    # print(x)
    # print(x.shape)
    # print(y_del)
    # print(y_del.shape)
    # print(y_factor)
    # ridge_reg(x, y_del)
    # print(x)
    # print("after corr\n")
    # print(identify_corelated_features(x,y_del))

# class StrongClassifer:

#     def __init__ (self, features):
#         pass

#     def predict(self, X):
#         pass

#     def train(self, X, y):
#         pass
