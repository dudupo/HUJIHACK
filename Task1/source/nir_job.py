import numpy as np
import pandas as pd
from models import *
import time
from strong import pre_proc, categorical, featuers, droped_fe


list_classifiers = [ Logistic, DecisionTree, KNearestNeighbor,
                    RidgeClassifier, LassoClassifier] # LDA,


def calculate_classifier(data):
    data1 = data.as_matrix(columns=["CRSArrTime"])
    data2 = data.as_matrix(columns=["CRSDepTime"])
    data3 = data.as_matrix(columns=["CRSElapsedTime"])
    data4 = data.as_matrix(columns=["Distance"])
    data5 = data.as_matrix(columns=["DayOfWeek"])
    data_cols = [("CRSArrTime", data1), ("CRSDepTime", data2),
                 ("CRSElapsedTime", data3), ("Distance", data4),
                 ("DayOfWeek", data5)]
    y = data.as_matrix(columns=["ArrDelay"])
    dictionary = {"CRSArrTime": [], "CRSDepTime": [], "CRSElapsedTime": [],
                  "Distance": [], "DayOfWeek": []}


    X, y = pre_proc(data ,droped_fe, categorical )
    for col in data_cols:
        for WeekLearner in list_classifiers:
            # X = col[1]
            print(X)
            print(y)
            # X = X.reshape(X.shape[0],)

            classifier = WeekLearner()
            start = time.time()
            classifier.fit(X, y)
            end = time.time()
            time_elapses = end - start
            score = classifier.score(X, y)
            dictionary[col[0]].append((WeekLearner, score, time_elapses))
    return dictionary


if __name__ == '__main__':
    path_csv = "../train_data.csv"
    df = pd.read_csv(path_csv)
    df.dropna(inplace=True)
    calculate_classifier(df)
    print(df.columns)