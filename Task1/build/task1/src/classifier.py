# from strong import *
from models import abcModel, DecisionTree, Logistic, SVM, DecisionStumpWarper
from adaboost import AdaBoost, AdaBoostList
from itertools import combinations
import numpy as np
import pandas as pd
from binarysearch import binarysearch
from datetime import date
from random import shuffle
from sklearn.model_selection import train_test_split
# from geopy.geocoders import Nominatim, ArcGIS
import json
import math
factor_reason = {'CarrierDelay': 0, 'WeatherDelay': 1, 'NASDelay': 2, 'LateAircraftDelay': 3}

#
# def pre_proc_class_old(_dataset, droped_fe, categorical):
#     y_factor = []
#     _dataset = _dataset.dropna()
#     _dataset['month'] = pd.DatetimeIndex(_dataset['FlightDate']).month
#     _dataset['day'] = pd.DatetimeIndex(_dataset['FlightDate']).day
#     _dataset['year'] = pd.DatetimeIndex(_dataset['FlightDate']).year
#
#     for factor in _dataset["DelayFactor"]:
#         y_factor.append(factor_reason[factor])
#     y_factor = np.array(y_factor)
#     for index, row in _dataset.iterrows():
#         # _dataset.loc[index, 'CRSElapsedTime'] = math.floor(row['CRSElapsedTime'] / 10)
#         _dataset.loc[index, 'CRSArrTime'] = math.floor(row['CRSArrTime'] / 100)
#         _dataset.loc[index, 'CRSDepTime'] = math.floor(row['CRSDepTime'] / 100)
#         _dataset.loc[index, 'Distance'] = math.floor(row['Distance'] / 10)
#
#     cat = pd.DataFrame(pd.get_dummies(_dataset[categorical].astype('category')))
#     _dataset = _dataset.drop(droped_fe + categorical, axis=1)
#     _dataset_prepoc = pd.concat([_dataset.reset_index(drop=True), cat.reset_index(drop=True)], axis=1)
#     return _dataset_prepoc, y_factor



def proc_weather( _dataset ):
    weather = json.loads( open("wether.json").read().replace("'", "\"") )
    for index, row in _dataset.iterrows():
        _date = f"{row['day']:02d}-{row['month']:02d}-{row['year']%2000}" 
        if _date in weather:
            for key, val in weather[ _date ].items():
                if key != 'day':
                    row[key] = val



def pre_proc_class(_dataset, categorical):


    _dataset['month'] = pd.DatetimeIndex(_dataset['FlightDate']).month
    _dataset['day'] = pd.DatetimeIndex(_dataset['FlightDate']).day
    _dataset['year'] = pd.DatetimeIndex(_dataset['FlightDate']).year
        
    #proc_weather( _dataset )

    for index, row in _dataset.iterrows():
        # _dataset.loc[index, 'CRSElapsedTime'] = math.floor(row['CRSElapsedTime'] / 10)
        _dataset.loc[index, 'CRSArrTime'] = math.floor(row['CRSArrTime'] / 100)
        _dataset.loc[index, 'CRSDepTime'] = math.floor(row['CRSDepTime'] / 100)
        _dataset.loc[index, 'Distance'] = math.floor(row['Distance'] / 10)


    # geolocator = ArcGIS(username="david.ponarovsky", password="mxSrWYYdSq++J7+",
    #  referer="efMmfQrfh1o_Ag-MEzx5-en9lLs-m_Vu5T_JU7K45vfUhjEccY6W2cilzPnn2r9TO7fpPnAeK_U8EWptaUiXOC7FosOKaovLPAXQR06PGnXrUYBRiN6RMJA8g6JoANpO5LM080ti3FTpMVgDsJ3Psg..",
    #  timeout=500000)

    def DynmicGeo( df, _keys ):
        import time
        from pprint import pprint

        #Dynmic = {} 
        Dynmic = json.loads( open("geo").read().replace("'", "\"") )
        try :
            for _key in _keys:
                df[f'lat{_key}']=0
                df[f'long{_key}']=0
                for x in range(len(df)):
                    if df[_key][x] in Dynmic:
                        location = Dynmic[df[_key][x]]
                        if type(location) == str:
                            df.at[x, f'lat{_key}'] , df.at[x, f'long{_key}'] = eval(location)
                        else:
                            df.at[x, f'lat{_key}'] , df.at[x, f'long{_key}'] = location
                    # else:
                    #     location = geolocator.geocode(df[_key][x])
                    #     Dynmic[df[_key][x]] = location.latitude,  location.longitude
                    #     # time.sleep(2)
                    # #print(location , location.latitude, location.longitude)
                    #     df.at[x, f'lat{_key}']= location.latitude
                    #     df.at[x, f'long{_key}'] = location.longitude
        except:
            print("except")
        
        pprint(Dynmic)

        for key, val in Dynmic.items():     
            Dynmic[key] = str(val)
        with open('geo', 'w') as f:
            json.dump(Dynmic, f)
        
        return df    


    geoKeys = [ "Origin",
            "OriginCityName",
            "OriginState",
            "Dest",
            "DestCityName",
            "DestState"]

    _dataset = DynmicGeo(_dataset, geoKeys)
   
    if len(categorical) >0 :
        cat = pd.DataFrame(pd.get_dummies(_dataset[categorical].astype('category')))
        _dataset_prepoc = pd.concat([_dataset.reset_index(drop=True), cat.reset_index(drop=True)], axis=1)
    else:
        _dataset_prepoc = _dataset
   
    return _dataset_prepoc


def pre_proc_delay(dataset, drop_fe_delay, categorical):
    y = []
    for _bool in dataset["ArrDelay"] > 0:
        y.append({False: [0], True: [1]}[_bool])
    y = np.array(y)
    dataset = dataset.drop(drop_fe_delay + categorical, axis=1)
    return dataset, y


def pre_proc_factor(dataset, drop_fe_factor, categorical):
    y_factor = []
    dataset = dataset.dropna()
    for factor in dataset["DelayFactor"]:
        y_factor.append(factor_reason[factor])
    y_factor = np.array(y_factor)
    dataset = dataset.drop(drop_fe_factor + categorical, axis=1)
    return dataset, y_factor


def pred_trees(x, y, x_test, y_test):
    from sklearn import datasets
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split

    # dividing X, y into train and test data

    # training a DescisionTreeClassifier
    from sklearn.tree import DecisionTreeClassifier
    for depth in [4, 5, 7, 10, 13, 15, 17, 18, 19, 20]:
        dtree_model = DecisionTreeClassifier(max_depth=depth).fit(x, y)
        dtree_predictions = dtree_model.predict(x_test)
        print(dtree_model.score(x_test, y_test))
        print(depth)


def pred_forest(x, y, x_test, y_test):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(random_state=20)
    model.fit(x, y)
    print(model.score(x_test, y_test))


def identify_corelated_features(df):
    # corr = df.corr()
    # #print(sns.heatmap(corr))
    # columns = np.full((corr.shape[0],), True, dtype=bool)
    # for i in range(corr.shape[0]):
    #     for j in range(i + 1, corr.shape[0]):
    #         if corr.iloc[i, j] >= 0.9:
    #             if columns[j]:
    #                 columns[j] = False
    # selected_columns = df.columns[columns]
    # df = df[selected_columns]
    return df


def pred(x, y, x_test, y_test):
    from sklearn import datasets
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split

    # training a KNN classifier
    from sklearn.neighbors import KNeighborsClassifier
    for n in [11, 13, 15, 20, 30, 35, 40, 45]:
        knn = KNeighborsClassifier(n_neighbors=n).fit(x, y)

        # accuracy on X_test
        accuracy = knn.score(x_test, y_test)
        print(accuracy)
        print(n)

    # creating a confusion matrix
    # knn_predictions = knn.predict(x)
    # cm = confusion_matrix(y_test, knn_predictions)


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

droped_fe_factor = ['Flight_Number_Reporting_Airline',
                    'Tail_Number',
                    'DelayFactor',
                    "OriginCityName",
                    "OriginState",
                    "DestCityName",
                    "DestState", 'FlightDate', "CRSElapsedTime"]

droped_fe_delay = ['Flight_Number_Reporting_Airline',
                   'Tail_Number',
                   "OriginCityName",
                   "OriginState",
                   "DestCityName",
                   "DestState", 'FlightDate', "CRSElapsedTime", 'ArrDelay', 'DelayFactor', 'Origin', 'Dest', 'Reporting_Airline' ]

categorical_new = [] #['Reporting_Airline']



def final_pre_proc_test(_dataset):
    return pre_proc_class(_dataset, categorical_new)

def final_pre_proc(_dataset):
    x = pre_proc_class(_dataset, categorical_new)
    x_delay,y_delay = pre_proc_delay(x,droped_fe_delay,categorical_new)
    x_factor,y_factor = pre_proc_delay(x,droped_fe_factor,categorical_new)

    x_factor = identify_corelated_features(x_factor)
    return x_delay, y_delay, x_factor,y_factor

if __name__ == '__main__':
    _dataset = pd.read_csv("~/data/train_data.csv")

    x = pre_proc_class(_dataset, categorical_new)
    
    print(x)
    
    x_delay,y_delay = pre_proc_delay(x,droped_fe_delay,categorical_new)
    x_factor,y_factor = pre_proc_delay(x,droped_fe_factor,categorical_new)

    x_factor = identify_corelated_features(x_factor)
    x_train, x_test, y_train, y_test = train_test_split(x_factor, y_factor)
    # print(x['month'])

    # print(y_factor)
    # pred_trees(x_train, y_train, x_test, y_test)
    from datetime import datetime

    # datetime object containing current date and time
    print(datetime.now())
    print(x_test.shape)
    # pred_forest(x_train, y_train, x_test, y_test)
    print(datetime.now())
    # pred(x_train, y_train, x_test, y_test)
    # print(identify_corelated_features(x, y_del))
