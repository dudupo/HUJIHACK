import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split


# TODO maybe to add feature categorical for countries (Cause it's only in USA).
#  We do know it's important that the distance and the countries themselves
#  affects on delay flight
def load_data(path_csv):
    df = pd.read_csv(path_csv)
    df.dropna(inplace=True)

    # drops unnecessarily columns.

    # Cause we have DestCityName.
    df = df.drop(columns="DestState")
    df = df.drop(columns="Dest")

    # Cause we have OriginCityName
    df = df.drop(columns="OriginState")
    df = df.drop(columns="Origin")

    # Doesn't matter
    df = df.drop(columns="Flight_Number_Reporting_Airline")
    df = df.drop(columns="Tail_Number")
    df = df.drop(columns="FlightDate")

    return df