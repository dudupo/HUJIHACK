

from model import FlightPredictor
import pandas as pd
from classifier import final_pre_proc
from strong  import WeakTeam 
from binarysearch import binarysearch
from adaboost import AdaBoost 
from ex4_tools import DecisionStump

if __name__ == "__main__":
    print("[@] create FlightPredictor")
    fp = FlightPredictor( )
    print("[@] predict")
    ret = fp.predict( pd.read_csv("~/data/train_data.csv", nrows=10000)  )
    print(ret)