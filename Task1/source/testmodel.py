

from model import FlightPredictor
import pandas as pd

if __name__ == "__main__":
    print("[@] create FlightPredictor")
    fp = FlightPredictor( )
    print("[@] predict")
    ret = fp.predict( pd.load("~/data/train_data.csv") ,  nrows=10000 )
    print(ret)