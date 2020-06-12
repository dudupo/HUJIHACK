import json 
import pandas as pd
import pickle 


if __name__ == "__main__" :
    df = pd.read_csv("~/data/all_weather_data.csv")
    _json = { }
    for index, row in df.iterrows():
        for key in df.keys():
            _json[ row['day'] ] = {}
            if key != "day":
                _json[ row['day'] ][key] = row[key]  
    json.dump( _json ,  open("./wether.json", "w") )

    