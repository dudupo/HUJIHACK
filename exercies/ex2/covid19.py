
from fit_linear_reggression import fit_linear_reggression


import numpy as np
from copy import deepcopy
import pandas as pd
from plotnine import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":
        _path = "covid19_israel.csv"
        dataset_covid = pd.read_csv(_path)
        dataset_covid["log_detected"] = np.log( dataset_covid["detected"])
        X, Y = dataset_covid["day_num"], dataset_covid["log_detected"]
        X = np.array([X])
        W, S = fit_linear_reggression( X.transpose(), Y )
        gplot = ggplot(dataset_covid, aes(x='day_num' , y='log_detected')) +\
        geom_point()+ geom_abline(slope =W[0]) + ggtitle('log detected as function of the dayes')

        ggsave(gplot, "log_fig.png")
        X, Y = dataset_covid["day_num"], dataset_covid["detected"]
        Z = np.e ** (W[0]*X)

        gplot = ggplot( dataset_covid , aes(x='day_num' , y='detected')) +\
        geom_point()+\
        geom_line(aes(y='Z')) + geom_abline(slope =W[0]) + ggtitle('detected as function of the dayes')
        ggsave(gplot, "fig.png")
