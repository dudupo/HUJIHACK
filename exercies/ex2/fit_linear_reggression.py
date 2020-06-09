import numpy as np
from copy import deepcopy
import pandas as pd
from plotnine import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def expand_bias(design_matrix):
    _shape = design_matrix.shape
    _ones = np.ones( (_shape[0], _shape[1] + 1) )
    _ones[:,:-1] = design_matrix
    return _ones

def fit_linear_reggression(design_matrix, respone_vec, bias=False):
    if bias:
        design_matrix = expand_bias( design_matrix )
        print(design_matrix)

    u, s, vh = np.linalg.svd(design_matrix)
    s_dagger = deepcopy( s )
    mask = s != 0
    s_dagger[ mask ] = 1 / s [ mask ]
    def diag_mul( _diag, _vec):
        return _diag * _vec[:_diag.shape[0]]
    t =  vh.transpose() @ diag_mul( s_dagger ,u.transpose() @ respone_vec )
    return t, s

def predict( _design_matrix, _coefficients, bias = False):
    if bias:
        _design_matrix = expand_bias( _design_matrix )
    return _design_matrix @ _coefficients

def mse(predicvec, responevec ):
    return np.linalg.norm(responevec - predicvec) ** 2 / responevec.shape[0]

def load_data(_path):

    def remove_end_cases(_frame):
        return _frame[  _frame['price'] > 0  ]

    _dataset_prepoc = remove_end_cases( pd.read_csv(_path) )
    target = 'price'
    prices = _dataset_prepoc[target]
    droped_fe = [ 'date', 'zipcode', 'yr_renovated', 'price', 'lat', 'long']
    categorical = ['view' , 'waterfront', 'bedrooms', 'grade', 'floors', 'condition', 'bathrooms']
    cat = pd.DataFrame ( { 'id' : _dataset_prepoc['id'] } , pd.get_dummies(_dataset_prepoc[categorical].astype('category') ))
    _dataset_prepoc = _dataset_prepoc.drop([target , 'id'] + categorical + droped_fe, axis=1)
    #_dataset_prepoc = pd.merge( _dataset_prepoc , cat, left_on='id', right_on='id')
    return _dataset_prepoc, prices

def plot_singular_values( singulars ):
    singulars.sort()
    plt.scatter( list(range(len(singulars))), singulars[::-1])
    plt.title("singular values by ascending order")
    plt.xlabel("i's singular")
    plt.ylabel('value')
    plt.savefig("plot_singular_values.png")
    plt.show()



def feature_evaluation():
    pass

def q1():

    design_matrix_frame, respone_vec_frame = load_data( "./kc_house_data.csv" )
    design_matrix , respone_vec = design_matrix_frame.to_numpy() , respone_vec_frame.to_numpy()

    respone_vec /= 10**6


    batchsize = int(len(respone_vec)/4)

    indices = np.random.randint(len(respone_vec), size=batchsize)
    W, S = fit_linear_reggression( design_matrix[:][indices], respone_vec[indices], bias=True )
    plot_singular_values( S )



    mask = np.ones(len(respone_vec) , bool)
    mask[indices] = False


    cases_design_matrix = design_matrix[:][mask]
    cases_prices = respone_vec[mask]
    precent = int(len(cases_prices) / 100)
    mses = []

    print(W)

    for i in range(1, 100 ):
        mses.append( mse(predict(design_matrix[:][:i*precent], W , bias=True), cases_prices[:i*precent]))

    print(mses)
    plt.plot( mses )
    plt.title("mse error by chank size")
    plt.xlabel("chank size ( in percentage of the orignal dataset)")
    plt.ylabel('mse error $ [\\times 10^{12}] $ ')
    plt.savefig("q1.png")
    plt.show()
    #respone_vec
    fig, ax = plt.subplots(3, 3)
    for i, _frame in enumerate(design_matrix_frame):
        ax[ int(i /3), i%3 ].scatter( design_matrix_frame[_frame], respone_vec, marker='.'  )
        ax[ int(i /3), i%3 ].set_title( 'price as function of {0}'.format(_frame))
        ax[ int(i /3), i%3 ].set_xlabel(_frame)
        ax[ int(i /3), i%3 ].set_ylabel('price')

        print(  np.cov(  design_matrix_frame[_frame], respone_vec)  /\
         (np.std( design_matrix_frame[_frame]) * np.std(respone_vec)))
    # design_matrix_frame.plot(subplots=True, layout=(4,5))
    plt.show()


if __name__ == '__main__':
    q1()

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
    
