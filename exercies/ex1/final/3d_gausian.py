import numpy as np
import pandas as pd
from plotnine import *
import matplotlib.pyplot as plt
from matplotlib import rc

from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr

# rc('text', usetex=True)
mean = [0, 0, 0]
cov = np.eye(3)
x_y_z = np.random.multivariate_normal(mean, cov, 50000).T


def get_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = qr(H)
    return Q


def plot_3d(x_y_z):
    '''
    plot points in 3D
    :param x_y_z: the points. numpy array with shape: 3 X num_samples (first dimension for x, y, z
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_y_z[0], x_y_z[1], x_y_z[2], s=1, marker='.', depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def plot_2d(x_y):
    '''
    plot points in 2D
    :param x_y_z: the points. numpy array with shape: 2 X num_samples (first dimension for x, y
    coordinate)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_y[0], x_y[1], s=1, marker='.')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')



def section11():

    plot_3d(x_y_z)
    plt.title("Normal distribution $ \sigma(x) = \sigma(y) = \sigma(z)= 1 $ and $ \mu = 0 $")
    plt.savefig("sec11.png")

def section12():
    S = np.diag([0.1 , 0.5, 2])
    S_transform = S.dot( x_y_z )
    plot_3d( S_transform )
    plt.title("transformed Normal distribution $ S \mathcal{N} $")
    plt.savefig("sec12.png")


    S_trandform_cov = np.cov(S_transform)
    print( S_trandform_cov )


def section13():
    S = np.diag([0.1 , 0.5, 2])
    S_transform = S.dot( x_y_z )


    rndmatrix = get_orthogonal_matrix( 3 )

    S_trans_mul_by_orth =\
        rndmatrix.dot( S_transform )

    plot_3d( rndmatrix.dot( x_y_z ) )
    plt.title("Normal distribution multiplied by random Diagonal $ diag \cdot  \mathcal{N} $")
    plt.savefig("sec13mulbyrand_orth.png")


    plot_3d( S_trans_mul_by_orth )
    plt.title("transformed Normal distribution multiplied by random Diagonal $ diag \cdot  S \mathcal{N} $")
    plt.savefig("sec13mulbyrand_orth_S.png")


def plot_xy_proj_histo(x_y_z, sec, _title):
    #ploting 2d proj
    plot_2d(x_y_z[:-1])
    plt.savefig("{0}proj.png".format(sec))
    # iterating over the x any y axes.
    for X, _dim in zip(x_y_z[:-1] , ["x" , "y"]):
        # calculate histogram
        fig = plt.figure()
        x , y  = np.histogram( X, bins = 600 )
        plt.plot( y[:-1], x,
         label="histogram of {0} dim".format(_dim) )
        plt.title(_title.format(_dim))
        plt.xlabel(_dim)
        plt.ylabel('points')
        plt.savefig("{0}proj_hist_{1}.png".format(sec,_dim))

def section14():

    plot_xy_proj_histo(x_y_z, "section14", "Histogram of projection Normal distribution  $ proj( data , {0}) $")



def section15():

    query =  (x_y_z[2] >= -0.4) & (x_y_z[2] <= 0.1)
    filtred = np.array( [ x_y_z[_][query] for _ in range(3) ]
     ,dtype=np.float64)
    print(filtred)
    plot_xy_proj_histo( filtred, "section15", "Histogram of projection Normal distribution for $ z \in (-0.4,0.1) proj( data , {0}) $")


def section16():
    length  = 10**5
    samples = 1000
    data = np.random.binomial(1, 0.25, (samples, length))
    epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]


    plt.figure()

    def plot_estimate_function_as_m(row , i):
        '''
            ploting the arithmetic mean of given sample.
        '''
        X, Y, _accsum = [0], [0], 0
        for coin, value in enumerate(row):
            if coin > 0 :
                _accsum += value
                X.append( coin )
                Y.append( _accsum / coin  )

        plt.scatter(X, Y, s=1, marker='.', label="{0}`s sample".format(i) )


    # plt.title("{0}'s dataset'".format(i))
    plt.xlabel("flips")
    plt.ylabel("aritmetic mean")
    for i in range(5):
        plot_estimate_function_as_m( data[i], i)

    plt.legend()
    plt.title("estimate as function of flips" )
    plt.savefig("section16a.png")

    def bound_by_one(X ,Y):
        '''
            bound_by_one - assignment 1's at the greater values.
        '''
        Y [Y > 1] = 1
        return X, Y

    def hoffding(eps, length):
        '''
            estimate of the probability by Hoffding.
        '''
        X = np.arange(length)
        return bound_by_one(X , 2* np.exp(-2*(X)*(eps**2)))

    def chaviapprox(eps, length):
        '''
            estimate of the probability by Chebyshevs.
        '''
        X = np.arange(length)
        return bound_by_one(X, 0.25  / ( X * eps**2))

    def plotingseq_distance_from_bound(eps, length, data):
        '''
            calculate the amount of samples which their aritmatic mean is
            greater than given epsilon.
        '''
        precentage = [ ]
        for m in range(1,length+1):
            count = 0
            for sample in data:
                if ( abs(np.sum(sample[:m]) - 0.25*m) > eps*m):
                    count += 1
            precentage.append(count / samples)
        return np.arange(length), precentage


    relevantlength = [100 ,100 ,100 ,10**4 ,10**5 ]

    funcnaems = [ "Chebyshevs" , "Hoffding"  ]
    for i, eps in enumerate( epsilon ):
        if i == 4:
            plt.figure()
            for j, func in enumerate( [chaviapprox, hoffding] ):
                plt.scatter( *(func(eps, relevantlength[i] )),
                 s=1, marker='.' ,  label="{0}".format(funcnaems[j]))
            plt.scatter(\
             *plotingseq_distance_from_bound(eps, relevantlength[i], data),\
              s=1, marker='.', label=" $ P( \\overline{{x}} - E[x] > {0})$".format(eps))
            plt.title("upper bound, $ \\varepsilon = {0} $".format(eps))
            plt.xlabel("flips")
            plt.ylabel("probability bound")
            plt.legend()
            plt.savefig("section16b{0}.png".format(i))



if __name__ == "__main__":

    section11()
    section12()
    section13()
    section14()
    section15()
    section16()
