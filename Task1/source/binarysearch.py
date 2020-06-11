import numpy as np


class binarysearch:

    def __init__ (self, model):
        self.mod = model
        self.iterations = 10

    def predict(self, X, times_delay , start_range, end_range):

        for j in range(self.iterations):
            middle =(start_range + end_range)/2
            y = np.array( [ {  False : [0] , True : [1]  }[ _bool ]
             for _bool in times_delay > middle ] )
            res = self.mod.predict( X, y ).flatten()
            start_range, end_range = middle * res  ,  middle + middle * res
        return middle


# testing the idea.
def foo( y, s , e ):
    middle = (s+e)/2
    return  middle * y  ,  middle + middle * y  


if __name__ == "__main__":
    s , e = np.array( [ 0, 0 ,0]), np.array( [ 1, 1 ,1]) 
    y = np.array( [1, 0, 1] )
    print( foo(y, s, e) )

    


