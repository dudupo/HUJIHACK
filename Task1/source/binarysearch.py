import numpy as np


class binarysearch:

    def __init__ (self, models, times_ranges):
        self.mods = models
        self.iterations = 3

    def get_model(self, times_delay):
        return self.mod

    '''
         for _bool in re > middle ] )
                y = np.array( [ {  False : [0] , True : [1]  }[ _bool ]
    '''
    # def load_classifers(self, classifers_steps):
        
    #     pass
    
    
    def _predict(self, X, treshold):
        return self.mods[treshold].predict(X)

    def predict(self, X, start_range, end_range):

        times_delay = 0
        for j in range(self.iterations):
            middle =(start_range + end_range)/2
            res = self._predict(X , start_range ).flatten()
            print(res)
            start_range, end_range = (1 - res) * start_range + res * middle  , (1 - res) * middle +  res * end_range
        return middle



# testing the idea.
def foo( y, s , e ):
    middle = (s+e)/2
    return  middle * y  ,  middle + middle * y  

if __name__ == "__main__":
    s , e = np.array( [ 0, 0 ,0]), np.array( [ 1, 1 ,1]) 
    y = np.array( [1, 0, 1] )
    print( foo(y, s, e) )

    


