from models import abcModl, DecisionTree


class WeakFactory:
    class WeakLernerByFeature(abcModel):

        def __init__(self, _model):
            self.mod = _model

        def fit(self, X, y):
            self.mod.fit(X,y)

        def predict(self, X):
            return self.mod.predict()

        

    def __init__ (self):
        pass
    
    @staticmethod
    def CreateWeaks(self):

        return { 

                "DayOfWeek" : WeakFactory ( DecisionTree() ),
                "FlightDate" : WeakFactory ( DecisionTree() ),
                "Reporting_Airline" : WeakFactory ( DecisionTree() ),
                "Tail_Number" : WeakFactory ( DecisionTree() ),
                "Flight_Number_Reporting_Airline" : WeakFactory ( DecisionTree() ),
                "Origin" : WeakFactory ( DecisionTree() ),
                "OriginCityName" : WeakFactory ( DecisionTree() ),
                "OriginState" : WeakFactory ( DecisionTree() ),
                "Dest" : WeakFactory ( DecisionTree() ),
                "DestCityName" : WeakFactory ( DecisionTree() ),
                "DestState" : WeakFactory ( DecisionTree() ),
                "CRSDepTime" : WeakFactory ( DecisionTree() ),
                "CRSArrTime" : WeakFactory ( DecisionTree() ),
                "CRSElapsedTime" : WeakFactory ( DecisionTree() ),
                "Distance" : WeakFactory ( DecisionTree() ),
                "ArrDelay" : WeakFactory ( DecisionTree() ),
                "DelayFacto" : WeakFactory ( DecisionTree() )
        }

    @staticmethod
    def generateTeam ( features ):

        class Team(abcModl):
            
            
        






class StrongClassifer:
    
    def __init__ (self, features):
        pass

    def predict(self, X):
        pass
    
    def train(self, X, y):
        pass

    

