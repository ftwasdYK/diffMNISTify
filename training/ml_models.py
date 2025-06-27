from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC

class RandomForest:
    def __init__(self):
        self._model = RandomForestClassifier()

    @property
    def param_grid(self):
        return {
            'model__n_estimators': [10, 25, 50],
            'model__max_depth': [10, 25, 35],
        }
    
    def get_model(self):
        return self._model
    
    @property
    def str_name(self):
        return self.__class__.__name__


class RidgeClassifierModel:
    def __init__(self):
        self._model = RidgeClassifier()

    @property
    def param_grid(self):
        return {
            'model__alpha': [0.1, 1.0, 10.0],
            'model__max_iter': [100, 200, 400]
        }
    
    def get_model(self):
        return self._model
    
    @property
    def str_name(self):
        return self.__class__.__name__
    
    
class SupportVectorMachine:
    def __init__(self):
        self._model = SVC()

    @property
    def param_grid(self):
        return {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['linear', 'rbf'],
            'model__gamma': ['scale', 'auto']
        }
    
    def get_model(self):
        return self._model
    
    @property
    def str_name(self):
        return self.__class__.__name__