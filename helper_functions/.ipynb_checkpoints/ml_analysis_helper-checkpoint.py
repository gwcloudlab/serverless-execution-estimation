import pandas as pd
import matplotlib.pyplot as plt
import numpy
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class Model:
    
    def __init__(self, model_name, model, function):
        self.model_name = model_name
        self.model = model
        self.function = function
        self.calculate_mae()
        self.calculate_number_overpredicting_underpredicting()
        
    def set_y_values(self):
        self.y_true = self.function.y_test.squeeze()
        self.y_pred = self.model.predict(self.function.x_test)
    


    def calculate_mae(self):
        self.model.fit(self.function.x_train, self.function.y_train)
        self.mae = mean_absolute_error(self.model.predict(self.function.x_test), self.function.y_test)
                                       
    def fit(self, x_test, y_test):
        return self.model.fit(self.function.x_train, self.function.y_train)
                                       
                                       
    def calculate_number_overpredicting_underpredicting(self):
        overpredicted = 0
        underpredicted = 0
        y_values = self.model.predict(self.function.x_test)
        self.predicted_y_values = numpy.reshape(y_values, y_values.size)
        
        self.difference = (self.predicted_y_values-self.function.y_test.squeeze()).tolist()
        self.difference.sort()
    
        
        for value in self.difference:
            if value > 0:
                overpredicted = overpredicted + 1
            if value < 0:
                underpredicted = underpredicted + 1
        self.overpredicted = overpredicted
        self.underpredicted = underpredicted  
                                       
    def update_model(self, updated_model):                    
        self.model = model_updated_model
        self.calculate_mae()
        self.calculate_number_overpredicting_underpredicting()




def optimize(models, penalty):
    minMae = models[0].mae
    minModel = models[0]
    previousunderpredicted = models[0].underpredicted 
    total = models[0].underpredicted + models[0].overpredicted
    for model in models:
        fittingDifference = ((model.underpredicted - previousunderpredicted)/total * penalty)
        if (model.mae + fittingDifference) < minMae:
            minMae = model.mae
            minModel = model
            previousunderpredicted = model.underpredicted
            
    return [minMae, minModel, previousunderpredicted]