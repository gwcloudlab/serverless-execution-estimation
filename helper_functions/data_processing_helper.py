import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

"""
This class represents a particular chain of functions

Example chain: input -> Function 1 -> Function 2 -> Function 3 -> output

class structure:

SeverlessChain

* run_id        - the id of the run of the chain
* run_dataset   - the dataset containing the chain
* functions     - dictionary mapping functions of the chain to their corresponding Serverless Function object
* models        - models representing the entire chain
* x_train       - the training input args
* x_test        - the test input args
* y_train       - the training execution times (output)
* y_test        - the testing execution times (output)

"""
class ServerlessChain:

    @staticmethod
    def combine_arrays(arr1, arr2, type):

        new_size = min(arr1.size, arr2.size)
        arr1 = arr1[0:new_size]
        arr2 = arr2[0:new_size]
        if type == "merge":
            return np.add(arr1, arr2)
        elif type == "concat":
            return np.concatenate(arr1, arr2)


    def __init__(self, run_id, functions, data):
        
        self.run_id = run_id

        self.functions = {}
        self.models = {}

        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []

        for function in functions:
            self.functions[function] = data.functions[function]

            if(len(self.x_train) == 0):

                self.y_train =  self.functions[function].y_train
                self.y_test  =  self.functions[function].y_test
                self.x_train = self.functions[function].x_train
                self.x_test = self.functions[function].x_test
            else:
    
                self.combine_arrays(self.y_train, np.array(self.functions[function].y_train), "merge")
                self.combine_arrays(self.y_test, np.array(self.functions[function].y_test), "merge")


    

    def add_model(self, model):
        self.models[model.model_name] = model
        

                 
class ServerlessFunction:

    def __init__(self, function_id, data):
        self.function_id = function_id
        self.function_dataset = data
        self.models = {}
    
    def add_model(self, model):
        self.models[model.model_name] = model
        
    def split_data(self, indexes, y_column,size, random):
        self.indexes = indexes
        self.y_column =  y_column
        x = self.function_dataset[indexes]
        y = self.function_dataset[y_column]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=size, random_state=random)

        
class Dataset:
    def __init__(self, dataset):
        """_summary_

        Args:
            dataset (_type_): _description_
        """
        self.dataset = dataset
        self.model_values = {}
            
    @staticmethod  
    def convert_non_numeric_argument(df, columnIndex):
        """_summary_

        Args:
            df (_type_): _description_
            columnIndex (_type_): _description_
        """
        uniqueValues = {}
        for value in df[columnIndex]:
            if value not in uniqueValues:
                uniqueValues[value] = len(uniqueValues)
        df[columnIndex] = df[columnIndex].map(lambda x: uniqueValues[x])
    
    def split_into_functions(self, columnID, columnPivot, valuesPivot, non_numeric_condition):
        """_summary_

        Args:
            columnID (_type_): _description_
            columnPivot (_type_): _description_
            valuesPivot (_type_): _description_
            non_numeric_condition (_type_): _description_
        """
        self.functions = {}
        groupdedFunctionData = self.dataset.groupby(columnID)
        for function in groupdedFunctionData.groups:
            functionData = groupdedFunctionData.get_group(function)
            indexes = list(functionData.drop([columnPivot, valuesPivot], axis=1).columns)
            self.functions[function] = ServerlessFunction(function, functionData.pivot(index=indexes, columns=columnPivot, values=valuesPivot).reset_index())
            if non_numeric_condition in self.functions[function].function_dataset.columns:
                Dataset.convert_non_numeric_argument(self.functions[function].function_dataset, non_numeric_condition)
    
    def split_into_runs(self, runs):
        self.runs = {}
        i = 1
        for run in runs:
            self.runs[i]  = ServerlessChain(i, run, self)
            i=i+1
       
    
    def group_function_values(self, model_name):

        self.labels = []
        overfitting = []
        underfitting = []
        mae_values = []

        for function_id in self.functions:

            function = self.functions[function_id]

            self.labels.append("f_ " + str(function_id))

            overfitting.append(function.models[model_name].overpredicted)
            underfitting.append(function.models[model_name].underpredicted)
            mae_values.append(function.models[model_name].mae)

        self.model_values[model_name] = {
            "overpredicted" : overfitting,
            "underpredicted" : underfitting,
            "mae_values" : mae_values
        }


    def group_run_values(self, model_name):

        self.run_labels = []
        overfitting = []
        underfitting = []
        mae_values = []

        for run_id in self.runs:
            run = self.runs[run_id]

            self.run_labels.append("r_ " + str(run_id) + "(" + str(run.functions.keys()) + ")")

            overfitting.append(run.models[model_name].overpredicted)
            underfitting.append(run.models[model_name].underpredicted)
            mae_values.append(run.models[model_name].mae)

        self.model_values[model_name] = {
            "overpredicted" : overfitting,
            "underpredicted" : underfitting,
            "mae_values" : mae_values
        }