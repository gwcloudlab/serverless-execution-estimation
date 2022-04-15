import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class ServerlessRun:
    def __init__(self, run_id, data, function_identifier, dataset, function_model_id, run_model_id):
        self.run_id = run_id
        self.run_dataset = data
        self.function_identifier = function_identifier
        self.function_model_id = function_model_id
        self.run_model_id = run_model_id

    def associate_data_with_models(self, dataset):
        self.function_models = {}
        self.function_datasets = {}
        groupdedFunctionData = self.dataset.groupby(self.function_identifier)
        for function in groupdedFunctionData.groups:
            self.function_models[function] = dataset.functions[function]
            self.function_datasets[function] = groupdedFunctionData.get_group(function);
        
    def estimate_overall_running_time(self, indexes, y_column):
        for function_id in self.function_datasets:
            function = self.function_datasets[function_id]
            function.models[function_model_id].fit(function, indexes, y_column)
    
    def set_y_values(self):
        self.y_true = self.y_test.squeeze()
        self.y_test  = self.model.predict(self.function.x_test)
    
                 
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
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=size, random_state=random);


        
class Dataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model_values = {}
            
    @staticmethod  
    def convert_non_numeric_argument(df, columnIndex):
        uniqueValues = {}
        for value in df[columnIndex]:
            if value not in uniqueValues:
                uniqueValues[value] = len(uniqueValues)
        df[columnIndex] = df[columnIndex].map(lambda x: uniqueValues[x])
    
    def split_into_functions(self, columnID, columnPivot, valuesPivot, non_numeric_condition):
        self.functions = {}
        groupdedFunctionData = self.dataset.groupby(columnID)
        for function in groupdedFunctionData.groups:
            functionData = groupdedFunctionData.get_group(function);
            indexes = list(functionData.drop([columnPivot, valuesPivot], axis=1).columns)
            self.functions[function] = ServerlessFunction(function, functionData.pivot(index=indexes, columns=columnPivot, values=valuesPivot).reset_index())
            if non_numeric_condition in self.functions[function].function_dataset.columns:
                Dataset.convert_non_numeric_argument(self.functions[function].function_dataset, non_numeric_condition)
    
    def group_function_values(self, model_name):
        self.labels = []
        overfitting = []
        underfitting = []
        mae_values = []
        for function_id in self.functions:
            function = self.functions[function_id]
            self.labels.append("f_ " + str(function_id))
            overfitting.append(function.models[model_name].overfitted)
            underfitting.append(function.models[model_name].underfitted)
            mae_values.append(function.models[model_name].mae)
        self.model_values[model_name] = {
            "overfitted" : overfitting,
            "underfitted" : underfitting,
            "mae_values" : mae_values
        }