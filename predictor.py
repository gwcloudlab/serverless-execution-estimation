import pandas as pd
import numpy as np
import random
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split

from helper_functions import data_processing_helper, ml_analysis_helper

global data
database = pd.read_csv('database_3.csv')
data = data_processing_helper.Dataset(database)
data.split_data_functions_2(['response_time_resize_small','response_time_resize_medium','response_time_resize_large'], ['file_size','sizex','sizey'], 0.5)

for function in data.functions:
    decision_tree_model = DecisionTreeRegressor(random_state=12)
    model = ml_analysis_helper.Model('decision_tree_model' , decision_tree_model, data.functions[function])
    data.functions[function].add_model(model)


def predict(function, file_size, file_x, file_y):
    args = pd.DataFrame(data = [[file_size, file_x, file_y]], columns=['file_size', 'sizex', 'sizey'])
    return data.functions[function].models['decision_tree_model'].model.predict(args)