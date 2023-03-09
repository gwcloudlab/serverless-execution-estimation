from sklearn.tree import _tree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)
execution_data = pd.read_csv("https://gitlab.com/lenapster/faascache/-/raw/master/machine-learning/data/data_image_caching.csv")
image_attributes = pd.read_csv("https://gitlab.com/lenapster/faascache/-/raw/master/machine-learning/data/features_image.csv", delimiter=';')
execution_data.groupby("function")
image_attributes["image_url"] = image_attributes["url"].map(lambda x: x.split("/")[-1])
execution_data = execution_data.drop(["init_ms", "end_ms"], axis=1)
image_attributes = image_attributes.drop(["sizex", "sizey", "url"], axis=1)
image_attributes = image_attributes.rename({"file_size": "file_size_argument"})
execution_data["name"] = execution_data["name"].map(lambda x: x + "_argument")
data = data_processing_helper.Dataset(execution_data)
data.split_into_functions('function', 'name', 'arguments', 'format_argument')
for function in data.functions:
    decision_tree_model = DecisionTreeRegressor(random_state=12)
    model = ml_analysis_helper.Model('decision_tree_model' , decision_tree_model, data.functions[function])
    data.functions[function].add_model(model)
print(data.functions[function].models[['decision_tree_model']])