import numpy as np
import pandas
import pandas as pd
from sklearn import preprocessing

diabetes = pd.read_csv('diabetes.csv')
diabetes['class'] = diabetes["class"].replace({"tested_negative": 0.0, "tested_positive": 1.0})


def remove_outliers(dataframe, column_name):
    print column_name
    values = dataframe[column_name]
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    print ' IQR: ', iqr
    median = values.median()
    low = median - 1.5*iqr
    high = median + 1.5*iqr
    dataframe_no_outliers =  dataframe[(dataframe[column_name] >= low)&(dataframe[column_name]<=high)]
    print 'Outliers no.: ', len(dataframe.index) - len(dataframe_no_outliers.index)
    return dataframe_no_outliers


for column in diabetes.drop(columns='class'):
    diabetes = remove_outliers(diabetes, column)

print 'Length of dataset without outliers: ', len(diabetes)

dataframe_values = diabetes.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(diabetes)
normalized_dataframe = pd.DataFrame(x_scaled)

print normalized_dataframe

normalized_dataframe.to_csv('normalized_diabetes.csv')
