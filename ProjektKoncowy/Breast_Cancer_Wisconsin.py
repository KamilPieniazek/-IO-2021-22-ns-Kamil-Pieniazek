import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydotplus
import seaborn as sns
from IPython.display import Image
from pandas.plotting import scatter_matrix
from sklearn.externals.six import StringIO
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

# DATA PREPROCESSING

data = pd.read_csv('data.csv')
print (data.shape)
print data.isnull().sum()
data.drop('Unnamed: 32', inplace=True, axis=1)
data.drop('id', inplace=True, axis=1)
print data['diagnosis'].value_counts()
data['diagnosis'] = data["diagnosis"].replace({"M": 1.0, "B": 0.0})


# for columns in data.columns[1:11]:
#     print "Mean value for ", columns, " :", data[columns].mean()
#     print "Max value for ", columns, " :", data[columns].max()
#     print "Min value for ", columns, " :", data[columns].min()
#     print "Standard deviation value for ", columns, " :", data[columns].std()

# DATA GRAPH VISUALISATION

features_mean = list(data.columns[1:11])

# plt.figure(figsize=(20, 20))
# sns.heatmap(data.corr(), annot=True)

plt.figure(figsize=(10, 10))
# sns.heatmap(data[features_mean].corr(), annot=True, square=True, cmap='coolwarm')

color_dic = {1.0: 'red', 0.0: 'blue'}
colors = data['diagnosis'].map(lambda x: color_dic.get(x))

# sm = scatter_matrix(data[features_mean], c=colors, alpha=0.4, figsize=(20, 20))

bins = 12
plt.figure(figsize=(15, 15))

# for i, feature in enumerate(features_mean):
#     rows = int(len(features_mean) / 2)
#
#     plt.subplot(rows, 2, i + 1)
#
#     sns.distplot(data[data['diagnosis'] == 1.0][feature], bins=bins, color='red', label='Malignant');
#     sns.distplot(data[data['diagnosis'] == 0.0][feature], bins=bins, color='blue', label='Benign');
#
#     plt.legend(loc='upper right')
#
# plt.tight_layout()
# plt.show()



# CLASSIFICATION

X = data.loc[:, features_mean]
y = data.loc[:, 'diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

accuracy = []
cvs = []
methods = []

# KNN

start_knn = time.time()

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
prediction_knn = knn.predict(X_test)
scores_knn = cross_val_score(knn, X, y, cv=5)

end_knn = time.time()

accuracy.append(accuracy_score(prediction_knn, y_test))
cvs.append(np.mean(scores_knn))
methods.append('KNN')

print ("KNN")
print("Accuracy: {0:.2%}".format(accuracy_score(prediction_knn, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores_knn), np.std(scores_knn) * 2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end_knn - start_knn))

# sns.heatmap(confusion_matrix(prediction_knn, y_test), annot=True, fmt="d")
# plt.show()

# DECISION TREE

start_dt = time.time()

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
prediction_dt = dt.predict(X_test)
score = dt.score(X_test, y_test)
scores_dt = cross_val_score(dt, X, y, cv=5)

end_dt = time.time()

accuracy.append(accuracy_score(prediction_dt, y_test))
cvs.append(np.mean(scores_dt))
methods.append('Decision Tree')

print ("Decision tree")
print("Accuracy: {0:.2%}".format(accuracy_score(prediction_dt, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores_dt), np.std(scores_dt) * 2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end_dt - start_dt))

# sns.heatmap(confusion_matrix(prediction_dt, y_test), annot=True, fmt="d")


dot_data = StringIO()
export_graphviz(dt, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=list(X), class_names=['Benign', 'Malignant'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('breast_cancer_wisconsin.png')
Image(graph.create_png())

# STOCHASTIC GRADIENT DESCENT

start_sgd = time.time()

sgd = SGDClassifier()
sgd.fit(X_train, y_train)
prediction_sgd = sgd.predict(X_test)
scores_sgd = cross_val_score(sgd, X, y, cv=5)

end_sgd = time.time()

accuracy.append(accuracy_score(prediction_sgd, y_test))
cvs.append(np.mean(scores_sgd))
methods.append('SGD')



print ("SGD")
print("SGD Classifier Accuracy: {0:.2%}".format(accuracy_score(prediction_sgd, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores_sgd), np.std(scores_sgd) * 2))
print("Execution time: {0:.5} seconds \n".format(end_sgd - start_sgd))
#
# sns.heatmap(confusion_matrix(prediction_sgd, y_test), annot=True, fmt="d")

# NAIVE BAYES

start_nb = time.time()

nb = GaussianNB()
nb.fit(X_train, y_train)
prediction_nb = nb.predict(X_test)
scores_nb = cross_val_score(nb, X, y, cv=5)

end_nb = time.time()

accuracy.append(accuracy_score(prediction_nb, y_test))
cvs.append(np.mean(scores_nb))
methods.append('Naive Bayes')


print("Accuracy: {0:.2%}".format(accuracy_score(prediction_nb, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores_nb), np.std(scores_nb) * 2))
print("Execution time: {0:.5} seconds \n".format(end_nb - start_nb))

# sns.heatmap(confusion_matrix(prediction_nb, y_test), annot=True, fmt="d")

# SUPPORT VECTOR MACHINES (SVC AND LINEAR SVC)

start_svc = time.time()

svc = SVC()
svc.fit(X_train, y_train)
prediction_svc = svc.predict(X_test)
scores_svc = cross_val_score(svc, X, y, cv=5)

end_svc = time.time()

accuracy.append(accuracy_score(prediction_svc, y_test))
cvs.append(np.mean(scores_svc))
methods.append('SVC')

#sns.heatmap(confusion_matrix(prediction_svc, y_test), annot=True, fmt="d")

print "SVC"
print("SVC Accuracy: {0:.2%}".format(accuracy_score(prediction_svc, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores_svc), np.std(scores_svc) * 2))
print("Execution time: {0:.5} seconds \n".format(end_svc - start_svc))

start_Linearsvc = time.time()

LinearSVC = LinearSVC()
LinearSVC.fit(X_train, y_train)
prediction_LinearSVC = LinearSVC.predict(X_test)
scores_LinearSVC = cross_val_score(LinearSVC, X, y, cv=5)

end_LinearSVC = time.time()

accuracy.append(accuracy_score(prediction_LinearSVC, y_test))
cvs.append(np.mean(scores_LinearSVC))
methods.append('Linear SVC')



print("Linear Accuracy: {0:.2%}".format(accuracy_score(prediction_LinearSVC, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores_LinearSVC), np.std(scores_LinearSVC) * 2))
print("Execution time: {0:.5} seconds \n".format(end_LinearSVC - start_Linearsvc))

#sns.heatmap(confusion_matrix(prediction_LinearSVC, y_test), annot=True, fmt="d")

# RANDOM FOREST

start_rf = time.time()

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
prediction_rf = rf.predict(X_test)
scores_rf = cross_val_score(rf, X, y, cv=5)

end_rf = time.time()

accuracy.append(accuracy_score(prediction_rf, y_test))
cvs.append(np.mean(scores_rf))
methods.append('RF')



print("Random Forest Accuracy: {0:.2%}".format(accuracy_score(prediction_rf, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores_rf), np.std(scores_rf) * 2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end_rf - start_rf))

#sns.heatmap(confusion_matrix(prediction_rf, y_test), annot=True, fmt="d")

def remove_outliers(dataframe, column_name):
    print column_name
    values = dataframe[column_name]
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    median = values.median()
    low = median - 1.5 * iqr
    high = median + 1.5 * iqr
    dataframe_no_outliers = dataframe[(dataframe[column_name] >= low) & (dataframe[column_name] <= high)]
    return dataframe_no_outliers


for column in data.drop(columns='diagnosis'):
    data_temp = remove_outliers(data, column)

# print 'Length of dataset without outliers: ', len(data_temp)

dataframe_values = data_temp.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(data_temp)
normalized_dataframe = pd.DataFrame(x_scaled)
X = data_temp.drop(columns='diagnosis', axis=1)
y = data_temp['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

start_MLPC = time.time()

classifier = MLPClassifier(hidden_layer_sizes=(6, 3), max_iter=500, activation='relu')
model = classifier.fit(X_train, y_train)

prediction_MLPC = classifier.predict(X_test)
scores_MLPC = cross_val_score(classifier, X, y, cv=5)

end_MLPC = time.time()

accuracy.append(accuracy_score(prediction_MLPC, y_test))
cvs.append(np.mean(scores_MLPC))
methods.append('MLPC')



print("Neural network MLP Classifier: {0:.2%}".format(accuracy_score(prediction_MLPC, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores_MLPC), np.std(scores_MLPC) * 2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end_MLPC - start_MLPC))

#sns.heatmap(confusion_matrix(prediction_MLPC, y_test), annot=True, fmt="d")


y_pos = np.arange(len(methods))
values_accuracy = accuracy

plt.bar(y_pos, values_accuracy, align='center', alpha=0.5)
plt.xticks(y_pos, methods)
plt.ylabel('Methods')
plt.title('Prediction accuracy')

y_pos = np.arange(len(methods))
values_cvs = cvs

plt.bar(y_pos, values_cvs, align='center', alpha=0.5)
plt.xticks(y_pos, methods)
plt.ylabel('Methods')
plt.title('CVS')

plt.show()
