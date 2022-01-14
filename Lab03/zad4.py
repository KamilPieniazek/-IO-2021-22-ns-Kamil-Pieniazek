import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

diabetes = pd.read_csv('diabetes.csv')
X = diabetes.drop(columns='class', axis=1)
y = diabetes['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_predict_tree = clf.predict(X_test)

knn1 = KNeighborsClassifier(n_neighbors=1)
knn3 = KNeighborsClassifier(n_neighbors=3)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn7 = KNeighborsClassifier(n_neighbors=7)
knn11 = KNeighborsClassifier(n_neighbors=11)

knn1.fit(X_train, y_train)
knn3.fit(X_train, y_train)
knn5.fit(X_train, y_train)
knn7.fit(X_train, y_train)
knn11.fit(X_train, y_train)

y_predict1 = knn1.predict(X_test)
y_predict3 = knn3.predict(X_test)
y_predict5 = knn5.predict(X_test)
y_predict7 = knn7.predict(X_test)
y_predict11 = knn11.predict(X_test)


prediction_accuracy_1 = accuracy_score(y_test, y_predict1)
prediction_accuracy_3 = accuracy_score(y_test, y_predict3)
prediction_accuracy_5 = accuracy_score(y_test, y_predict5)
prediction_accuracy_7 = accuracy_score(y_test, y_predict7)
prediction_accuracy_11 = accuracy_score(y_test, y_predict11)
prediction_accuracy_tree = accuracy_score(y_test, y_predict_tree)

methods = ('KNN1', 'KNN3', 'KNN5', 'KNN7', 'KNN11', 'Decision_tree')
y_pos = np.arange(len(methods))
values = [prediction_accuracy_1, prediction_accuracy_3, prediction_accuracy_5, prediction_accuracy_7, prediction_accuracy_11, prediction_accuracy_tree]

plt.bar(y_pos, values, align='center', alpha=0.5)
plt.xticks(y_pos,methods)
plt.ylabel('Methods')
plt.title('Prediction accuracy')

plt.show()

# print 'KNN 1: ', prediction_accuracy_1
# print 'KNN 3: ', prediction_accuracy_3
# print 'KNN 5: ', prediction_accuracy_5
# print 'KNN 7: ', prediction_accuracy_7
# print 'KNN 11: ', prediction_accuracy_11
# print 'Tree: ', prediction_accuracy_tree



