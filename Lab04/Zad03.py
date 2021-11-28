import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


diabetes = pd.read_csv('normalized_diabetes.csv')
X = diabetes.drop(columns='8', axis=1)
y = diabetes['8']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

classifier = MLPClassifier(hidden_layer_sizes=(6, 3), max_iter=500, activation='relu')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print "Accuracy: {:.2f}".format(classifier.score(X_test, y_test))
print confusion_matrix(y_test, y_pred)
