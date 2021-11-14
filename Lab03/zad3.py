from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

iris = load_iris()
type(iris)

print iris.data
print iris.feature_names
print iris.target
print iris.target_names

X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

print X_train.shape
print X_test.shape

print y_train.shape
print y_test.shape

classes = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
knn3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn5 = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn11 = KNeighborsClassifier(n_neighbors=11, metric='euclidean')

knn3.fit(X_train, y_train)
knn5.fit(X_train, y_train)
knn11.fit(X_train, y_train)

y_predict3 = knn3.predict(X_test)
y_predict5 = knn5.predict(X_test)
y_predict11 = knn11.predict(X_test)

for row in y_predict3:
    print classes[y_predict3[row]]
print '***'
for row in y_predict5:
    print classes[y_predict5[row]]
print '***'
for row in y_predict11:
    print classes[y_predict11[row]]

print ('K = 3: ')
print (classification_report(y_test, y_predict3))
print (confusion_matrix(y_test, y_predict3))

print ('K = 5: ')
print (classification_report(y_test, y_predict5))
print (confusion_matrix(y_test, y_predict5))

print ('K = 11: ')
print (classification_report(y_test, y_predict11))
print (confusion_matrix(y_test, y_predict11))
