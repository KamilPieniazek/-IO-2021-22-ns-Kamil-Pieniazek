from sklearn import tree
from sklearn.datasets import load_iris
import graphviz

iris = load_iris()
X, y = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)

print graph

