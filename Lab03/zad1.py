from csv import reader
import numpy as np


def classify_iris(sepallength, sepalwidth, petallength, petalwidth):
    if float(petalwidth) < 1.0:
        return 'setosa'
    if 2.0 < float(petallength) <= 5.0:
        return 'versicolor'
    else:
        return 'virginica'


compatibility = 0
with open('iris.csv', 'r') as irisdatabase:
    csv_reader = reader(irisdatabase)
    next(csv_reader)

    for rows in csv_reader:
        iris_classification = classify_iris(rows[0], rows[1], rows[2], rows[3])
        if iris_classification == rows[4]:
            compatibility = compatibility + 1
        print iris_classification

    print compatibility
    print round(np.divide(float(compatibility), 150) * 100)

