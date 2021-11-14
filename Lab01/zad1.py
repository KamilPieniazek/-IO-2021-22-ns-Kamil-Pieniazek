import math
import random
import numpy as np

# a
from numpy import matrix

a = 123
b = 16
print "Zadanie a: ", (a * b)

# b
c = [3, 8, 9, 10, 12]
d = [8, 7, 7, 5, 6]
e = []

print "Zadanie b: ", np.sum((math.fsum(c), math.fsum(d)))

for x, y in zip(c, d):
    e.append(x * y)

print e

# c
print "Zadanie c"
print "Iloczyn skalarny: ", np.sum(e)

m = []
for k, l in zip(c, d):
    m.append(np.abs(k - l))
print "Dlugosci eukidlesowe: ", m

# d
f = np.arange(2, 11).reshape(3, 3)

# e
i = 0
e = []
while i < 50:
    e.append(random.randrange(1, 100))
    i += 1

    if i == 50:
        break

print "Zadanie e"
print e

# f
print "Zadanie f"
print "Srednia :", np.mean(e)
print "Min: ", np.min(e)
print "Max: ", np.max(e)
print "Odchylenie standardowe: ", np.std(e)

# g
