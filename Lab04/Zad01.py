import numpy as np


def forwardPass(wiek, waga, wzrost):
    hidden1 = (wiek*-0.46122+waga*0.97314+wzrost*-0.39203+1*0.80109)
    hidden1_poaktywacji = 1/(1+np.exp(-hidden1))
    hidden2 = (wiek*0.78548+waga*2.10584+wzrost*-0.57847+1*0.43529)
    hidden2_poaktywacji = 1/(1+np.exp(-hidden2))
    output = (hidden1_poaktywacji*-0.81546+hidden2_poaktywacji*1.03775+1*-0.2368)
    print output


forwardPass(23,75,176)
forwardPass(48,97,178)
