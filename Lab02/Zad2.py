przedmiot =[1,2,3,4,5,6,7,8,9,10]
wartosc =[11,12,13,14,15,16,17,18,19,20]
waga= [21,22,23,24,25,26,27,28,29,30]

def zawartosc(x,y,z):
    plecak=[]
    sumWartosc= 0
    sumWaga=0
    for i in range(10):
        sumWartosc=sumWartosc+y[i]
        sumWaga=sumWaga+z[i]
        plecak.append(sumWartosc)

# def fitness(przedmiot, wartosc, waga):

print zawartosc(przedmiot,wartosc,waga)

# if()

