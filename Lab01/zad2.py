import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('miasta.csv')
# data = pd.DataFrame({'Rok': [2010],'Gdansk': [460], 'Poznan': [555], 'Szczecin': [405]}, columns=['Rok','Gdansk','Poznan','Szczecin'])
# data.to_csv('miasta.csv', mode='a', index=False, header=False)
print df

years = df['Rok']
gdansk = df['Gdansk']
poznan = df['Poznan']
szczecin = df['Szczecin']
plt.plot(years.astype('str'), gdansk, color='b', marker='o')
plt.plot(years.astype('str'), poznan, color='r', marker='o')
plt.plot(years.astype('str'), szczecin, color='k', marker='o')

# df.plot(years.astype('str'), y=['Gdansk', 'Poznan', 'Szczecin'], color=['b','k','c'], marker='o')
plt.xlabel('Lata')
plt.legend()
plt.ylabel('Liczba ludnosci [w tys]')
plt.title('Ludnosc w miastach Polski')

plt.show()
