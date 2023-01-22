import pandas as pd
import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt

#2.1
print("zad.2.1")
p1_date = np.array(['2020-03-01','2020-03-02','2020-03-03','2020-03-04','2020-03-05',]).reshape((5,1))
p1_rand = np.random.normal(size=(5, 3))
p1 = pd.DataFrame(np.hstack((p1_date, p1_rand)), columns=['data','A', 'B', 'C'])
print(p1)

#2.2
print("zad.2.2")
p2 = pd.DataFrame(np.random.randint(-100, 100, size=(20,3)), columns=['A', 'B', 'C'])
p2.index.name = 'id'
print(p2.head(3))
print(p2.tail(3))
print(p2.index.name)
print(p2.columns) #nazwy kolumn
print(p2.to_string(index= False, header=False))#dane bez indesków i nagłówków kolumn
print(p2.sample(n = 5)) #losowo wybrane wiersze
print(p2['A'], p2[['A', 'B']], sep="\n")
print(p2.iloc[:3, :2]) #iloc - indeksowane
print(p2.iloc[4])
print(p2.iloc[[0,5,6,7], [1, 2]])

#2.3
print("zad.2.3")
des = p2.describe()
print(des > 0)
print(p2.mask(p2 <= 0).describe())
print(p2['A'].mask(p2['A'] <= 0).describe())
print(scipy.stats.describe(p2).mean)
print(scipy.stats.describe(p2, axis=1).mean)

#2.4
print("zad.2.4")
p41 = pd.DataFrame(np.arange(0,20).reshape(5, 4), columns=['a', 'b', 'c', 'd'])
p42 = pd.DataFrame(np.arange(20,40).reshape(5, 4), columns=['a', 't', 'c', 'u'])
p4 = pd.concat([p41, p42])
p4_trn = np.transpose(p4)
print(p4)
print(p4_trn)

#2.5
print("zad.2.5")
df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y":['a', 'b', 'a', 'b', 'b']}, index = np.arange(5))
df.index.name = 'id'
print(df)
print(df.sort_index)
print(df.sort_values('y', ascending=False))

#2.6
slownik = {'Day':['Mon','Tue','Mon','Tue','Mon'], 'Fruit': ['Apple','Apple','Banana','Banana','Apple'], 'Pound': [10, 15, 50, 40, 5], 'Profit':[20, 30, 25, 20, 10]}
df3 = pd.DataFrame(slownik)
print(df3)
print(df3.groupby('Day').sum())
print(df3.groupby(['Day', 'Fruit']).sum())

#2.7
print('2.7')
df = pd.DataFrame(np.random.randn(20, 3), index = np.arange(20), columns=['A','B','C'])
df.index.name='id'
print(df)
df['B'] = 1 #zmiana wartości w calej kolumnie B na 1
print(df)
df.iloc[1,2] = 10 #zmiana wartości w pierwszym wierszu i 2 kolumnie na 10
print(df)
df[df<0] = -df #zamiana wartosci ujemnej na dodatnia
print(df)

#2.8
df.iloc[[0, 3], 1] = np.nan #zamiana wartosci w pierwszej kolumnie, zerowym i trzecim wierszy na NaN
print(df)
df.fillna(0, inplace=True) #na miejsce wartosci NaN wpisuje odpowiednio wartosc pierwszego argumentu
print(df)
df.iloc[[0,3], 1] = np.nan
df = df.replace(to_replace=np.nan, value=-9999) #zastepuje wartosc pierwszego argumentu na wartosc w drugim argumencie
print(df)
df.iloc[[0,3], 1] = np.nan
print(pd.isnull(df)) #zwraca True/False w zależności czy są wartości Nan, None

#ZADANIE
df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': ['a','b','a','b','b']})
print('3.1')
print(df.groupby('y').mean())

print('3.2')
print(df.value_counts())

print('3.3')
#p31 = np.loadtxt('../data/autos.csv', delimiter=',',dtype='str')
#print(p31)
p32 = pd.read_csv('../data/autos.csv')
#print(p32.dtypes)
print(p32)
#w loadtxt trzeba podać typy danych jesli jest ich wiecej niz jeden, trzeba podac jaki jest ogranicznik
#read_csv wyswietla ile jest lacznie kolumn i wierszy

print('3.4')
print(p32[['make','city-mpg','highway-mpg']].groupby('make').mean().mean(axis=1))

print('3.5')
print(p32[['fuel-type', 'make']].groupby('make').value_counts())
#print(p32.groupby('make').value_counts('fuel-type'))

print('3.6')
p61=np.polyfit(np.array(p32['city-mpg']),p32['length'],1)
p62=np.polyfit(np.array(p32['city-mpg']),p32['length'],2)
print(p61, p62) #wspolczynniki przy potegach

print('3.7')
r, p = scipy.stats.pearsonr(p32['city-mpg'], p32['length'])
print(f'wspolczynnik korelacji: {r}')

#print('3.8')
slope, intercept, r, p, stderr = scipy.stats.linregress(p32['city-mpg'], p32['length'])
fig, ax = plt.subplots()
ax.plot(p32['city-mpg'], p32['length'], 'o')
ax.plot(p32['city-mpg'], intercept + slope * p32['city-mpg'])
plt.show()

#3.9
#x = np.linspace(0,209,210).reshape(1, 210)
#kernel = scipy.stats.gaussian_kde(p32['length'])
#plt.plot(x, kernel)
#plt.show()

