import numpy as np
import numpy.lib.stride_tricks


print('tablice')
b = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
b = np.transpose(b)
tab1 = np.arange(100)
tab2 = np.linspace(0,2,10)
tab3 = np.arange(0,101,5)
print(tab1, tab2, tab3, sep='\n')

print('liczby losowe')
tab4 = np.random.normal(size=20).round(2)
tab5 = np.random.randint(1,1001,100)
tab6 = np.zeros((3,2))
tab7 = np.ones((3,2))
tab8 = np.random.randint(0, 300, size = (5, 5), dtype = np.int32)
print(tab4, tab5, tab6, tab7, tab8, sep='\n')
tab9 = np.random.uniform(0, 10, 10)
nowa_tab9 = tab9.astype(int)
tab92 = tab9.round(0).astype(int)
print(f'zadanie\na: {tab9}\nb: {nowa_tab9}\na: {tab92}\n')

print('selekcja danych')
b = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype = np.int32)
print(f'ilość wymiarow: {b.ndim}\nilość elementów: {b.size}')
print(b[:,1:4:2])
print(b[0])
print(b[:,0])
matrix1 = np.random.randint(0, 101, size = (20, 7))
print(matrix1[:,0:4])

print('operacje matematyczne i logiczne')
a = np.random.uniform(0, 11, size=(3, 3))
b = np.random.uniform(0, 11, size=(3, 3))
print(a*b)
print(a + b)
print(a/b)
print(a ** b)
print(b ** a)
print(f'czy wartość macierzy >= 4: {a >= 4}\nczy wartość macierzy >= 1 i <= 4: {np.logical_and(a >=1, a <= 4)}')
print(f'{b}\nsuma głównej przekątnej: {np.matrix.trace(b)}')

print('dane statystyczne')
print(f'suma: {b.sum()}\nminimum: {b.min()}\nmaksimum: {b.max()}\nodchylenie: {b.std()}')
print(f'średnia dla kolumn {np.mean(b, 0)}\nśrednia dla wierszy{np.mean(b, 1)}')

print('rzutowanie wymiarów za pomocą shape lub resize')
tab = np.arange(50)
reshape_tab = tab.reshape(10, 5)
resize_tab = tab.resize(10, 5)
print(f'tabela po użyciu reshape: {reshape_tab}\nwynik resize: {resize_tab}\ntabela po użyciu resize:{tab}')
print(f'Komenda ravel sprowadza wielowymiarową tablicę do jednego wymiaru')
tab1 = np.linspace(0, 3, 5)
tab2 = np.linspace(3, 5, 4)
new_tab = np.append(tab1, tab2)
print(new_tab)
print(new_tab[:, np.newaxis])

print('sortowanie danych')
a = np.random.randn(5, 5)
print(a)
a.sort(axis = 1)
a[::-1].sort(axis = 0)
print(a)

b = np.array([(1, 'MZ', 'mazowieckie'), (2, 'ZP', 'zachodniopomorskie'), (3, 'ML', 'małopolskie')])
b.resize(3, 3)
b = b[b[:,1].argsort()]
print(b)
print(b[2, 2])

print('3. zadanie podsumowujące')
#1
matrix = np.random.randint(0, 300, size = (10, 5))
sum = matrix.trace()
print(f'{np.diag(matrix)}\n{sum}')

#2
tab1 = np.random.normal(size=5)
tab2 = np.random.normal(size=5)
print(tab1, tab2)
print(np.multiply(tab1, tab2))

#3
p3_tab1 = np.random.randint(1, 101, 10)
p3_tab2 = np.random.randint(1, 101, 10)
p3_matrix1 = p3_tab1.reshape(2, 5)
p3_matrix2 = p3_tab2.reshape(2, 5)
print(p3_tab1, p3_tab2, p3_matrix1 + p3_matrix2, sep='\n')

#4
p4_matrix1 = np.arange(1, 21).reshape((4, 5))
p4_matrix2 = np.arange(21, 41).reshape((5, 4))
new_p4 = np.ravel(p4_matrix1) + np.ravel(p4_matrix2)
print(f'{p4_matrix1}\n{p4_matrix2}\n{new_p4}')

#5
print(p4_matrix1[:,2] * p4_matrix1[:, 3])
print(p4_matrix2[:,2] * p4_matrix2[:, 3])

#6
p6_matrix1 = np.random.normal(size=(2, 4))
print(f'{p6_matrix1}\nsrednia: {np.average(p6_matrix1)}\nodchylenie: {np.std(p6_matrix1)}\nwariancja: {np.var(p6_matrix1)}\nmediana: {np.median(p6_matrix1)}\nminimum: {np.min(p6_matrix1)}\nmaksimum: {np.max(p6_matrix1)}')
p6_matrix2 = np.random.uniform(size=(2, 4))
print(f'{p6_matrix2}\nsrednia: {np.average(p6_matrix2)}\nodchylenie: {np.std(p6_matrix2)}\nwariancja: {np.var(p6_matrix2)}\nmediana: {np.median(p6_matrix2)}\nminimum: {np.min(p6_matrix2)}\nmaksimum: {np.max(p6_matrix2)}')

#7
a = np.random.randint(0, 10, size=(3, 3))
b = np. random.randint(0, 10, size= (3, 3))
print(f'{a}\n{b}\n{a*b}\n{np.dot(a, b)}')
print('Dot (iloczyn skalarny) warto używać do szukania kąta między dwoma wektorami')

#8
p8_matrix1 = np.random.uniform(1, 10, (8, 6))
#print(p8_matrix1.strides)
print(numpy.lib.stride_tricks.as_strided(p8_matrix1, shape=(1, 3, 5), strides=(48, 48, 8)))

#9
tab1 = np.random.randint(6, 40, 7)
tab2 = np.random.randint(1, 10, 7)
print(np.vstack((tab1, tab2)))
print(np.hstack((tab1, tab2)))
print('vstack łączy tablice wierszowo, a hstack kolumnowo. Warto je stosować przy tworzeniu macierzy składającej się z wielu tablic')