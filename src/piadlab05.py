import numpy as np
from pandas import read_csv, DataFrame
from scipy import sparse
data = read_csv('../data/zoo.csv', sep=',')

def freq(x, prob=True):
    data = x.value_counts(normalize=prob)
    xi = data.keys().tolist()
    ni = data.tolist()
    return xi, ni

def freq2(x, y, prob=True):
    d = data[[x.name, y.name]].value_counts(normalize=prob)
    xi = d.keys().tolist()
    #yi = data[[x.name, y.name]].groupby([y.name, x.name]).value_counts().keys().tolist()
    ni = d.tolist()
    #ni_yi = data[[x.name, y.name]].groupby([y.name, x.name]).value_counts(normalize=prob).tolist()
    return xi, ni

def entropy(x):
    _, h = freq(x)
    res = 0
    [res := res + h[i] * np.log2(h[i]) for i in range(len(h))]
    return -res

def entropy2(x, y):
    _, h = freq2(x, y)
    res = 0
    [res := res + h[i] * np.log2(h[i]) for i in range(len(h))]
    return -res

def infogain(x, y):
    return entropy(x) + entropy(y) - entropy2(x, y)

#print(freq(data['milk']))
#print(freq(data['milk'], False))
#print(freq2(data['legs'], data['milk']))
#print(freq2(data['legs'], data['milk'], False))
#print(entropy(data['milk']))
#print(entropy2(data['legs'], data['toothed']))
print(infogain(data['type'], data['legs']))

