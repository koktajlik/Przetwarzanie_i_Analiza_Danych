import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cluster


def distp(X, C, e = None):
    X = np.transpose(X)
    tmp_array, tmp_array2 = np.empty(shape=(len(C), len(X[0]))), np.empty(shape=(len(C), len(X[0])))
    for i in range(len(C[0])):
        for j in range(len(X[0])):
            tmp_array[i, j] = np.sqrt((C[0][i] - X[0][j])**2 + (C[1][i] - X[1][j])**2)
            # tmp_array2[i, j] = np.linalg.norm(C[])
            #tmp_array2[j, i] = np.sqrt((np.subtract(C[i], X[j]))*np.transpose((np.subtract(C[i], X[j]))))
    return np.transpose(tmp_array)

def distm(X, C):
    V = np.cov(X[0], X[1])
    tmp_array = np.empty(shape=(len(C), len(X[0])))
    for i in range(len(C)):
        for j in range(len(X[0])):
            pass

def ksrodki(X, k, iter):
    pass
    # for _ in range(iter):
    #     centroids = []
    #     for i in k:
    #         cen = x[]
from scipy.spatial.distance import cdist

# Defining our function
def kmeans(x, k, C, iter):
    distances = cdist(x, C, 'euclidean')
    p2 = [np.argmin(i) for i in distances]
    for _ in range(iter):
        centroids = []
        print(len(np.unique(p2)))
        #centroids.append(temp_cent)
        for j in range(k):
            # Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = x[p2 == j, :].mean(axis=0)
            centroids.append(temp_cent)
        centroids = np.vstack(centroids)  # Updated Centroids
        distances = cdist(x, C, 'euclidean')
        p2 = [np.argmin(i) for i in distances]
    return p2

csv = pd.read_csv('../data/autos.csv')
v1, v2, v3 = csv['length'], csv['width'], csv['city-mpg']
X = list([v1, v2, v3])
n = len(X)
m = len(X[0])
K = 3
k = np.arange(1, K+1)
X = np.transpose(np.array(X))
C = []
for _ in range(K):
    rn = np.random.randint(0, m)
    C.append(X[rn])
C = np.array(C)
#
label = kmeans(X, K, C, 1000)
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(X[label == i , 0] , X[label == i , 1] , label = i)
C = np.transpose(C)
plt.scatter(C[0], C[1], color='red')
plt.legend()
plt.show()

# Load Data
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
data = load_digits().data
pca = PCA(2)

# Transform the data
# df = pca.fit_transform(data)
# print('xd')
# label = kmeans(df,2,1000)
# u_labels = np.unique(label)
# for i in u_labels:
#     plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
# plt.legend()
# plt.show()
#print(X)
# print(distp(X, C))
# plt.scatter(X[0], X[1])
# plt.show()