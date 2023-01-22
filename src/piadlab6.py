from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
digits = datasets.load_digits()

def wiPCA(x: 'dane', p):
    w, v = np.linalg.eigh(np.cov(x.T), UPLO='U')
    idx = np.argsort(w)[::-1]
    v = v[:, idx]
    return np.dot(x, v[:, 0:p]), v, w[idx]

#1
zad_1a2 = np.dot(np.random.randn(2,2), np.random.randn(2,100)).T
plt.scatter(zad_1a2[:,0], zad_1a2[:,1], color='green')
X, war_wl, wek_wl  = wiPCA(zad_1a2, 1)
v = X * wiPCA(zad_1a2, 1)[0].T
plt.scatter(v[:,0], v[:,1], color='red')
#pocz = [v[:,0].mean(), v[:,1].mean()]
#dl_wek = 3 * np.sqrt(war_wl)
sorted_arr = v[v[:,0].argsort()]
#kon_wek = pocz+wek_wl
#plt.quiver(v[:,0].mean(), v[:,1].mean(), kon_wek[0], kon_wek[1])
plt.quiver(sorted_arr[int(len(sorted_arr)/2),0],sorted_arr[int(len(sorted_arr)/2),1], sorted_arr[len(sorted_arr)-1, 0], sorted_arr[len(sorted_arr)-1, 1] )
plt.quiver(sorted_arr[int(len(sorted_arr)/2),0],sorted_arr[int(len(sorted_arr)/2),1], -sorted_arr[len(sorted_arr)-1, 1], sorted_arr[len(sorted_arr)-1, 0] )
plt.show()

#2 a,b,c
AAA = wiPCA(iris.data, 2)[0]
plt.scatter(AAA[:,0], AAA[:,1], c=iris.target)
plt.show()

pca = PCA(n_components=2)
X_r = pca.fit(iris.data).transform(iris.data)
# print(X_r)
plt.scatter(X_r[:,0], X_r[:,1], c=iris.target)
plt.show()

# 3
#a, b, d
AAA= wiPCA(digits.data, 2)[0]
plt.scatter(AAA[:,0], AAA[:,1], c=digits.target, alpha=0.6)
plt.show()

#c
pca = PCA()
pca.fit(digits.data)
cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumsum)
plt.show()

#e
error_record= []
for i in range(0, 64):
    pca = PCA(n_components=i).fit(digits.data)
    data_reduced = pca.fit_transform(digits.data)
    data_original = pca.inverse_transform(data_reduced)
    loss = np.linalg.norm((digits.data - data_original), None)
    error_record.append(loss)

plt.plot(error_record)
plt.show()