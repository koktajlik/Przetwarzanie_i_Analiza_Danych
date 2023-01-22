from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

class Knn:
    def __init__(self, n_neighbors = 1, use_KDTree = False):
        self.n_neighbors = n_neighbors
        self.use_kDTree = use_KDTree

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train

    def predict(self, X_test):
        ans = []
        for cnt, val in enumerate(X_test):
            tmp = []
            for cnt2, val2 in enumerate(self.X):
                tmp.append((val[0]-val2[0])**2+(val[1]-val2[1])**2)
            index = np.argsort(tmp)
            cnt = 0
            [cnt := cnt + self.y[index[i]] for i in range(self.n_neighbors)]
            ans.append(0) if self.n_neighbors - cnt < cnt else ans.append(1)
        return ans

    def score(self, X, y):
        self.y_pred = self.predict(X)
        self.y_pred2 = y
        return np.mean(self.y_pred == self.y_pred2)

# 3.1
X, y = make_classification(
    n_samples = 100,
    n_features = 2,
    n_informative = 2,
    n_repeated = 0,
    n_redundant = 0,
    random_state = 3
)

# 3.2
val = np.linspace(-3,3, 100)
tab = [[i, j] for i in val for j in val]

obj = Knn(1);
obj.fit(X,y)
test = obj.predict(tab)
print(f'Klasyfikacja: {test}')

# 3.3
Xv, Yv = np.meshgrid(val, val)
plt.scatter(X[:,0],X[:,1],c=y)
plt.contour(Xv, Yv, np.reshape(test, (100,100), order='F'))
plt.show()

# 3.4
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_prediction = knn.predict(X_test)
score = np.mean(y_test == y_prediction)
print(f'Klasyfikacja iris: {y_prediction}\nDokładność dopasowania: {score}')

# 3.5
pca = PCA(n_components=2)
y = iris.target
X_r = pca.fit(iris.data).transform(iris.data)

plt.scatter(X_r[:,0],X_r[:,1],c=y)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_r, y)
y_prediction = knn.predict(tab)
plt.contour(Xv, Yv, np.reshape(y_prediction, (100,100), order='F'))
plt.show()

#3.6
def leave_one_out(neighborsTest = 2, X = None, y = None):
    ans = []
    knn = KNeighborsClassifier(n_neighbors=neighborsTest)
    for j in range(len(X)):
        trainData, trainTarget, testSample = [], [], []
        for k in range(len(X)):
            if k is j:
                testSample.append(X[k])
            else:
                trainData.append(X[k])
                trainTarget.append(y[k])
        knn.fit(trainData, trainTarget)
        y_prediction = knn.predict(testSample)
        ans.append(1) if y_prediction == y[j] else ans.append(0)
    return np.sum(ans)/len(ans)
print(f'Kroswalidacja - dopasowanie: {leave_one_out(2, iris.data, iris.target)}')
