import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, naive_bayes, discriminant_analysis, neighbors, svm, tree, model_selection, metrics
import pandas as pd
from sklearn.metrics import roc_curve
from matplotlib import rcParams
from itertools import chain
import time

def findDiff(l1, l2):
    val = []
    for i in range(len(l1)):
        val.append(l1[i] == l2[i])
    return val

#1, 2
X, Y = datasets.make_classification(n_classes=2, n_clusters_per_class=2, n_features=2, n_informative=2, n_repeated=0, n_redundant=0)
plt.scatter(X[:, 0], X[:, 1], c = Y)
plt.show()

#3
nb = naive_bayes.GaussianNB()
da = discriminant_analysis.QuadraticDiscriminantAnalysis()
ngb = neighbors.KNeighborsClassifier()
sv = svm.SVC(probability=True)
tr = tree.DecisionTreeClassifier()
arr = [nb, da, ngb, sv, tr]
ans = []
for i in range(100):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
    train_time, test_time = [], []
    #uczenie na zbiorze
    for a in arr:
        t1 = time.time()
        a.fit(X_train, y_train)
        t2 = time.time()
        train_time.append(t2 - t1)
    y_pred_tab = []
    #predykcja na zbiorze
    for a in arr:
        t1 = time.time()
        pr = a.predict(X_test)
        t2 = time.time()
        y_pred_tab.append(pr)
        test_time.append(t2 - t1)
    #miary jakości
    for ind, val in enumerate(y_pred_tab):
        v1 = metrics.accuracy_score(y_test, val)
        v2 = metrics.recall_score(y_test, val)
        v3 = metrics.precision_score(y_test, val)
        v4 = metrics.f1_score(y_test, val)
        v5 = metrics.roc_auc_score(y_test, val)
        ans.append([v1, v2, v3, v4, v5, train_time[ind], test_time[ind]])

    if i == 99:
        for ind, pred in enumerate(y_pred_tab):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,4))
            ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
            ax1.title.set_text(f'oczekiwane')
            ax2.scatter(X_test[:, 0], X_test[:, 1], c=pred)
            ax2.title.set_text(f'obliczone')
            ax3.scatter(X_test[:, 0], X_test[:, 1], c=findDiff(y_test, pred))
            ax3.title.set_text(f'różnice')
            fig.suptitle(arr[ind])
            plt.show()

            y_pred_proba = arr[ind].predict_proba(X_test)[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="ROC curve (area = %0.2f)" % ans[ind][4])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.show()

            val = np.linspace(-4, 4, 100)
            tab = [[i, j] for i in val for j in val]
            predictedMeshGrid = arr[ind].predict(tab)
            XX, YY = np.meshgrid(val, val)
            plt.scatter(X_test[:, 0], X_test[:, 1], c=pred)
            plt.contour(XX, YY, np.reshape(predictedMeshGrid, (100, 100), order='F'))
            plt.show()

avgRes = np.average(np.reshape(ans,(-1, 35)), axis=0)
avgRes = np.transpose(np.reshape(avgRes, (-1,7)))
avgRes[5:][:] = avgRes[5:][:] * 500 #zwiększenie wartości, by byly widoczne na wykresie
index = ['accuracy_score', 'recall_score', 'precision_score', 'f1_score', 'roc_auc', 'train_time', 'test_time']
columns = ['GaussianNB', 'QuadricD', 'KNeighbors', 'SVC', 'DecisionTr']
rcParams.update({'figure.autolayout': True})
df = pd.DataFrame(avgRes, columns=columns, index=index)
df.plot(kind="bar",y=columns)
plt.show()