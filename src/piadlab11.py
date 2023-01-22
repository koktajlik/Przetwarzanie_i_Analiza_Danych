from sklearn import datasets, model_selection, metrics
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from matplotlib import rcParams
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def findDiff(l1, l2):
    return [l1_v == l2_v for l1_v, l2_v in zip(l1, l2)]

X, Y = datasets.make_classification(
    n_samples=1600,
    n_features=2,
    n_clusters_per_class=1,
    n_informative=2,
    n_redundant=0,
    n_classes=4
)
# plt.scatter(X[:, 0], X[:, 1], c = Y)
# plt.show()

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.5)

svm_linear1 = OneVsOneClassifier(SVC(kernel='linear', probability=True))
svm_rbf1 = OneVsOneClassifier(SVC(kernel='rbf', probability=True))
log_reg1 = OneVsOneClassifier(LogisticRegression())
perc1 = OneVsOneClassifier(Perceptron())

svm_linear2 = OneVsRestClassifier(SVC(kernel='linear', probability=True))
svm_rbf2 = OneVsRestClassifier(SVC(kernel='rbf', probability=True))
log_reg2 = OneVsRestClassifier(LogisticRegression())
perc2 = OneVsRestClassifier(Perceptron())

met = [svm_linear1, svm_rbf1, log_reg1, perc1, svm_linear2, svm_rbf2, log_reg2, perc2]
s = 4

pred_tab, ans = [], []
for i, val in enumerate(met):
    val.fit(X_train, y_train)
    pred = val.predict(X_test)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
    ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    ax1.title.set_text(f'oczekiwane')
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=pred)
    ax2.title.set_text(f'obliczone')
    ax3.scatter(X_test[:, 0], X_test[:, 1], c=findDiff(y_test, pred))
    ax3.title.set_text(f'różnice')
    plt.suptitle(val)
    plt.show()

    v1 = metrics.accuracy_score(y_test, pred)
    v2 = metrics.recall_score(y_test, pred, average='weighted')
    v3 = metrics.precision_score(y_test, pred, average='weighted')
    v4 = metrics.f1_score(y_test, pred,average='weighted')

    pred_decision = val.decision_function(X_test)

    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(s):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, pred_decision[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    filtered_vals = [v for _, v in roc_auc.items()]
    v5 = sum(filtered_vals) / len(filtered_vals)
    ans.append([v1, v2, v3, v4, v5])

    plt.figure()
    colors = ['blue', 'orange', 'olive','magenta']
    for i in range(s):
        plt.plot(fpr[i], tpr[i], color=colors[i], label=f'krzywa ROC (AUC = %0.2f) dla klasy {i}' % roc_auc[i])
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

    values = np.linspace(-4, 4, 100)
    tab = [[i, j] for i in values for j in values]
    predictedMeshGrid = val.predict(tab)
    XX, YY = np.meshgrid(values, values)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=pred)
    plt.contour(XX, YY, np.reshape(predictedMeshGrid, (100, 100), order='F'))
    plt.show()

avgRes = np.average(np.reshape(ans,(-1, 40)), axis=0)
avgRes = np.transpose(np.reshape(avgRes, (-1,5)))
index = ['accuracy_score', 'recall_score', 'precision_score', 'f1_score', 'roc_auc',]
columns = ['OVO SVC linear','OVO SVC kernel','OVO LogReg','OVO Percep','OVR SVC linear','OVR SVC kernel','OVR LogReg','OVR Percep']
rcParams.update({'figure.autolayout': True})
df = pd.DataFrame(avgRes, columns=columns, index=index)
df.plot(kind="bar",y=columns)
plt.legend(loc="lower right")
plt.show()