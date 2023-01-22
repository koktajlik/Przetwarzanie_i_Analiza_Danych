from sklearn import datasets, mixture
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import jaccard_score
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

iris = datasets.load_iris()
X, Y = iris.data, iris.target

#2
#najblizsze sasiedztwa
cl_single = AgglomerativeClustering(n_clusters=3, linkage = 'single').fit(X)
#srednie polaczenia
cl_average = AgglomerativeClustering(n_clusters=3, linkage = 'average').fit(X)
#najdalsze polaczenia
cl_complete = AgglomerativeClustering(n_clusters=3, linkage = 'complete').fit(X)
#ward
cl_ward = AgglomerativeClustering(n_clusters=3, linkage = 'ward').fit(X)

#Każda wartośc zostaje przypisana do jednoelementowego klastra, potem dwa najbliższe są ze sobą łączone
# i w każdej kolejnej iteracji klastry te są łączone by kończowo stworzyć jeden wspólny
# warunek łączenia w każdej metodzie jest inny:
# najbliższe - Odległość między klastrami jest minimalną odległością między obserwacją w jednym klastrze a obserwacją w innym klastrze
# średnie - odległość między klastrami jest średnią odległością między obserwacją w jednym klastrze a obserwacją w innym klastrze
# najdalsze - odległość między klastrami jest maksymalną odległością między obserwacją w jednym klastrze a obserwacją w innym klastrze
# warda - odległość między klastrami jest sumą kwadratów odchyleń od punktów do centroidów


#3
def find_perm(clusters, Y_real, Y_pred):
    perm = []
    for i in range(clusters):
        idx = Y_pred == i #porównuje zawartość Y_pred z danym i i zwraca tablicę typu bool
        new_label = stats.mode(Y_real[idx])[0] #zwraca najczęściej występującą wartość (jeśli nie ma powtórzeń to najmniejszą), wyników będzie tyle ile jest targetów
        perm.append(new_label) #dodaje do tablicy wynik poprzedniej linii
    return [perm[label] for label in Y_pred] #zwraca tablicę przynależności pobierając wartości z perma przy użyciu wartości Y_pred jako indeksów
                                            #dzięki czemu jest możliwość podmiany wartości by podział na klasy zgadzał się z tym co jest w Y_real

#4
cl = cl_single, cl_average, cl_complete, cl_ward
for i in cl:
    print(jaccard_score(Y, find_perm(len(Y), Y, i.labels_), average=None))

# indeks Jaccarda to wartość od 0 do 1 określająca podobieństwo dwóch zbiorów (obliczona ze stosunku iloczynu zbiorów do ich sumy).
# 1 oznacza, że zbiory posiadają idenyczne elementy, a 0, że nie posiadają żadnych wspólnych.

#5
def findDiff(l1, l2):
    val = []
    for i in range(len(l1)):
        val.append(l1[i] == l2[i])
    return val
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=Y)
plt.show()

kmeans = KMeans(3)
label = kmeans.fit_predict(X_reduced)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=find_perm(len(Y), Y, label))
plt.show()
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=findDiff(find_perm(len(Y), Y, label), Y))
plt.show()

gaus = mixture.GaussianMixture(3)
label = gaus.fit_predict(X_reduced)
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=find_perm(len(Y), Y, label))
plt.show()
plt.scatter(X_reduced[:,0], X_reduced[:,1], c=findDiff(find_perm(len(Y), Y, label), Y))
plt.show()

#7
met = ['single','average','complete','ward']
for i in met:
    Z = linkage(X, i)
    plt.figure(figsize=(25, 10))
    dendrogram(Z)
    plt.title(i)
    plt.axhline()
    plt.show()
#9
to_rem = ['animal','type']
csv = pd.read_csv('../data/zoo.csv').drop(to_rem, axis='columns')
met = ['single','average','complete','ward']
for i in met:
    Z = linkage(csv, i)
    plt.figure(figsize=(25, 10))
    dendrogram(Z)
    plt.axhline()
    plt.show()
