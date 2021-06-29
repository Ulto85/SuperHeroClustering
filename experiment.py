from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD
df=pd.read_csv('al2/superheroes_nlp_dataset.csv')
df= df[["history_text","overall_score","name"]].dropna()
SuperHeroText = df['history_text']
score= df['overall_score']
SuperHeroText= list(df['history_text'])
names = list(df['name'])
print(score.shape[0])
#print(names.shape[0])
vectors = pickle.load(open('al2/vectors.pickle','rb'))
print(vectors.shape[0])
tfid = pickle.load(open('al2/vectorizer.pickle','rb'))
X = np.array(list(zip(tfid.idf_,score)))
'''
cluster= KMeans(n_clusters=10)
cluster.fit(vectors)
print(cluster.labels_)
cluster_pick = np.random.choice(len(set(cluster.labels_)))
names = list(names)
lists= []
print(names)
print(len(names))
for x in range(9):
    clust=[]
    for i in np.where(cluster.labels_ ==x)[0]:
        print(issubclass)
        print(names[i])
        clust.append(names[i])
    lists.append(clust)
print(lists[0])
print(numbers)
plt.scatter(numbers[:, 0], numbers[:, 1], marker = "x", c = cluster.labels_)
plt.show()
'''

def find_elbow_curve(vectors):
    noice = []
    inertia = []
    for x in range(1,20):
        cluster= KMeans(n_clusters=x)
        
        vectors1=TruncatedSVD(n_components=x).fit_transform(vectors)
        numbers=  TSNE().fit_transform(vectors1)
        cluster.fit(numbers)
        noice.append(sum(np.min(cdist(numbers, cluster.cluster_centers_,
                                   'euclidean'),axis=1)) / X.shape[0])
        inertia.append(cluster.inertia_)
    return (noice,inertia)
#elbow curve is 8
#thing = find_elbow_curve(vectors)
#plt.plot(range(1,20),thing[1],'bx-')
#plt.show()

cluster= KMeans(n_clusters=7)
        
vectors1=TruncatedSVD(n_components=7).fit_transform(vectors)
numbers=  TSNE().fit_transform(vectors1)
print(numbers.shape[0])
cluster.fit(numbers)

name_cluster= [ ]
print(names[72])
for x in range(7):
    clust=[]
    for i in np.where(cluster.labels_ ==x)[0]:
        print(issubclass)
        print(names[i])
        clust.append(names[i])
    name_cluster.append(clust)
origin_cluster= [ ]
for x in range(7):
    clust=[]
    for i in np.where(cluster.labels_ ==x)[0]:
        print(issubclass)
        
        clust.append(SuperHeroText[i])
    origin_cluster.append(clust)
pickle.dump(name_cluster,open("name_cluster.pickle",'wb'))
pickle.dump(origin_cluster,open("origin.pickle",'wb'))
plt.scatter(numbers[:,0],numbers[:,1],c=cluster.labels_)
plt.show()
