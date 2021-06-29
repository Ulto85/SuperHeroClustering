import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from processingutils import processing
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
tfid = pickle.load(open('al2/vectorizer.pickle','rb'))
df=pd.read_csv('al2/superheroes_nlp_dataset.csv')
df= df[["history_text","overall_score","name"]].dropna()
SuperHeroText = df['history_text']
score= df['overall_score']
SuperHeroText= list(df['history_text'])
cluster=KMeans(n_clusters=7)
names = list(df['name'])
print(score.shape[0])
#print(names.shape[0])
vectors = pickle.load(open('al2/vectors.pickle','rb'))
print(vectors.shape[0])   
vectors1=TruncatedSVD(n_components=7).fit_transform(vectors)
numbers=  TSNE().fit_transform(vectors1)
print(numbers.shape[0])
cluster.fit(numbers)

vectors2 =tfid.transform([processing('Bob was born in gotham but made is way to england'),processing('Bob liked noodles')])
vectors3=TruncatedSVD(n_components=7).fit_transform(vectors2)
numbers1=  TSNE().fit_transform(vectors3)
cluster.predict(numbers1)
