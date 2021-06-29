
from processingutils import processing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer
import random
import pandas as pd
import pickle
import numpy as np

df=pd.read_csv('al2/superheroes_nlp_dataset.csv')
df= df[["history_text","overall_score","name"]].dropna()
SuperHeroText = df['history_text']
SuperHeroScore = df['overall_score']
names = df['name']


# First vectorize text

text_data = []
x=0
texts=[]
print(SuperHeroText[0])

for text in SuperHeroText:
    print(text)
    processed = processing(text)
    texts.append(processed)
    print(processed)

tfid =TfidfVectorizer()
vectors = tfid.fit_transform(texts)
pickle.dump(tfid,open('al2/vectorizer.pickle','wb'))
pickle.dump(texts,open('al2/texts.pickle','wb'))
pickle.dump(vectors,open('al2/vectors.pickle','wb'))

