import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from processingutils import processing
nc = pickle.load(open('al2/name_cluster.pickle',"rb"))
oc = pickle.load(open('al2/origin.pickle',"rb"))
def theme(cluster1,number):
    cluster = []
    for item in cluster1:
        processed = processing(item)
        #print(processed)
        cluster.append(processed)
    tfidf = TfidfVectorizer()
    tfidf.fit_transform(cluster)
    extra_stop_words= ["one","use","later","power","new"]
    themes = list(zip(tfidf.get_feature_names(),tfidf.idf_))


    themes.sort(key=sorts)
    them = [text for text,idf in themes]
    for item in extra_stop_words:
        while item in them:
            
            themes.pop(them.index(item))
            them.pop(them.index(item))


    return [text for text,idf in themes[:number]]
def sorts(item):
    return item[1]
print(nc[3])
print(theme(oc[3],4))