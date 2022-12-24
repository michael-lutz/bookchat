from pymongo import MongoClient
from sklearn.cluster import KMeans
import numpy as np
import openai
import os

openai.api_key = os.getenv("OPENAI_KEY")

def get_database():
    CONNECTION_STRING = os.getenv("CONNECTION_STRING")
    client = MongoClient(CONNECTION_STRING)
    return client['chatassist-development']
  
# This is added so that many files can reuse the function get_database()
N_CLUSTERS = 20
if __name__ == "__main__":   
    dbname = get_database()
    data = []
    X = []
    for post in dbname.Article.find():
        data.append(post)
        X.append(np.array(post["embedding"]))
    X = np.array(X)
    print("Aggregated Data")

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init=10).fit(X)
    print("Clustered Data")
    for l in range(N_CLUSTERS):
        print(f"Cluster {l}")
        for i, label in enumerate(kmeans.labels_):
            if l == label:
                print(f"\t{data[i]['title']}")
        print()
        
    print("Finding a cluster that most relates to a specific article")
    text = input('Input Question or Search Query: ')
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    embedding = np.array(response['data'][0]['embedding'])
    l = kmeans.predict(embedding[None, ...])[0]
    print(f"Related articles")
    for i, label in enumerate(kmeans.labels_):
        if l == label:
            print(f"\t{data[i]['title']}")
    print()