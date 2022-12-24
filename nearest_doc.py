from pymongo import MongoClient
import openai
import numpy as np
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
    # X = rows of data X 1536
    X = np.array(X)
    print("Aggregated Data")

    text = input('Input Question or Search Query: ')
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    embedding = np.array(response['data'][0]['embedding'])

    # num rows of data X 1
    o = X @ embedding   
    l  = [(v, i) for i, v in enumerate(o)]
    l.sort(key=lambda x: x[0])

    # Print top 10 relevant pieces of text
    print("\nRelevant articles:")
    for i in range(10):
        print(data[l[-(i+1)][1]]["title"])