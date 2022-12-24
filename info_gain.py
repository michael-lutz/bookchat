from pymongo import MongoClient
import openai
import numpy as np
import os

openai.api_key = os.getenv("OPENAI_KEY")

NUM_DOCUMENTS = 10

def get_database():
    CONNECTION_STRING = os.getenv("CONNECTION_STRING")
    client = MongoClient(CONNECTION_STRING)
    return client['chatassist-development']
  
# This is added so that many files can reuse the function get_database()
N_CLUSTERS = 20
if __name__ == "__main__": 
    # Do the same as in nearest doc
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

    # Run something similar to OMP
    docs = []
    indices = []
    for i in range(NUM_DOCUMENTS):
        if len(indices) == 0:
            # num rows of data X 1
            o = X @ embedding
            indices.append(np.argmax(o))
            docs.append(data[indices[-1]])
        else:
            # Remove orthogonal components and remax
            xi = X[indices[-1]]
            X = X - np.expand_dims((X @ xi), 1) @ xi[None, ...]
            o = X @ embedding
            indices.append(np.argmax(o))
            docs.append(data[indices[-1]])

    # Print top 10 relevant pieces of text
    print("\nRelevant articles:")
    for i in range(NUM_DOCUMENTS):
        print(docs[i]["title"])