from pymongo import MongoClient
import openai
import numpy as np
import os
import time
import requests
from sentence_transformers import SentenceTransformer

openai.api_key = os.getenv("OPENAI_KEY")



text_samples = [
    '-held territory after a long war.',
]

input = "What were the effects of hoover's presidency?"

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

start = time.time()
embeddings = model.encode(text_samples)
input_emb = model.encode(input)
print(embeddings.shape)

print(embeddings @ input_emb)

end = time.time()
print(end - start)

# Using Open AI Embedding API
"""
embedding_samples = []
start = time.time()
for i in text_samples:
    embedding_samples.append(np.array(openai.Embedding.create(input=i, model="text-embedding-ada-002")['data'][0]['embedding']))

end = time.time()
print(end - start)



query_embedding = np.array(openai.Embedding.create(input=input, model="text-embedding-ada-002")['data'][0]['embedding'])


similarity = np.array(embedding_samples) @ query_embedding
print(similarity)"""

# Using Hugging Face Similarity API
"""
start = time.time()
api_token = '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'

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/msmarco-distilbert-base-tas-b"
headers = {"Authorization": f"Bearer {api_token}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

data = query(
    {
        "inputs": {
            "source_sentence": query_,
            "sentences": text_samples
        }
    }
)
end = time.time()
print(end - start)

print(data)"""