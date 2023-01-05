import time
from sentence_transformers import SentenceTransformer

with open('scrapedtext.txt', 'r') as f:
    BOOK_TEXT = f.read()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

start = time.time()

paragraphs = []
i = 0
while i + 500 < len(BOOK_TEXT):
    paragraphs.append(BOOK_TEXT[i:i+1100])

    i += 1000

embeddings = model.encode(paragraphs)
input_emb = model.encode('Who is herbert hoover?')
print(embeddings.shape)

print(embeddings @ input_emb)

end = time.time()
print(end - start)