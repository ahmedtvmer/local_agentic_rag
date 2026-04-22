import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

os.makedirs("./local_models/embedder", exist_ok=True)
os.makedirs("./local_models/reranker", exist_ok=True)

print("Fetching Embedding Model...")
SentenceTransformer('all-MiniLM-L6-v2').save("./local_models/embedder")

print("Fetching Cross-Encoder...")
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2').save("./local_models/reranker")
print("Local cache rebuilt.")