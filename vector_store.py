import os
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# Project paths
DOCUMENT_FOLDER = "dataset/documents"
CHROMA_FOLDER = "chroma_db"

# Load Sentence Transformer model
print("Loading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load documents
documents = []
document_ids = []
metadatas = []

for filename in sorted(os.listdir(DOCUMENT_FOLDER)):
    if filename.endswith(".txt"):
        file_path = os.path.join(DOCUMENT_FOLDER, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read().strip()

        documents.append(content)
        document_ids.append(filename)

        metadatas.append({
            "source": filename,
            "domain": "Healthcare"
        })

# Generate embeddings
print("Generating vector embeddings...")

embeddings = model.encode(documents, convert_to_numpy=True)

# Save embeddings in NumPy format
np.save("dataset/embeddings.npy", embeddings)

# Create persistent ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_FOLDER)

# Create collection
collection = client.get_or_create_collection(
    name="healthcare_documents"
)

# Store documents and embeddings
collection.add(
    ids=document_ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)

print("\nVector embeddings generated successfully!")
print("Total documents:", len(documents))
print("Embedding shape:", embeddings.shape)
print("Documents stored in ChromaDB successfully!")