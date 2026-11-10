
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer model
print("Loading Sentence Transformer model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="healthcare_documents"
)

# Load user queries
queries_df = pd.read_csv("dataset/queries.csv")

results = []

print("\nPerforming Similarity Search...\n")

# Process each query
for _, row in queries_df.iterrows():

    query_id = row["query_id"]
    query_text = row["query"]

    # Generate query embedding
    query_embedding = model.encode(query_text).tolist()

    # Retrieve Top-5 relevant documents
    search_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    # Store retrieved results
    for rank in range(len(search_results["ids"][0])):

        results.append({
            "query_id": query_id,
            "query": query_text,
            "rank": rank + 1,
            "document_id": search_results["ids"][0][rank],
            "document": search_results["documents"][0][rank],
            "distance": search_results["distances"][0][rank]
        })

    print("Query:", query_text)
    print("Top 5 documents retrieved successfully!\n")

# Save results to CSV
results_df = pd.DataFrame(results)

results_df.to_csv(
    "dataset/retrieved_results.csv",
    index=False
)

print("Similarity Search Completed Successfully!")
print("Total queries:", len(queries_df))
print("Total retrieved results:", len(results_df))
print("Results saved in dataset/retrieved_results.csv")
