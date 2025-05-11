# Import necessary libraries for handling embeddings, vector storage, and documents
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
import pandas as pd

# Load the restaurant review data from CSV file
df = pd.read_csv("realistic_restaurant_reviews.csv")

# Initialize the embedding model using Ollama's mxbai-embed-large
# This model converts text into numerical vectors for similarity searches
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Define where the FAISS database will be stored on disk
db_location = "./faiss_langchain_db"

# Check if the vector database already exists on disk
# This prevents recreating the database every time the script runs
add_documents = not os.path.exists(db_location)

if add_documents:
    # If database doesn't exist, create it from scratch
    documents = []

    # Loop through each review in the CSV file
    for i, row in df.iterrows():
        # Create a Document object for each review
        # Combine Title and Review text as the main content
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},  # Store extra info
            id=str(i)  # Unique identifier for each document
        )
        documents.append(document)

    # Create a new FAISS vector store from the documents
    # This will convert all documents to vectors using the embedding model
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    # Save the vector store to disk for future use
    vector_store.save_local(db_location)
else:
    # If database already exists, load it from disk
    # This is much faster than recreating all embeddings
    vector_store = FAISS.load_local(
        db_location,
        embeddings,
        allow_dangerous_deserialization=True  # Required security flag for FAISS
    )

# Create a retriever object that can search the vector store
# k=5 means it will return the 5 most similar documents for each query
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

print("Vector store setup complete!")