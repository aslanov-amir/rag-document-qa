"""
main.py — FastAPI server for the RAG Document Q&A System.

Provides a POST /query endpoint that:
1. Takes a user question
2. Embeds it using the same model as ingestion
3. Retrieves the most relevant chunks from ChromaDB
4. Sends the question + retrieved chunks to the Gemini API
5. Returns the grounded answer

Also serves the static HTML frontend.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

# --- Configuration ---
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3  # number of chunks to retrieve

# --- Initialize components ---

# Load the same embedding model used during ingestion
model = SentenceTransformer(EMBEDDING_MODEL)

# Connect to the existing ChromaDB database
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

# Configure the Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Create a .env file with: GEMINI_API_KEY=your_key_here"
    )
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-2.5-pro")

# --- FastAPI app ---
app = FastAPI(
    title="RAG Document Q&A",
    description="Ask questions about ingested PDF documents.",
)

# Serve the static frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")


class QueryRequest(BaseModel):
    """Schema for incoming query requests."""
    question: str


class QueryResponse(BaseModel):
    """Schema for outgoing query responses."""
    answer: str
    sources: list[str]


@app.get("/")
def serve_frontend():
    """Serve the HTML frontend when the user visits the root URL."""
    return FileResponse("static/index.html")


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Answer a question using retrieved document context.

    Steps:
        1. Embed the user's question using the same model as ingestion.
        2. Query ChromaDB for the top-k most similar chunks.
        3. Build a prompt with the retrieved chunks as context.
        4. Send the prompt to Gemini and return the answer.

    Args:
        request: A JSON body containing a 'question' field.

    Returns:
        A JSON object with 'answer' and 'sources' fields.
    """
    # Step 1: Embed the question
    question_embedding = model.encode(request.question).tolist()

    # Step 2: Retrieve relevant chunks from ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K,
    )

    # Extract the chunk texts
    retrieved_chunks = results["documents"][0]  # list of strings
    sources = retrieved_chunks  # we'll return these so the user can see what was retrieved

    # Step 3: Build the prompt for Gemini
    context = "\n\n---\n\n".join(retrieved_chunks)
    prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the
provided context. If the context does not contain enough information to answer
the question, say "I don't have enough information in the provided documents to
answer this question."

Context:
{context}

Question: {request.question}

Answer:"""

    # Step 4: Call Gemini
    response = gemini_model.generate_content(prompt)
    answer = response.text

    return QueryResponse(answer=answer, sources=sources)