"""
ingest.py — PDF ingestion pipeline for the RAG Document Q&A System.

This script reads PDF files from the /data folder, splits the text into
overlapping chunks, generates embeddings using sentence-transformers,
and stores everything in a local ChromaDB vector database.

Run this once (or whenever you add new PDFs) before starting the FastAPI server.
"""

import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb

# --- Configuration ---
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # characters of overlap between consecutive chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Read a PDF file and return all its text as a single string.

    Uses PyMuPDF (fitz) to open the PDF and extract text from each page.
    Pages are joined with newlines.

    Args:
        pdf_path: Full path to the PDF file.

    Returns:
        The full text content of the PDF as one string.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a long text string into overlapping chunks.

    Each chunk is approximately chunk_size characters long. Consecutive chunks
    overlap by 'overlap' characters so that sentences at chunk boundaries appear
    in both neighboring chunks.

    Args:
        text: The full text to split.
        chunk_size: Target number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        A list of text chunks (strings).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # step back by 'overlap' characters
    return chunks


def ingest():
    """
    Main ingestion function. Reads all PDFs in DATA_DIR, chunks them,
    embeds them, and stores everything in ChromaDB.
    """
    # Step 1: Load the embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Model loaded: {EMBEDDING_MODEL} (produces {model.get_sentence_embedding_dimension()}-dimensional vectors)")

    # Step 2: Initialize ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if it exists (so re-running ingest is clean)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection — starting fresh.")
    except Exception:
        pass  # Collection didn't exist, that's fine

    collection = client.create_collection(name=COLLECTION_NAME)
    print(f"Created ChromaDB collection: '{COLLECTION_NAME}'")

    # Step 3: Process each PDF in the data folder
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in '{DATA_DIR}/'. Add a PDF and run again.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.")

    all_chunks = []
    all_ids = []
    all_metadatas = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATA_DIR, pdf_file)
        print(f"\nProcessing: {pdf_file}")

        # Extract text
        text = extract_text_from_pdf(pdf_path)
        print(f"  Extracted {len(text)} characters of text")

        # Chunk
        chunks = chunk_text(text)
        print(f"  Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

        # Track chunks with IDs and metadata
        for i, chunk in enumerate(chunks):
            chunk_id = f"{pdf_file}__chunk_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({"source": pdf_file, "chunk_index": i})

    # Step 4: Embed all chunks
    print(f"\nEmbedding {len(all_chunks)} total chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")

    # Step 5: Store in ChromaDB
    print("Storing in ChromaDB...")
    collection.add(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings.tolist(),
        metadatas=all_metadatas,
    )

    print(f"\nDone! {collection.count()} chunks stored in '{CHROMA_DIR}/'.")
    print("You can now start the FastAPI server with: uvicorn main:app --reload")


if __name__ == "__main__":
    ingest()