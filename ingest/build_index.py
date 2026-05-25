"""
ingest/build_index.py

Loads all HR policy documents from the /docs folder,
chunks them, embeds them using OpenAI, and saves a FAISS index to /data.

Run this once after adding documents:
    python -m ingest.build_index

Re-run it whenever you add or update files in /docs.
"""

import os
import json
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from openai import OpenAI
import faiss
import numpy as np

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

DOCS_DIR = Path(os.getenv("DOCS_DIR", "./docs"))
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "./data/faiss.index"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", "./data/metadata.json"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_documents(docs_dir: Path) -> List[Document]:
    docs = []
    loaders = {
        "**/*.pdf": PyPDFLoader,
        "**/*.docx": Docx2txtLoader,
        "**/*.txt": TextLoader,
    }
    for glob_pattern, loader_cls in loaders.items():
        for file_path in docs_dir.glob(glob_pattern):
            try:
                log.info(f"Loading: {file_path.name}")
                loader = loader_cls(str(file_path))
                file_docs = loader.load()
                for doc in file_docs:
                    doc.metadata["source_file"] = file_path.name
                    doc.metadata["file_type"] = file_path.suffix.lower()
                docs.extend(file_docs)
            except Exception as e:
                log.warning(f"Could not load {file_path.name}: {e}")

    log.info(f"Loaded {len(docs)} pages total")
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    log.info(f"Created {len(chunks)} chunks")
    return chunks


def embed_chunks(chunks: List[Document]) -> np.ndarray:
    texts = [chunk.page_content for chunk in chunks]
    log.info(f"Embedding {len(texts)} chunks using {EMBEDDING_MODEL}...")

    all_embeddings = []
    batch_size = 100  # OpenAI allows up to 2048 per request, 100 is safe

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        batch_embeddings = [r.embedding for r in response.data]
        all_embeddings.extend(batch_embeddings)
        log.info(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.array(all_embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info(f"FAISS index built — {index.ntotal} vectors, dim={dim}")
    return index


def save_artifacts(index: faiss.Index, chunks: List[Document]):
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    log.info(f"FAISS index saved → {FAISS_INDEX_PATH}")

    metadata = [
        {
            "id": i,
            "text": chunk.page_content,
            "source_file": chunk.metadata.get("source_file", "unknown"),
            "page": chunk.metadata.get("page", 0),
        }
        for i, chunk in enumerate(chunks)
    ]

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    log.info(f"Metadata saved → {METADATA_PATH} ({len(metadata)} chunks)")


def main():
    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        log.error(f"No documents found in {DOCS_DIR}/. Add your HR policy files first.")
        return

    log.info("=== PensionBox HR Bot — Building FAISS Index ===")

    docs = load_documents(DOCS_DIR)
    if not docs:
        log.error("No documents could be loaded. Check file formats (PDF, DOCX, TXT only).")
        return

    chunks = chunk_documents(docs)
    embeddings = embed_chunks(chunks)
    index = build_faiss_index(embeddings)
    save_artifacts(index, chunks)

    log.info("=== Done. You can now start the API server. ===")


if __name__ == "__main__":
    main()
