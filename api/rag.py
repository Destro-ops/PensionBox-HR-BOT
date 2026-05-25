"""
api/rag.py

Core RAG logic:
1. Embed the user's question using OpenAI
2. Search FAISS for top-k relevant chunks
3. Build a prompt with the retrieved context
4. Call GPT-4o and return the answer + sources
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "./data/faiss.index"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", "./data/metadata.json"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K = int(os.getenv("TOP_K", 5))


class HRBotRAG:
    def __init__(self):
        self._index: faiss.Index = None
        self._metadata: List[Dict] = []
        self._client: OpenAI = None
        self._loaded = False

    def load(self):
        """Load all artifacts into memory. Called once at API startup."""
        if self._loaded:
            return

        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Run `python -m ingest.build_index` first."
            )

        log.info("Loading FAISS index...")
        self._index = faiss.read_index(str(FAISS_INDEX_PATH))

        log.info("Loading metadata...")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self._loaded = True
        log.info(f"RAG engine ready — {self._index.ntotal} chunks indexed.")

    def _embed(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL,
        )
        vec = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        return vec

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        query_vec = self._embed(query)
        scores, indices = self._index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self._metadata[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def answer(self, question: str, employee_name: str = "Employee") -> Dict:
        if not self._loaded:
            self.load()

        chunks = self.search(question)

        if not chunks:
            return {
                "answer": "I couldn't find relevant information in the HR policies. Please contact HR directly.",
                "sources": [],
            }

        # Build context
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {i}: {chunk['source_file']}, Page {chunk.get('page', 'N/A')}]\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        system_prompt = """You are PensionBox's internal HR assistant. You help employees understand company policies, processes, and HR-related questions.

Guidelines:
- Answer based ONLY on the provided policy excerpts. Do not make up information.
- Be concise, warm, and helpful. This is a colleague asking, not a customer.
- If the answer is only partially covered, give what you know and suggest contacting HR for the rest.
- Always mention which document the information came from.
- If a question is outside the provided context, say so clearly and direct them to HR.
- For sensitive topics (PIP, termination, legal matters), always recommend speaking directly with HR.
- Format your answer clearly for Slack — use bullet points where it helps readability."""

        user_message = f"""Employee question: {question}

Relevant policy excerpts:

{context}

Please answer the employee's question based on the above excerpts."""

        response = self._client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1024,
            temperature=0.2,
        )

        answer_text = response.choices[0].message.content

        # Deduplicate sources
        seen = set()
        sources = []
        for chunk in chunks:
            src = chunk["source_file"]
            if src not in seen:
                seen.add(src)
                sources.append(src)

        return {
            "answer": answer_text,
            "sources": sources,
        }


# Singleton — one instance shared across all API requests
rag_engine = HRBotRAG()
