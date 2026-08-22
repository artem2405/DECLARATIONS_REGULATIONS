import json
import os
from pathlib import Path

import faiss
import torch
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).parent / ".env")


def make_vectors(data_dir="./data"):
    data_dir = Path(data_dir)
    index_path = data_dir / os.getenv("INDEX_NAME")
    chunks_path = data_dir / os.getenv("CHUNKS_NAME")
    if index_path.exists() and chunks_path.exists():
        print("vectors already exist, skip")
        return

    text = (data_dir / "tnved_knowledge.txt").read_text(encoding="utf-8")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),
    ).split_text(text)
    chunks = [c.strip() for c in chunks if c.strip()]

    model = SentenceTransformer(os.getenv("EMBEDDING_MODEL"), device='cuda' if torch.cuda.is_available() else 'cpu')
    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        batch_size=int(os.getenv("BATCH_SIZE")),
        show_progress_bar=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    print(f"saved {len(chunks)} chunks")