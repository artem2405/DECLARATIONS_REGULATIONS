import json
import os
from pathlib import Path

import faiss
import numpy as np
import torch
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel

load_dotenv(Path(__file__).parent / ".env")


def retrieve(query, index, chunks, model):
    vec = model.encode([query], batch_size=12, max_length=8192)["dense_vecs"]
    vec = np.asarray(vec, dtype=np.float32)
    faiss.normalize_L2(vec)
    _, ids = index.search(vec, int(os.getenv("TOP_K_FRAGMENTS")))
    return [chunks[i] for i in ids[0] if i >= 0]


def retrieve_context(data_dir="./data"):
    data_dir = Path(data_dir)
    declarations = json.loads((data_dir / os.getenv("DECS_DATA_OUT_NAME")).read_text(encoding="utf-8"))
    chunks = json.loads((data_dir / os.getenv("CHUNKS_NAME")).read_text(encoding="utf-8"))
    index = faiss.read_index(str(data_dir / os.getenv("INDEX_NAME")))
    model = BGEM3FlagModel(
        os.getenv("EMBEDDING_MODEL"),
        use_fp16=torch.cuda.is_available(),
    )

    for decl in declarations:
        product_profile = f"{decl['G31_1']}\n{decl['desc_extention']}"
        knowledge_chunks = retrieve(query=product_profile, index=index, chunks=chunks, model=model)
        decl["retrieved_context"] = "\n\n".join(knowledge_chunks)

    (data_dir / os.getenv("DECS_DATA_OUT_NAME")).write_text(
        json.dumps(declarations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"retrieved context for {len(declarations)} declarations")
