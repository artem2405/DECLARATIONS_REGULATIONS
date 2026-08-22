import csv
import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv(Path(__file__).parent / ".env")


def _reg_text(r):
    return "\n".join(filter(None, [
        r.get("code", ""),
        r.get("description", ""),
        r.get("explanation", ""),
        r.get("notes", ""),
    ]))


def _decl_text(d):
    return "\n".join(filter(None, [
        d.get("G31_1", ""),
        d.get("desc_extention", ""),
        d.get("retrieved_context", ""),
    ]))


def rerank_regulations(data_dir="./data", out_dir="./out"):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    declarations = json.loads((data_dir / os.getenv("DECS_DATA_OUT_NAME")).read_text(encoding="utf-8"))
    regulations = []
    with open(data_dir / os.getenv("REGULATIONS_JSONL"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                regulations.append(json.loads(line))

    model_name = os.getenv("RERANKER_MODEL")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    if device == "cuda":
        model.half()

    reg_texts = [_reg_text(r) for r in regulations]
    batch_size = int(os.getenv("BATCH_SIZE"))

    rows = []
    for i, decl in enumerate(declarations, 1):
        print(f"rerank {i}/{len(declarations)} {decl['declaration_id']}")
        pairs = [[_decl_text(decl), t] for t in reg_texts]
        scores = []
        with torch.no_grad():
            for j in range(0, len(pairs), batch_size):
                inputs = tokenizer(
                    pairs[j:j + batch_size],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = model(**inputs, return_dict=True).logits.view(-1).float()
                scores.extend(torch.sigmoid(logits).cpu().tolist())
        results = [
            {"regulation_id": r["regulation_id"], "score": float(s)}
            for r, s in zip(regulations, scores)
        ]
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:10]
        for rank, item in enumerate(results, 1):
            rows.append((decl["declaration_id"], rank, item["regulation_id"], item["score"]))

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["declaration_id", "rank", "regulation_id", "score"])
        w.writerows(rows)
    print(f"saved {len(rows)} rows")
