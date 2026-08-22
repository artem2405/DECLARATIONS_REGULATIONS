import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def extract_declarations(data_dir="./data"):
    data_dir = Path(data_dir)
    rows = []
    with open(data_dir / os.getenv("DECLARATIONS_JSONL"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "declaration_id": d["declaration_id"],
                "G31_1": d.get("G31_1", ""),
                "desc_extention": d.get("desc_extention", ""),
                "retrieved_context": "",
            })
    (data_dir / os.getenv("DECS_DATA_OUT_NAME")).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"extracted {len(rows)} declarations")
