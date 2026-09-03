"""Inspect local messages JSONL files without exporting their content."""
import argparse
import hashlib
import json
from pathlib import Path

def inspect(path: Path):
    rows = 0
    valid = []
    invalid_lines = []
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for line_number, raw in enumerate(file, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            rows += 1
            try:
                row = json.loads(raw)
                messages = row["messages"]
                if not isinstance(messages, list) or len(messages) < 2:
                    raise ValueError("messages")
                if messages[-1]["role"] != "assistant":
                    raise ValueError("last role")
                for message in messages:
                    if message["role"] not in {"system", "user", "assistant"}:
                        raise ValueError("role")
                    if not isinstance(message["content"], str) or not message["content"].strip():
                        raise ValueError("content")
                prompt = json.dumps(messages[:-1], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                full = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                valid.append((hashlib.sha256(prompt.encode()).hexdigest(), hashlib.sha256(full.encode()).hexdigest()))
            except (ValueError, KeyError, TypeError):
                invalid_lines.append(line_number)
    return {
        "file_sha256": digest.hexdigest(), "nonempty_rows": rows,
        "valid_rows": len(valid), "invalid_lines": invalid_lines,
        "duplicate_examples": len(valid) - len({full for _, full in valid}),
    }, valid

def report_for(train: Path, validation: Path):
    train_stats, train_hashes = inspect(train)
    validation_stats, validation_hashes = inspect(validation)
    return {
        "train": train_stats, "validation": validation_stats,
        "shared_prompt_hashes": len({p for p, _ in train_hashes} & {p for p, _ in validation_hashes}),
        "shared_example_hashes": len({f for _, f in train_hashes} & {f for _, f in validation_hashes}),
        "scope": "Exact normalized messages only; semantic duplication and token truncation not checked. No source text exported.",
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = report_for(args.train, args.validation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))
    return 1 if report["train"]["invalid_lines"] or report["validation"]["invalid_lines"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
