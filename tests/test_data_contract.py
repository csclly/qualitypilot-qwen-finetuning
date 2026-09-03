import ast
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inspect_dataset", ROOT / "inspect_dataset.py")
inspector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspector)

class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = "".join(f"<{m['role']}>{m['content']}</end>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")
    def __call__(self, text, *, add_special_tokens, truncation, max_length):
        ids = [ord(char) for char in text][:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

def functions(max_length=1024):
    tree = ast.parse((ROOT / "train.py").read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"preprocess", "has_trainable_token"}]
    namespace = {"tokenizer": Tokenizer(), "MAX_LENGTH": max_length}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "train-functions", "exec"), namespace)
    return namespace

def test_assistant_only_masks_prompt_but_keeps_final_answer():
    funcs = functions()
    messages = [{"role":"system","content":"规则"}, {"role":"user","content":"问题"}, {"role":"assistant","content":"答案"}]
    encoded = funcs["preprocess"]({"messages":messages})
    start = len(Tokenizer().apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True))
    assert all(label == -100 for label in encoded["labels"][:start])
    assert encoded["labels"][start:] == encoded["input_ids"][start:]
    assert funcs["has_trainable_token"](encoded)

def test_truncated_assistant_is_filtered():
    funcs = functions(8)
    result = funcs["preprocess"]({"messages":[{"role":"user","content":"很长的问题"}, {"role":"assistant","content":"答案"}]})
    assert not funcs["has_trainable_token"](result)

def test_bad_final_role_rejected():
    with pytest.raises(ValueError):
        functions()["preprocess"]({"messages":[{"role":"user","content":"问题"},{"role":"user","content":"补充"}]})

def test_dataset_overlap_and_invalid_lines_without_text_export(tmp_path):
    import json
    train = tmp_path/"train.jsonl"
    validation = tmp_path/"validation.jsonl"
    row={"messages":[{"role":"user","content":"private question"},{"role":"assistant","content":"private answer"}]}
    train.write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n", encoding="utf-8")
    validation.write_text(json.dumps(row)+"\n{bad json}\n", encoding="utf-8")
    report=inspector.report_for(train, validation)
    assert report["train"]["duplicate_examples"] == 1
    assert report["validation"]["invalid_lines"] == [2]
    assert report["shared_prompt_hashes"] == 1
    assert report["shared_example_hashes"] == 1
    assert "private question" not in json.dumps(report)

def test_source_compiles_without_importing_gpu_runtime():
    for path in ROOT.glob("*.py"):
        compile(path.read_text(), str(path), "exec")


def test_training_refuses_existing_output_without_touching_files(tmp_path):
    import os
    tree = ast.parse((ROOT / "train.py").read_text())
    guard = next(node for node in tree.body if isinstance(node, ast.If) and "os.path.isdir(OUTPUT_DIR)" in ast.unparse(node.test))
    existing = tmp_path / "adapter"
    existing.mkdir()
    weight = existing / "adapter.safetensors"
    weight.write_bytes(b"keep-original")
    with pytest.raises(FileExistsError):
        exec(compile(ast.Module(body=[guard], type_ignores=[]), "output-guard", "exec"), {"os":os, "OUTPUT_DIR":str(existing)})
    assert weight.read_bytes() == b"keep-original"
    fresh = tmp_path / "fresh"
    exec(compile(ast.Module(body=[guard], type_ignores=[]), "output-guard", "exec"), {"os":os, "OUTPUT_DIR":str(fresh)})
