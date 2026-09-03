"""Execute source authentication checks without importing the GPU runtime."""
import ast
import os
from pathlib import Path
from typing import Optional

import pytest

ROOT = Path(__file__).resolve().parents[1]

class HTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail

def server_auth(key):
    tree = ast.parse((ROOT / "model_server.py").read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "verify_api_key")
    namespace = {"API_KEY": key, "Optional": Optional, "HTTPException": HTTPException}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "server-auth", "exec"), namespace)
    return namespace["verify_api_key"]

@pytest.mark.parametrize("value", [None, "", "   "])
def test_startup_requires_explicit_nonempty_key(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("PCB_LLM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("PCB_LLM_API_KEY", value)
    tree = ast.parse((ROOT / "model_server.py").read_text())
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "API_KEY" for t in node.targets))
    guard = next(node for node in tree.body if isinstance(node, ast.If) and ast.unparse(node.test) == "not API_KEY")
    with pytest.raises(RuntimeError, match="PCB_LLM_API_KEY"):
        exec(compile(ast.Module(body=[assignment, guard], type_ignores=[]), "server-config", "exec"), {"os": os})

@pytest.mark.parametrize("header", [None, "Bearer wrong", "test-only-key"])
def test_missing_or_wrong_bearer_rejected(header):
    with pytest.raises(HTTPException) as caught:
        server_auth("test-only-key")(header)
    assert caught.value.status_code == 401

def test_configured_key_accepted():
    assert server_auth("test-only-key")("Bearer test-only-key") is None

def test_unconfigured_runtime_cannot_bypass_auth():
    with pytest.raises(HTTPException) as caught:
        server_auth(None)(None)
    assert caught.value.status_code == 503
