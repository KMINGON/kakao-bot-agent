"""Reviewed domain packs: configuration, code and immutable source data."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml
from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class LoadedPack:
    root: Path
    manifest: dict[str, Any]
    digest: str
    data: dict[str, Any]
    _files: dict[str, str]

    @property
    def snapshot(self) -> dict[str, str]:
        return {"pack_id": self.manifest["id"], "version": self.manifest["version"],
                "digest": self.digest, "dataset_version": self.manifest["dataset_version"]}

    @property
    def evidence_ids(self) -> set[str]:
        return {source["id"] for source in self.data.get("sources", [])}

    def python_tools(self) -> list[BaseTool]:
        module = self._module()
        tools = module.build_tools(self.data)
        if not all(isinstance(tool, BaseTool) for tool in tools):
            raise ValueError("Pack tools must be LangChain BaseTool objects")
        if len({tool.name for tool in tools}) != len(tools):
            raise ValueError("Duplicate tool names")
        if {tool.name for tool in tools} != set(self.manifest["allowed_tools"]):
            raise ValueError("Pack tool allowlist mismatch")
        return tools

    def _module(self):
        return self._load_module(self.manifest.get("tools_module", "tools.py"), "tools")

    def adapter_module(self) -> ModuleType | None:
        """Load optional operator-reviewed adapters from the admitted pack snapshot."""
        if "adapters_module" not in self.manifest:
            return None
        return self._load_module(self.manifest["adapters_module"], "adapters")

    def _load_module(self, relative: str, kind: str) -> ModuleType:
        path = _module_path(self.root, relative, self._files)
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != self._files[relative]:
            raise ValueError(f"Pack content hash mismatch: {relative}")
        spec = importlib.util.spec_from_file_location(f"domain_pack_{self.digest}_{kind}", path)
        if spec is None or spec.loader is None:
            raise ValueError("Invalid pack Python module")
        module = importlib.util.module_from_spec(spec)
        previous = sys.modules.get(spec.name)
        sys.modules[spec.name] = module
        try:
            # Execute the checked bytes; timestamp-based .pyc caches can contain older code.
            exec(compile(source, str(path), "exec"), module.__dict__)
        except BaseException:
            if previous is None:
                sys.modules.pop(spec.name, None)
            else:
                sys.modules[spec.name] = previous
            raise
        return module

    def validate_answer(self, answer: dict, tool_results: list[dict]) -> dict:
        """Optional reviewed domain hook checks typed facts and can render them deterministically."""
        hook = getattr(self._module(), "validate_answer", None)
        return hook(answer, tool_results) if hook else answer


def _module_path(root: Path, relative: Any, files: dict[str, str]) -> Path:
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or Path(relative).is_absolute() or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative or Path(relative).suffix != ".py"):
        raise ValueError("Pack module must be a relative .py path inside the pack")
    path = root / relative
    if relative not in files or not path.is_file():
        raise ValueError("Pack module must be locked and present")
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Pack module must stay inside the pack without symlinks")
    return path


def load_pack(root: Path | str, ref: str | dict[str, Any]) -> LoadedPack:
    name = ref["pack_id"] if isinstance(ref, dict) else ref
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name):
        raise ValueError("Invalid pack ID")
    base = Path(root).resolve()
    path = (base / name).resolve()
    if not path.is_relative_to(base) or path == base:
        raise ValueError("Pack path escaped configured root")
    manifest = yaml.safe_load((path / "pack.yaml").read_text())
    if not isinstance(manifest, dict) or manifest.get("id") != name:
        raise ValueError("Pack manifest ID mismatch")
    for field in ("version", "dataset_version"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"Missing pack {field}")
    if not isinstance(manifest.get("allowed_tools"), list):
        raise ValueError("Missing explicit tool allowlist")
    lock = json.loads((path / "pack.lock.json").read_text())
    files = lock.get("files", {})
    if not files or "pack.yaml" not in files or "data/records.json" not in files:
        raise ValueError("Incomplete pack file lock")
    for relative, expected in sorted(files.items()):
        source = path / relative
        if source.is_symlink() or not source.resolve().is_relative_to(path):
            raise ValueError("Pack symlinks/path traversal are forbidden")
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pack content hash mismatch: {relative}")
    actual_files = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()
                    and p.name != "pack.lock.json" and "__pycache__" not in p.parts}
    if actual_files != set(files):
        raise ValueError("Untracked or missing pack files; regenerate the lock after review")
    _module_path(path, manifest.get("tools_module", "tools.py"), files)
    if "adapters_module" in manifest:
        _module_path(path, manifest["adapters_module"], files)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if isinstance(ref, dict) and any(ref.get(k) != v for k, v in {
        "digest": digest, "version": manifest["version"],
        "dataset_version": manifest["dataset_version"],
    }.items()):
        raise ValueError("Pack differs from the admitted execution snapshot")
    data = json.loads((path / "data/records.json").read_text())
    ids = [s["id"] for s in data.get("sources", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evidence IDs")
    return LoadedPack(path, manifest, digest, data, files)


def lock_pack(path: Path) -> str:
    """Operator-only: re-lock a reviewed pack; never called by an agent tool."""
    files = {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(path.rglob("*")) if p.is_file()
             and p.name != "pack.lock.json" and "__pycache__" not in p.parts}
    (path / "pack.lock.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
