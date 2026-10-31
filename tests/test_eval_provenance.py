import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kakao_bot.runtime import RuntimeConfig


spec = importlib.util.spec_from_file_location("gyodong_live_test", Path("evals/gyodong_live.py"))
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


@pytest.fixture
def snapshot(tmp_path):
    shutil.copytree("src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    return tmp_path


def test_provenance_records_loaded_sources_actual_versions_and_safe_config(snapshot):
    config = RuntimeConfig(database_url="postgresql://secret", openai_api_key="secret-key",
                           max_output_tokens=3072, summary_trigger_tokens=17000)
    result = live.runtime_provenance(snapshot, config)
    agent = result["loaded_modules"]["kakao_bot.runtime.agent"]
    assert Path(agent["path"]) == Path("src/kakao_bot/runtime/agent.py").resolve()
    assert agent["snapshot_path"] == "src/kakao_bot/runtime/agent.py"
    assert result["installed_distributions"]["deepagents"] == "0.7.32"
    assert result["python"]["executable"] == sys.executable
    assert result["effective_runtime_config"]["max_output_tokens"] == 3072
    assert result["effective_runtime_config"]["summary_trigger_tokens"] == 17000
    assert "secret" not in json.dumps(result)


def test_provenance_rejects_an_installed_package_outside_the_current_source(snapshot, monkeypatch):
    monkeypatch.setitem(sys.modules, "kakao_bot.foreign", SimpleNamespace(__file__=__file__))
    with pytest.raises(RuntimeError, match="outside the current src"):
        live.runtime_provenance(snapshot, RuntimeConfig(database_url="", openai_api_key=""))


def test_provenance_rejects_a_source_snapshot_mismatch(snapshot):
    (snapshot / "src/kakao_bot/runtime/agent.py").write_text("different source\n")
    with pytest.raises(RuntimeError, match="differs from its snapshot"):
        live.runtime_provenance(snapshot, RuntimeConfig(database_url="", openai_api_key=""))
