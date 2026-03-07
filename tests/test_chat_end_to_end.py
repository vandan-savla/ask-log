import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from log_whisperer.cli import main as cli_main


class DummyResponse:
    def __init__(self, content: str):
        self.content = content


class DummyLLM:
    def invoke(self, messages):
        return DummyResponse("dummy response")


@pytest.fixture()
def temp_home(monkeypatch, tmp_path):
    # Redirect Path.home() used by Config to a temporary directory
    class _FakeHome(Path):
        _flavour = Path('.')._flavour

    fake_home = _FakeHome(tmp_path)
    monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)
    return tmp_path


def test_chat_without_config_shows_error(temp_home, tmp_path):
    # Create a minimal log file
    log_file = tmp_path / "sample.log"
    log_file.write_text("INFO start\nERROR something bad\n")

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        [
            "chat",
            "--log-file",
            str(log_file),
        ],
    )

    assert result.exit_code == 0
    assert "No LLM provider configured" in result.output


def test_chat_with_config_and_exit(monkeypatch, temp_home, tmp_path):
    # Prepare config
    config_dir = Path(temp_home) / ".log-whisperer"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    config_file.write_text(
        """
provider:
  provider: openai
  model: gpt-4o-mini
  api_key: test-key
        """.strip()
    )

    # Create a minimal log file
    log_file = Path(temp_home) / "sample.log"
    log_file.write_text("INFO start\nWARN be careful\nERROR boom\n")

    # Patch LLM factory to avoid network calls
    from log_whisperer import llm_factory as llm_factory_module
    monkeypatch.setattr(
        llm_factory_module.llm_factory, "create_llm", lambda provider, model, cfg: DummyLLM()
    )

    # Avoid building the vector index during the test (exercise fallback path)
    from log_whisperer import chat as chat_module
    monkeypatch.setattr(chat_module.LogAnalyzer, "_initialize_rag", lambda self, force_rebuild=False: None)

    # Simulate two prompts: one user question, then exit
    inputs = iter(["what errors are present?", "/exit"]) 
    monkeypatch.setattr(chat_module, "prompt", lambda *args, **kwargs: next(inputs))

    # Run CLI
    runner = CliRunner()
    save_path = Path(temp_home) / "conversation.json"
    result = runner.invoke(
        cli_main,
        [
            "chat",
            "--log-file",
            str(log_file),
            "--save",
            str(save_path),
        ],
    )

    assert result.exit_code == 0
    # Welcome banner
    assert "Welcome to Log Whisperer" in result.output
    # Progress line or related messages
    assert "Analyzing" in result.output or "✓ Loaded log file" in result.output
    # Goodbye message
    assert "Goodbye! Your conversation has been saved" in result.output

    # Verify conversation file written
    assert save_path.exists()
    data = json.loads(save_path.read_text(encoding="utf-8"))
    assert data["log_file"].endswith("sample.log")
    conv = data.get("conversation", [])
    assert any(m["type"] == "human" for m in conv)
    assert any(m["type"] == "ai" for m in conv)


