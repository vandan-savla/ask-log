import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from ask_log import cli
from ask_log.chat import LogAnalyzer


class ChatSaveTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.config_dir = self.root / ".ask-log"
        self.log_file = self.root / "app.log"
        self.log_file.write_text("test log", encoding="utf-8")
        self.analyzers = []

        config = type(
            "TestConfig",
            (),
            {
                "config_dir": self.config_dir,
                "get_provider_config": lambda self: {"provider": "test"},
            },
        )()

        class FakeAnalyzer:
            def __init__(inner_self, log_file_path, save_path=None):
                inner_self.log_file_path = Path(log_file_path)
                inner_self.save_path = Path(save_path) if save_path else None
                inner_self.session_id = "test-session"
                inner_self.conversation_history = []
                if inner_self.save_path and inner_self.save_path.exists():
                    LogAnalyzer._load_conversation(inner_self)
                self.analyzers.append(inner_self)

            def start_chat(inner_self):
                inner_self.conversation_history.extend(
                    [
                        {"type": "human", "content": "question"},
                        {"type": "ai", "content": "answer"},
                    ]
                )

            def _save_conversation(inner_self):
                LogAnalyzer._save_conversation(inner_self)

        patches = (
            patch.object(cli, "Config", return_value=config),
            patch.object(cli, "ensure_configured"),
            patch.object(cli, "LogAnalyzer", FakeAnalyzer),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def invoke_chat(self, *args):
        return CliRunner().invoke(
            cli.main,
            ["chat", "--log-file", str(self.log_file), *args],
        )

    def test_explicit_save_uses_filename_in_history_and_writes_json(self):
        supplied_path = self.root / "outside" / "session"

        result = self.invoke_chat("--save", str(supplied_path))

        saved_path = self.config_dir / "history" / "session.json"
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(saved_path.is_file())
        self.assertFalse(supplied_path.exists())
        data = json.loads(saved_path.read_text(encoding="utf-8"))
        self.assertEqual(len(data["conversation"]), 2)
        self.assertIn(str(saved_path), result.output)

    def test_prompted_save_still_writes_under_history(self):
        with (
            patch.object(cli.questionary, "confirm") as confirm,
            patch.object(cli.questionary, "text") as text,
        ):
            confirm.return_value.ask.return_value = True
            text.return_value.ask.return_value = "prompted-session"
            result = self.invoke_chat()

        saved_path = self.config_dir / "history" / "prompted-session.json"
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(saved_path.is_file())
        self.assertEqual(
            len(json.loads(saved_path.read_text(encoding="utf-8"))["conversation"]),
            2,
        )

    def test_explicit_save_resumes_and_updates_existing_transcript(self):
        saved_path = self.config_dir / "history" / "resume.json"
        saved_path.parent.mkdir(parents=True)
        saved_path.write_text(
            json.dumps({"conversation": [{"type": "human", "content": "earlier"}]}),
            encoding="utf-8",
        )

        result = self.invoke_chat("--save", "resume")

        self.assertEqual(result.exit_code, 0, result.output)
        conversation = json.loads(saved_path.read_text(encoding="utf-8"))["conversation"]
        self.assertEqual(
            [message["content"] for message in conversation],
            ["earlier", "question", "answer"],
        )

    def test_explicit_save_failure_is_reported(self):
        with patch.object(LogAnalyzer, "_save_conversation", side_effect=OSError("disk full")):
            result = self.invoke_chat("--save", "session")

        self.assertIn("Error during chat: disk full", result.output)
        self.assertNotIn("Conversation saved to:", result.output)


if __name__ == "__main__":
    unittest.main()
