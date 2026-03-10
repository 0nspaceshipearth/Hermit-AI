import os
import tempfile
import unittest
from pathlib import Path

from chatbot import config
from chatbot.chat import clear_runtime_checkpoint, load_runtime_checkpoint
from chatbot.cli import ChatbotCLI
from chatbot.teleport import TeleportEnvelope


class TestCliCheckpointArtifacts(unittest.TestCase):
    def setUp(self):
        self.original_mode = getattr(config, "RUNTIME_MODE", "classic")
        config.RUNTIME_MODE = "wave"
        clear_runtime_checkpoint()
        self.cli = ChatbotCLI.__new__(ChatbotCLI)
        self.cli.cwd = Path("/mnt/space/Hermit-AI-Public")

    def tearDown(self):
        config.RUNTIME_MODE = self.original_mode
        clear_runtime_checkpoint()

    def test_execute_teleport_records_artifact(self):
        envelope = TeleportEnvelope(
            contract_version="teleport.v1",
            intent="shell_command",
            source_mode="wave",
            target_mode="chamber",
            objective="pwd",
            constraints={"command": "pwd"},
        )

        result = self.cli._execute_teleport(envelope)
        checkpoint = load_runtime_checkpoint()

        self.assertIn("Command executed successfully", result)
        self.assertTrue(checkpoint["artifacts"])
        latest = checkpoint["artifacts"][-1]
        self.assertEqual(latest["mode"], "shell_command")
        self.assertEqual(latest["status"], "ok")
        self.assertEqual(checkpoint["source"]["routing_mode"], "shell_chamber")

    def test_file_write_records_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "hello.py")
            envelope = TeleportEnvelope(
                contract_version="teleport.v1",
                intent="file_write",
                source_mode="wave",
                target_mode="chamber",
                objective="write hello file",
                constraints={"target_path": target, "language": "python"},
            )

            result = self.cli._execute_file_write_from_response(
                envelope,
                "[HERMIT_FILE]print('hello')\n[/HERMIT_FILE]",
            )
            checkpoint = load_runtime_checkpoint()

            self.assertIn("File written successfully", result)
            self.assertTrue(os.path.exists(target))
            latest = checkpoint["artifacts"][-1]
            self.assertEqual(latest["mode"], "file_write")
            self.assertEqual(latest["status"], "ok")


if __name__ == "__main__":
    unittest.main()
