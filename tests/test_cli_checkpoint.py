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

    def test_script_create_routes_through_contract_and_sets_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "runme.sh")
            envelope = TeleportEnvelope(
                contract_version="teleport.v1",
                intent="script_create",
                source_mode="wave",
                target_mode="chamber",
                objective="create a bash script",
                constraints={"target_path": target, "language": "bash"},
            )

            result = self.cli._execute_file_write_from_response(
                envelope,
                "[HERMIT_FILE]#!/bin/bash\necho hi\n[/HERMIT_FILE]",
            )
            checkpoint = load_runtime_checkpoint()

            self.assertIn("Script written successfully", result)
            self.assertIn("Executable: yes", result)
            self.assertTrue(os.path.exists(target))
            self.assertTrue(os.stat(target).st_mode & 0o111)
            latest = checkpoint["artifacts"][-1]
            self.assertEqual(latest["mode"], "script_create")
            self.assertEqual(latest["status"], "ok")

    def test_shell_artifact_payload_is_compacted_for_checkpoint(self):
        envelope = TeleportEnvelope(
            contract_version="teleport.v1",
            intent="shell_command",
            source_mode="wave",
            target_mode="chamber",
            objective="python3 -c \"print('x'*2000)\"",
            constraints={"command": "python3 -c \"print('x'*2000)\""},
        )

        self.cli._execute_teleport(envelope)
        checkpoint = load_runtime_checkpoint()
        latest = checkpoint["artifacts"][-1]
        payload = latest.get("payload", {})

        self.assertIn("stdout", payload)
        self.assertLessEqual(len(payload["stdout"]), 800)
        self.assertTrue(payload["stdout"].endswith("..."))


if __name__ == "__main__":
    unittest.main()
