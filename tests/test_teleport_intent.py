import unittest

from chatbot import config
from chatbot.intent import detect_intent
from chatbot.teleport import build_shell_envelope, detect_command, execute_teleport_contract


class TestTeleportIntentRouting(unittest.TestCase):
    def setUp(self):
        self.original_mode = getattr(config, "RUNTIME_MODE", "classic")

    def tearDown(self):
        config.RUNTIME_MODE = self.original_mode

    def test_classic_mode_keeps_normal_factual_queries_factual(self):
        config.RUNTIME_MODE = "classic"

        intent = detect_intent("what is paris")

        self.assertEqual(intent.mode_name, "FACTUAL")
        self.assertIsNone(intent.shell_intent)
        self.assertIsNone(intent.teleport_envelope)

    def test_classic_mode_blocks_shell_only_when_shell_intent_detected(self):
        config.RUNTIME_MODE = "classic"

        intent = detect_intent("write a python script to my desktop")

        self.assertEqual(intent.mode_name, "SHELL_BLOCKED")
        self.assertEqual(intent.shell_intent, "wave_mode_required")
        self.assertIsNotNone(intent.teleport_envelope)
        self.assertTrue(intent.teleport_envelope.constraints.get("refused"))

    def test_wave_mode_builds_shell_command_envelope(self):
        config.RUNTIME_MODE = "wave"

        intent = detect_intent("run the command pwd")

        self.assertEqual(intent.mode_name, "SHELL")
        self.assertEqual(intent.shell_intent, "shell_command")
        self.assertEqual(intent.teleport_envelope.intent, "shell_command")

    def test_detect_command_strips_filler_words(self):
        self.assertEqual(detect_command("run the command pwd"), "pwd")
        self.assertEqual(detect_command("execute the script test.py"), "test.py")

    def test_execute_shell_command_contract_runs_real_command(self):
        config.RUNTIME_MODE = "wave"
        envelope = build_shell_envelope("run the command pwd", workspace=".")

        self.assertIsNotNone(envelope)
        result = execute_teleport_contract(envelope)

        self.assertTrue(result.ok)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.artifact.get("exit_code"), 0)
        self.assertTrue(result.message)


if __name__ == "__main__":
    unittest.main()
