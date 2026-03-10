import unittest
from pathlib import Path
from unittest.mock import patch

from chatbot import config
from chatbot.agent_runtime import RuntimeTurnResult, handle_turn
from chatbot.models import Message


class TestAgentRuntimeHandleTurn(unittest.TestCase):
    def setUp(self):
        self.original_mode = getattr(config, "RUNTIME_MODE", "classic")
        config.RUNTIME_MODE = "wave"

    def tearDown(self):
        config.RUNTIME_MODE = self.original_mode

    def test_handle_turn_runs_shell_intent_through_executor(self):
        history = [Message(role="user", content="run the command pwd")]
        executed = []

        def fake_execute(envelope, workspace):
            executed.append((envelope.intent, envelope.constraints.get("command"), workspace))
            return "✅ Command executed successfully (exit=0)\n/tmp"

        result = handle_turn(
            system_prompt="system",
            history=history,
            workspace="/tmp",
            execute_teleport=fake_execute,
            build_messages_fn=lambda system_prompt, history, user_query=None: [{"role": "system", "content": system_prompt}],
            build_messages_with_intent_fn=lambda system_prompt, history, user_query=None: (
                [{"role": "system", "content": system_prompt}],
                __import__("chatbot.intent", fromlist=["detect_intent"]).detect_intent("run the command pwd"),
            ),
            generate_text_fn=lambda messages: "done",
        )

        self.assertIsInstance(result, RuntimeTurnResult)
        self.assertTrue(result.handled)
        self.assertEqual(result.path, "wave_shell")
        self.assertEqual(executed[0][0], "shell_command")
        self.assertEqual(executed[0][1], "pwd")
        self.assertEqual(result.assistant_reply, "done")
        self.assertTrue(result.events)

    def test_handle_turn_adds_file_generation_contract(self):
        history = [Message(role="user", content="write a python script to my desktop")]
        seen = {}

        def fake_build_messages(system_prompt, history, user_query=None):
            seen["system_prompt"] = system_prompt
            return [{"role": "system", "content": system_prompt}]

        result = handle_turn(
            system_prompt="base-system",
            history=history,
            workspace="/tmp",
            execute_teleport=lambda envelope, workspace: None,
            build_messages_fn=fake_build_messages,
            build_messages_with_intent_fn=lambda system_prompt, history, user_query=None: (
                [{"role": "system", "content": system_prompt}],
                __import__("chatbot.intent", fromlist=["detect_intent"]).detect_intent("write a python script to my desktop"),
            ),
            generate_text_fn=lambda messages: "[HERMIT_FILE]print('hi')\n[/HERMIT_FILE]",
            execute_file_write_fn=lambda envelope, response, workspace: "✅ File written successfully",
        )

        self.assertTrue(result.handled)
        self.assertEqual(result.path, "wave_file")
        self.assertIn("✅ File written successfully", result.display_text)


if __name__ == "__main__":
    unittest.main()
