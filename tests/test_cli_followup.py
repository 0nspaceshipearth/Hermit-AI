import unittest
from pathlib import Path
from unittest.mock import patch

from chatbot.cli import ChatbotCLI


class TestCliFollowUpChaining(unittest.TestCase):
    def test_follow_up_loop_chains_multiple_commands(self):
        cli = ChatbotCLI.__new__(ChatbotCLI)
        cli.cwd = Path("/tmp")
        cli.history = []

        executed = []

        def fake_execute(envelope):
            cmd = envelope.constraints.get("command")
            executed.append(cmd)
            return f"ok: {cmd}"

        responses = iter([
            "First command done. Next: [HERMIT_CMD]echo two[/HERMIT_CMD]",
            "All done with no more commands.",
        ])

        cli._execute_teleport = fake_execute
        cli._agentic_generate = lambda _messages: next(responses)

        with patch("chatbot.cli.build_messages", return_value=[{"role": "system", "content": "stub"}]):
            final = cli._run_follow_up_commands("Run first [HERMIT_CMD]echo one[/HERMIT_CMD]", max_rounds=5)

        self.assertEqual(executed, ["echo one", "echo two"])
        self.assertEqual(final, "All done with no more commands.")
        self.assertEqual(len(cli.history), 4)


if __name__ == "__main__":
    unittest.main()
