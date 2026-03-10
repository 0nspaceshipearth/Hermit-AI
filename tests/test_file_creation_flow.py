import unittest

from chatbot.cli import ChatbotCLI


class TestFileCreationFlow(unittest.TestCase):
    def test_extract_file_content_requires_marker_when_strict(self):
        response = "Here is your script:\nprint('hello')"

        content, found = ChatbotCLI._extract_file_content(None, response, language="python", require_marker=True)

        self.assertIsNone(content)
        self.assertFalse(found)

    def test_extract_file_content_accepts_hermit_file_marker(self):
        response = "[HERMIT_FILE]\nprint('hello')\n[/HERMIT_FILE]"

        content, found = ChatbotCLI._extract_file_content(None, response, language="python", require_marker=True)

        self.assertEqual(content, "print('hello')")
        self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()
