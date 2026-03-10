import unittest

from chatbot import config
from chatbot.rag.orchestrator import OrchestrationModule
from chatbot.state import HermitContext


class _DummyOrchestrator(OrchestrationModule):
    def __init__(self):
        self.use_joints = False

    def retrieve(self, _query, top_k=10):
        raise RuntimeError("simulated retrieval failure")


class TestOrchestratorBackwardCompatibility(unittest.TestCase):
    def test_search_restores_orchestration_flag_on_failure(self):
        dummy = _DummyOrchestrator()
        ctx = HermitContext(original_query="test")

        original_flag = config.USE_ORCHESTRATION
        config.USE_ORCHESTRATION = True
        try:
            dummy._orchestrate_search(ctx)
            self.assertTrue(config.USE_ORCHESTRATION)
        finally:
            config.USE_ORCHESTRATION = original_flag


if __name__ == "__main__":
    unittest.main()
