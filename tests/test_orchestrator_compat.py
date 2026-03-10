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


class _ExplodingRetrieverOrchestrator(OrchestrationModule):
    def __init__(self):
        self.use_joints = True
        self.entity_joint = type("_Entity", (), {"suggest_expansion": lambda *_: ["foo"]})()

    def retrieve(self, _query, top_k=10):
        raise RuntimeError("forced failure")


class TestOrchestratorFlagGuard(unittest.TestCase):
    def test_helper_restores_orchestration_flag_on_failure(self):
        dummy = _ExplodingRetrieverOrchestrator()

        original_flag = config.USE_ORCHESTRATION
        config.USE_ORCHESTRATION = True
        try:
            with self.assertRaises(RuntimeError):
                dummy._retrieve_without_orchestration("test", top_k=2)
            self.assertTrue(config.USE_ORCHESTRATION)
        finally:
            config.USE_ORCHESTRATION = original_flag

    def test_expand_restores_flag_when_retrieve_fails(self):
        dummy = _ExplodingRetrieverOrchestrator()
        ctx = HermitContext(original_query="test")

        original_flag = config.USE_ORCHESTRATION
        config.USE_ORCHESTRATION = True
        try:
            dummy._orchestrate_expand(ctx)
            self.assertTrue(config.USE_ORCHESTRATION)
        finally:
            config.USE_ORCHESTRATION = original_flag


class _DispatchProbeOrchestrator(OrchestrationModule):
    def __init__(self):
        self.use_joints = False
        self.called = []

    def _orchestrate_extract(self, ctx):
        self.called.append("extract")


class _StatusSurfaceOrchestrator(OrchestrationModule):
    def __init__(self):
        self.use_joints = False
        self.entity_joint = type("_Entity", (), {"suggest_expansion": lambda *_: ["seed expansion"]})()

    def retrieve(self, query, top_k=10):
        if query == "seed":
            return []
        return [{"metadata": {"title": query}, "score": 1.0}]


class TestOrchestrationStatusSurfaces(unittest.TestCase):
    def test_retrieve_populates_status_and_snapshot(self):
        dummy = _StatusSurfaceOrchestrator()

        results = dummy.retrieve_with_orchestration("seed", top_k=5)

        self.assertGreaterEqual(len(results), 1)
        self.assertIsInstance(dummy.last_orchestration_status, dict)
        self.assertIsInstance(dummy.last_orchestration_snapshot, dict)
        self.assertEqual(dummy.last_orchestration_status["mode"], "retrieval_tools")
        self.assertGreaterEqual(dummy.last_orchestration_status["artifact_summary"]["count"], 1)
        self.assertIn("contract", dummy.last_orchestration_snapshot)
        self.assertTrue(dummy.last_orchestration_snapshot["contract"]["ok"])
        self.assertIn("base_mind", dummy.last_orchestration_snapshot)
        self.assertIn("objective_frontier_risk", dummy.last_orchestration_snapshot)
        self.assertTrue(
            any(a["step"] == "expand" for a in dummy.last_orchestration_snapshot["artifacts"])
        )


class TestOrchestratorStepDispatch(unittest.TestCase):
    def test_dispatch_known_step_invokes_handler(self):
        dummy = _DispatchProbeOrchestrator()
        ctx = HermitContext(original_query="test")

        handled = dummy._dispatch_orchestration_step(ctx, "extract")

        self.assertTrue(handled)
        self.assertEqual(dummy.called, ["extract"])

    def test_dispatch_unknown_step_marks_skipped(self):
        dummy = _DispatchProbeOrchestrator()
        ctx = HermitContext(original_query="test")

        handled = dummy._dispatch_orchestration_step(ctx, "not_a_real_step")

        self.assertFalse(handled)
        self.assertTrue(any(r.step == "not_a_real_step" and r.status == "skipped" for r in ctx.residue))


if __name__ == "__main__":
    unittest.main()
