import unittest

from chatbot.state import HermitContext, RoutingDirective, route_plan_for_goal


class TestStateRouting(unittest.TestCase):
    def test_default_factual_route(self):
        route = route_plan_for_goal("Who created Python?", "simple")
        self.assertEqual(route.goal, "factual_lookup")
        self.assertEqual(route.plan, ["extract", "search", "score", "verify"])

    def test_complex_route_includes_resolve(self):
        route = route_plan_for_goal("What university did the creator of Python attend?", "complex")
        self.assertEqual(route.goal, "multi_hop_resolution")
        self.assertIn("resolve", route.plan)

    def test_context_residue_tracking(self):
        ctx = HermitContext(original_query="test")
        ctx.add_residue("search", "ok", metrics={"coverage_ratio": 0.5})
        self.assertEqual(len(ctx.residue), 1)
        self.assertEqual(ctx.residue[0].step, "search")

    def test_excursion_artifact_envelope(self):
        ctx = HermitContext(original_query="test")
        ctx.record_excursion(
            step="expand",
            mode="retrieval_tool",
            query="python programming language",
            payload={"result_count": 3},
        )
        snapshot = ctx.residue_snapshot(limit=5)

        self.assertEqual(len(ctx.artifacts), 1)
        self.assertEqual(ctx.artifacts[0].step, "expand")
        self.assertEqual(snapshot["artifacts"][0]["mode"], "retrieval_tool")
        self.assertEqual(snapshot["artifacts"][0]["payload"]["result_count"], 3)

    def test_orchestration_status_report_compact_surface(self):
        ctx = HermitContext(original_query="latest python release")
        ctx.apply_routing(
            directive=RoutingDirective(
                mode="retrieval_tools",
                reason="freshness_or_live_intent_detected",
                confidence=0.7,
            ),
            step="router_query",
        )
        ctx.add_residue("search", "ok", "retrieved local candidates")
        ctx.record_excursion(
            step="targeted_search",
            mode="retrieval_tool",
            query="python 3.13 release",
            status="ok",
            note="freshness verification",
        )

        report = ctx.orchestration_status_report()

        self.assertEqual(report["mode"], "retrieval_tools")
        self.assertTrue(report["residue_present"])
        self.assertEqual(report["artifact_summary"]["count"], 1)
        self.assertEqual(report["artifact_summary"]["by_mode"]["retrieval_tool"], 1)
        self.assertEqual(report["artifact_summary"]["latest"]["step"], "targeted_search")


if __name__ == "__main__":
    unittest.main()
