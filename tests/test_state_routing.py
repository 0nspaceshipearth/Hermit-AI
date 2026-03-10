import unittest

from chatbot.state import (
    HermitContext,
    RoutingDirective,
    classify_routing_mode,
    detect_gap_routing,
    route_plan_for_goal,
)


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

    def test_residue_snapshot_includes_base_mind_contract(self):
        ctx = HermitContext(original_query="Who created Python?")
        ctx.update_ofr_residue(step="router_query", status="ok")

        snapshot = ctx.residue_snapshot(limit=3)

        self.assertIn("contract", snapshot)
        self.assertEqual(snapshot["contract"]["version"], "base_mind.v1")
        self.assertTrue(snapshot["contract"]["ok"])
        self.assertEqual(snapshot["contract"]["issues"], [])
        self.assertIn("base_mind", snapshot)
        self.assertEqual(snapshot["base_mind"]["objective"], "factual_lookup")
        self.assertIn("routing", snapshot["base_mind"])
        self.assertIn("steps", snapshot["base_mind"])

    def test_apply_routing_normalizes_invalid_mode(self):
        ctx = HermitContext(original_query="test")
        ctx.apply_routing(RoutingDirective(mode="unknown_mode", reason="  ", confidence=0.5))

        self.assertEqual(ctx.routing_mode, "local_only")
        self.assertEqual(ctx.routing_reason, "unspecified")

    def test_residue_snapshot_limit_is_safeguarded(self):
        ctx = HermitContext(original_query="test")
        ctx.add_residue("extract", "ok", "seed")

        snapshot = ctx.residue_snapshot(limit=0)

        self.assertEqual(len(snapshot["residue"]), 1)

    def test_classify_routing_mode_detects_freshness_intent(self):
        directive = classify_routing_mode("What is the latest Python version?")

        self.assertEqual(directive.mode, "retrieval_tools")
        self.assertEqual(directive.reason, "freshness_or_live_intent_detected")

    def test_detect_gap_routing_escalates_for_empty_search(self):
        ctx = HermitContext(original_query="test")

        directive = detect_gap_routing(ctx, step="search")

        self.assertIsNotNone(directive)
        self.assertEqual(directive.mode, "retrieval_tools")
        self.assertEqual(directive.reason, "no_initial_hits")

    def test_detect_gap_routing_escalates_for_low_coverage(self):
        ctx = HermitContext(original_query="test")
        ctx.retrieved_data = [{"title": "Python"}]
        ctx.signals["highest_source_score"] = 7.0
        ctx.signals["coverage_ratio"] = 0.2

        directive = detect_gap_routing(ctx, step="verify")

        self.assertIsNotNone(directive)
        self.assertEqual(directive.mode, "retrieval_tools")
        self.assertEqual(directive.reason, "low_quality_or_coverage_gap")


if __name__ == "__main__":
    unittest.main()
