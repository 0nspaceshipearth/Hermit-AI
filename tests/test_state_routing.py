import unittest

from chatbot.state import HermitContext, route_plan_for_goal


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


if __name__ == "__main__":
    unittest.main()
