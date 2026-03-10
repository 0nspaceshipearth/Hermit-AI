
# Hermit - Offline AI Chatbot for Wikipedia & ZIM Files
# Copyright (C) 2026 Hermit-AI, Inc.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
HermitContext: Shared State Manager for Dynamic Orchestration.

This module defines the context object ("Blackboard") that tracks the
state of RAG processing across multiple joint executions, enabling
signal-based decision making and emergent awareness.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time


@dataclass
class ResidueEntry:
    """Compact trace item emitted by each orchestration step."""

    step: str
    status: str  # ok | skipped | error
    note: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class GoalRoute:
    """Goal-driven routing decision for the orchestration plan."""

    goal: str
    plan: List[str]
    confidence: float = 0.5


@dataclass
class RoutingDirective:
    """Minimal routing hook decision: stay local or activate retrieval tool-paths."""

    mode: str  # "local_only" | "retrieval_tools"
    reason: str
    confidence: float = 0.5


@dataclass
class BlackboardEvent:
    """Minimal event envelope for blackboard activity."""

    kind: str
    step: str
    status: str = "info"
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExcursionArtifact:
    """Typed envelope for tool/retrieval excursions triggered by orchestration."""

    step: str
    mode: str  # local_retrieval | retrieval_tool
    query: str
    status: str = "ok"
    note: str = ""
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectiveFrontierRiskResidue:
    """Compact planning/status frame for downstream inspection."""

    objective: str
    frontier: List[str] = field(default_factory=list)
    risk: str = "low"
    residue: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class HermitContext:
    """
    Context Envelope for Dynamic RAG Orchestration.
    
    This dataclass acts as a "Blackboard" architecture, maintaining shared
    state across multiple processing steps. The controller reads and updates
    this context, using mathematical signals to make dynamic routing decisions.
    
    Attributes:
        original_query: The user's original query string
        current_plan: A queue of processing steps to execute
        extracted_entities: Entities extracted by EntityExtractorJoint
        retrieved_data: Articles/chunks retrieved from ZIM files
        signals: Mathematical metrics for decision-making
        logs: Audit trail of controller decisions and actions
        iteration_results: Stores intermediate results per iteration
    """
    
    # Input
    original_query: str
    
    # Dynamic Planning Queue
    current_plan: List[str] = field(default_factory=lambda: [
        "extract",
        "search",
        "score",
        "verify"
    ])

    # Top-level goal selected by router (can evolve over time)
    active_goal: str = "factual_lookup"

    # High-level routing hook (local retrieval vs tool-path assist)
    routing_mode: str = "local_only"
    routing_reason: str = "default"
    
    # Extracted Data
    extracted_entities: Optional[Dict[str, Any]] = None
    retrieved_data: List[Dict[str, Any]] = field(default_factory=list)
    
    # Mathematical Signals (The "Consciousness" Layer)
    signals: Dict[str, float] = field(default_factory=lambda: {
        "ambiguity_score": 0.0,        # 0.0 = clear, 1.0 = highly ambiguous
        "highest_source_score": 0.0,   # 0.0 = irrelevant, 10.0 = perfect match
        "coverage_ratio": 0.0,         # 0.0 = no coverage, 1.0 = all entities covered
        "step_counter": 0,             # Safety: prevent infinite loops
    })
    
    # Audit Trail
    logs: List[str] = field(default_factory=list)
    
    # Iteration Results (for multi-hop or refinement)
    iteration_results: Dict[str, Any] = field(default_factory=dict)

    # Residue schema: compact execution trail from the orchestration loop
    residue: List[ResidueEntry] = field(default_factory=list)

    # Blackboard event stream: minimal, append-only orchestration telemetry
    events: List[BlackboardEvent] = field(default_factory=list)

    # Typed excursion artifacts (tool/local retrieval passes that branched from core plan)
    artifacts: List[ExcursionArtifact] = field(default_factory=list)

    # Compact objective/frontier/risk + residue frame (single object, continuously refreshed)
    ofr_residue: Optional[ObjectiveFrontierRiskResidue] = None

    _VALID_STEP_STATUS = {"ok", "skipped", "error", "info"}
    _VALID_ARTIFACT_STATUS = {"ok", "error", "skipped"}
    _VALID_ARTIFACT_MODES = {"local_retrieval", "retrieval_tool"}
    
    def log(self, message: str) -> None:
        """
        Add a log entry with automatic formatting.
        
        Args:
            message: The log message to record
        """
        step = int(self.signals.get("step_counter", 0))
        self.logs.append(f"[Step {step}] {message}")
    
    def add_step(self, step: str, priority: str = "normal") -> None:
        """
        Add a step to the processing plan.
        
        Args:
            step: The step name to add (e.g., "expand_query", "targeted_search")
            priority: "high" to insert at front, "normal" to append
        """
        if priority == "high":
            self.current_plan.insert(0, step)
            self.log(f"🔴 HIGH PRIORITY: Injecting '{step}' at front of plan")
        else:
            self.current_plan.append(step)
            self.log(f"📋 Added '{step}' to plan")
    
    def pop_step(self) -> Optional[str]:
        """
        Remove and return the next step from the plan.
        
        Returns:
            The next step to execute, or None if plan is empty
        """
        if self.current_plan:
            step = self.current_plan.pop(0)
            self.signals["step_counter"] += 1
            return step
        return None
    
    def is_complete(self) -> bool:
        """
        Check if processing is complete.
        
        Returns:
            True if no more steps in plan or safety limit reached
        """
        return (
            len(self.current_plan) == 0 or 
            self.signals["step_counter"] >= 10
        )
    
    def add_residue(self, step: str, status: str, note: str = "", metrics: Optional[Dict[str, float]] = None) -> None:
        """Append a compact residue record for post-hoc reasoning/debugging."""
        normalized_status = self._normalize_status(status, allowed=self._VALID_STEP_STATUS, fallback="info")
        self.residue.append(
            ResidueEntry(
                step=step,
                status=normalized_status,
                note=note,
                metrics=metrics or {},
            )
        )

    def emit_event(
        self,
        kind: str,
        step: str,
        status: str = "info",
        message: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a minimal blackboard event for observability and replay."""
        normalized_status = self._normalize_status(status, allowed=self._VALID_STEP_STATUS, fallback="info")
        self.events.append(
            BlackboardEvent(
                kind=kind,
                step=step,
                status=normalized_status,
                message=message,
                payload=payload or {},
            )
        )

    def record_excursion(
        self,
        step: str,
        mode: str,
        query: str,
        status: str = "ok",
        note: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a compact, typed artifact for retrieval/tool excursions."""
        normalized_mode = self._normalize_mode(mode)
        normalized_status = self._normalize_status(status, allowed=self._VALID_ARTIFACT_STATUS, fallback="ok")
        normalized_payload = self._normalize_payload(payload)
        normalized_query = (query or "").strip()

        envelope = ExcursionArtifact(
            step=step,
            mode=normalized_mode,
            query=normalized_query,
            status=normalized_status,
            note=note,
            payload=normalized_payload,
        )
        self.artifacts.append(envelope)
        self.emit_event(
            kind="excursion",
            step=step,
            status=normalized_status,
            message=note,
            payload={
                "mode": normalized_mode,
                "query": normalized_query,
                **normalized_payload,
            },
        )

    def _normalize_status(self, status: str, allowed: set, fallback: str) -> str:
        value = (status or "").strip().lower()
        return value if value in allowed else fallback

    def _normalize_mode(self, mode: str) -> str:
        value = (mode or "").strip().lower()
        if value in self._VALID_ARTIFACT_MODES:
            return value
        return "retrieval_tool" if "tool" in value else "local_retrieval"

    def _normalize_payload(self, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not payload:
            return {}

        normalized: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalized[key] = value
            else:
                normalized[key] = repr(value)
        return normalized

    def set_route(self, route: GoalRoute) -> None:
        """Apply goal-driven route decision to this context."""
        self.active_goal = route.goal
        self.current_plan = list(route.plan)
        self.log(f"🧭 Goal route '{route.goal}' selected (confidence={route.confidence:.2f})")
        self.emit_event(
            kind="route_selected",
            step="planner",
            status="ok",
            message=f"goal={route.goal}",
            payload={"confidence": route.confidence, "plan": list(route.plan)},
        )

    def apply_routing(self, directive: RoutingDirective, step: str = "router") -> None:
        """Record a routing-hook decision in the blackboard for inspectability."""
        self.routing_mode = directive.mode
        self.routing_reason = directive.reason
        self.log(
            f"🪝 Routing hook: mode={directive.mode} reason='{directive.reason}' "
            f"(confidence={directive.confidence:.2f})"
        )
        self.emit_event(
            kind="routing_hook",
            step=step,
            status="ok",
            message=directive.reason,
            payload={"mode": directive.mode, "confidence": directive.confidence},
        )

    def update_ofr_residue(self, step: str = "controller", status: str = "info") -> None:
        """Refresh compact objective/frontier/risk residue frame from current context state."""
        ambiguity = float(self.signals.get("ambiguity_score", 0.0))
        highest = float(self.signals.get("highest_source_score", 0.0))
        coverage = float(self.signals.get("coverage_ratio", 0.0))

        if highest < 4.0 or coverage < 0.5:
            risk = "high"
        elif ambiguity > 0.6 or highest < 6.0 or coverage < 0.8:
            risk = "medium"
        else:
            risk = "low"

        last = self.residue[-1] if self.residue else None
        residue_brief = {
            "step": step,
            "status": status,
            "last_step": last.step if last else "",
            "last_status": last.status if last else "",
            "last_note": last.note if last else "",
            "events": len(self.events),
            "artifacts": len(self.artifacts),
        }

        self.ofr_residue = ObjectiveFrontierRiskResidue(
            objective=self.active_goal,
            frontier=list(self.current_plan[:3]),
            risk=risk,
            residue=residue_brief,
        )

    def base_mind_snapshot(self) -> Dict[str, Any]:
        """Return a compact base-mind frame for planners/inspectors.

        This intentionally overlaps with objective_frontier_risk to preserve
        compatibility while exposing a clearer contract for future base-mind
        controllers.
        """
        return {
            "objective": self.active_goal,
            "frontier": list(self.current_plan[:3]),
            "routing": {
                "mode": self.routing_mode,
                "reason": self.routing_reason,
            },
            "risk": self.ofr_residue.risk if self.ofr_residue else "unknown",
            "signals": {
                "ambiguity_score": float(self.signals.get("ambiguity_score", 0.0)),
                "highest_source_score": float(self.signals.get("highest_source_score", 0.0)),
                "coverage_ratio": float(self.signals.get("coverage_ratio", 0.0)),
            },
            "steps": {
                "executed": int(self.signals.get("step_counter", 0)),
                "remaining": len(self.current_plan),
            },
        }

    def residue_snapshot(self, limit: int = 12) -> Dict[str, Any]:
        """Return a compact serializable view of residue/events for downstream use."""
        return {
            "active_goal": self.active_goal,
            "routing_mode": self.routing_mode,
            "routing_reason": self.routing_reason,
            "base_mind": self.base_mind_snapshot(),
            "objective_frontier_risk": {
                "objective": self.ofr_residue.objective,
                "frontier": list(self.ofr_residue.frontier),
                "risk": self.ofr_residue.risk,
                "residue": dict(self.ofr_residue.residue),
                "timestamp": self.ofr_residue.timestamp,
            } if self.ofr_residue else {},
            "signals": {
                "ambiguity_score": float(self.signals.get("ambiguity_score", 0.0)),
                "highest_source_score": float(self.signals.get("highest_source_score", 0.0)),
                "coverage_ratio": float(self.signals.get("coverage_ratio", 0.0)),
            },
            "residue": [
                {
                    "step": r.step,
                    "status": r.status,
                    "note": r.note,
                    "metrics": dict(r.metrics),
                }
                for r in self.residue[-limit:]
            ],
            "events": [
                {
                    "kind": e.kind,
                    "step": e.step,
                    "status": e.status,
                    "message": e.message,
                    "timestamp": e.timestamp,
                    "payload": dict(e.payload),
                }
                for e in self.events[-limit:]
            ],
            "artifacts": [
                {
                    "step": a.step,
                    "mode": a.mode,
                    "query": a.query,
                    "status": a.status,
                    "note": a.note,
                    "timestamp": a.timestamp,
                    "payload": dict(a.payload),
                }
                for a in self.artifacts[-limit:]
            ],
        }

    def orchestration_status_report(self) -> Dict[str, Any]:
        """Small observability surface for orchestration routing + residue health."""
        artifact_by_mode: Dict[str, int] = {}
        artifact_by_status: Dict[str, int] = {}
        for artifact in self.artifacts:
            artifact_by_mode[artifact.mode] = artifact_by_mode.get(artifact.mode, 0) + 1
            artifact_by_status[artifact.status] = artifact_by_status.get(artifact.status, 0) + 1

        latest_artifact = self.artifacts[-1] if self.artifacts else None

        return {
            "mode": self.routing_mode,
            "routing_reason": self.routing_reason,
            "active_goal": self.active_goal,
            "residue_present": len(self.residue) > 0,
            "residue_count": len(self.residue),
            "events_count": len(self.events),
            "artifact_summary": {
                "count": len(self.artifacts),
                "by_mode": artifact_by_mode,
                "by_status": artifact_by_status,
                "latest": {
                    "step": latest_artifact.step,
                    "mode": latest_artifact.mode,
                    "status": latest_artifact.status,
                    "query": latest_artifact.query,
                    "note": latest_artifact.note,
                } if latest_artifact else {},
            },
            "ofr_risk": self.ofr_residue.risk if self.ofr_residue else "unknown",
            "steps_executed": int(self.signals.get("step_counter", 0)),
            "steps_remaining": len(self.current_plan),
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context state.
        
        Returns:
            Dictionary with key context metrics
        """
        return {
            "query": self.original_query,
            "active_goal": self.active_goal,
            "routing_mode": self.routing_mode,
            "routing_reason": self.routing_reason,
            "steps_executed": int(self.signals["step_counter"]),
            "steps_remaining": len(self.current_plan),
            "plan": self.current_plan,
            "signals": self.signals,
            "num_results": len(self.retrieved_data),
            "num_logs": len(self.logs),
            "residue_events": len(self.residue),
            "blackboard_events": len(self.events),
            "excursion_artifacts": len(self.artifacts),
            "ofr_risk": self.ofr_residue.risk if self.ofr_residue else "unknown",
            "status_report": self.orchestration_status_report(),
        }


def route_plan_for_goal(query: str, complexity_level: str = "simple") -> GoalRoute:
    """Base goal router for orchestration plan selection.

    Keeps behavior close to current pipeline while introducing explicit routing.
    """
    q = (query or "").lower()

    if any(token in q for token in ["compare", "versus", " vs ", "difference"]):
        return GoalRoute(
            goal="comparison_analysis",
            plan=["extract", "search", "score", "verify"],
            confidence=0.75,
        )

    if complexity_level == "complex":
        return GoalRoute(
            goal="multi_hop_resolution",
            plan=["extract", "search", "resolve", "score", "verify"],
            confidence=0.70,
        )

    return GoalRoute(
        goal="factual_lookup",
        plan=["extract", "search", "score", "verify"],
        confidence=0.80,
    )


def classify_routing_mode(query: str, complexity_level: str = "simple") -> RoutingDirective:
    """Minimal query classifier for local-vs-tool-path routing hooks."""
    q = (query or "").lower()

    external_hint_tokens = ["latest", "current", "today", "now", "live", "breaking"]
    pronoun_tokens = [" he ", " she ", " they ", " it ", " his ", " her ", " their "]

    if any(token in q for token in external_hint_tokens):
        return RoutingDirective(
            mode="retrieval_tools",
            reason="freshness_or_live_intent_detected",
            confidence=0.70,
        )

    padded = f" {q} "
    if complexity_level == "complex" or any(token in padded for token in pronoun_tokens):
        return RoutingDirective(
            mode="retrieval_tools",
            reason="reference_resolution_likely_needed",
            confidence=0.68,
        )

    return RoutingDirective(mode="local_only", reason="straightforward_local_lookup", confidence=0.82)


def detect_gap_routing(ctx: HermitContext, step: str) -> Optional[RoutingDirective]:
    """Simple context gap detector to escalate into retrieval tool-paths."""
    if step == "search" and len(ctx.retrieved_data) == 0:
        return RoutingDirective("retrieval_tools", "no_initial_hits", confidence=0.92)

    if step in ("score", "verify"):
        highest = float(ctx.signals.get("highest_source_score", 0.0))
        coverage = float(ctx.signals.get("coverage_ratio", 0.0))
        if highest < 4.0 or coverage < 0.5:
            return RoutingDirective("retrieval_tools", "low_quality_or_coverage_gap", confidence=0.78)

    return None
