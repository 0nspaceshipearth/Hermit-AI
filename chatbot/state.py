
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
        self.residue.append(
            ResidueEntry(
                step=step,
                status=status,
                note=note,
                metrics=metrics or {},
            )
        )

    def set_route(self, route: GoalRoute) -> None:
        """Apply goal-driven route decision to this context."""
        self.active_goal = route.goal
        self.current_plan = list(route.plan)
        self.log(f"🧭 Goal route '{route.goal}' selected (confidence={route.confidence:.2f})")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context state.
        
        Returns:
            Dictionary with key context metrics
        """
        return {
            "query": self.original_query,
            "active_goal": self.active_goal,
            "steps_executed": int(self.signals["step_counter"]),
            "steps_remaining": len(self.current_plan),
            "plan": self.current_plan,
            "signals": self.signals,
            "num_results": len(self.retrieved_data),
            "num_logs": len(self.logs),
            "residue_events": len(self.residue),
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
