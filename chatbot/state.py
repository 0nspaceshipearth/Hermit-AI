
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
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current context state.
        
        Returns:
            Dictionary with key context metrics
        """
        return {
            "query": self.original_query,
            "steps_executed": int(self.signals["step_counter"]),
            "steps_remaining": len(self.current_plan),
            "plan": self.current_plan,
            "signals": self.signals,
            "num_results": len(self.retrieved_data),
            "num_logs": len(self.logs)
        }
