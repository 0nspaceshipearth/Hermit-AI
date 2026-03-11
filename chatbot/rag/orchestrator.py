
import time
import re
from typing import List, Dict, Optional, Any
from chatbot import config
from chatbot.debug_utils import debug_print
from chatbot.state import (
    HermitContext,
    route_plan_for_goal,
    classify_routing_mode,
    detect_gap_routing,
)
from chatbot.intent import classify_query_complexity, QueryComplexity

class OrchestrationModule:
    """
    Mixin for RAGSystem that handles dynamic orchestration (Gear Shifting).
    Includes:
    - retrieve_with_orchestration()
    - Implementation of orchestration steps (extract, resolve, search, score, etc.)
    """

    _ORCHESTRATION_STEP_HANDLERS = {
        "extract": "_orchestrate_extract",
        "resolve": "_orchestrate_resolve",
        "search": "_orchestrate_search",
        "score": "_orchestrate_score",
        "verify": "_orchestrate_verify",
        "expand": "_orchestrate_expand",
        "targeted_search": "_orchestrate_targeted",
    }

    def retrieve_with_orchestration(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Dynamic orchestration-based retrieval with signal-driven decision making.
        Uses HermitContext to track state and apply gear-shifting logic.
        Includes query complexity classification for conditional joint skipping.
        """
        # Classify query complexity for optimization
        complexity = classify_query_complexity(query)

        # Initialize context with complexity hints
        ctx = HermitContext(original_query=query)
        ctx.complexity = complexity
        ctx.set_route(route_plan_for_goal(query, complexity.level))

        # First routing hook: classify if we should proactively activate tool-paths.
        routing = classify_routing_mode(query, complexity.level)
        ctx.apply_routing(routing, step="router_query")
        if routing.mode == "retrieval_tools" and "resolve" not in ctx.current_plan:
            ctx.add_step("resolve", priority="high")
        ctx.update_ofr_residue(step="router_query", status="ok")

        ctx.log(f"🚀 Starting orchestrated retrieval for: '{query}'")
        ctx.log(f"📊 Query complexity: {complexity.level} (max_steps={complexity.max_steps})")
        ctx.emit_event(
            kind="orchestration_started",
            step="controller",
            status="ok",
            message="retrieval cycle started",
            payload={"query": query, "complexity": complexity.level},
        )
        
        try:
            # Processing loop
            while not ctx.is_complete():
                step = ctx.pop_step()
                if not step:
                    break

                ctx.log(f"▶ Executing step: {step}")
                ctx.emit_event("step_started", step=step, status="info", message="dispatch")

                # Dispatch to appropriate handler
                try:
                    if not self._dispatch_orchestration_step(ctx, step):
                        continue

                    ctx.add_residue(
                        step,
                        "ok",
                        metrics={
                            "ambiguity_score": float(ctx.signals.get("ambiguity_score", 0.0)),
                            "highest_source_score": float(ctx.signals.get("highest_source_score", 0.0)),
                            "coverage_ratio": float(ctx.signals.get("coverage_ratio", 0.0)),
                        },
                    )
                    ctx.emit_event(
                        "step_finished",
                        step=step,
                        status="ok",
                        payload={
                            "signals": {
                                "ambiguity_score": float(ctx.signals.get("ambiguity_score", 0.0)),
                                "highest_source_score": float(ctx.signals.get("highest_source_score", 0.0)),
                                "coverage_ratio": float(ctx.signals.get("coverage_ratio", 0.0)),
                            }
                        },
                    )
                    ctx.update_ofr_residue(step=step, status="ok")
                except Exception as exc:
                    ctx.add_residue(step, "error", str(exc))
                    ctx.emit_event("step_finished", step=step, status="error", message=str(exc))
                    ctx.update_ofr_residue(step=step, status="error")
                    raise

                # Context gap hook: escalate to retrieval tool-paths when local results are weak.
                gap_route = detect_gap_routing(ctx, step)
                if gap_route and gap_route.mode == "retrieval_tools":
                    ctx.apply_routing(gap_route, step=f"router_gap:{step}")
                    if "expand" not in ctx.current_plan:
                        ctx.add_step("expand", priority="normal")
                    if "targeted_search" not in ctx.current_plan:
                        ctx.add_step("targeted_search", priority="normal")
                    ctx.update_ofr_residue(step=f"router_gap:{step}", status="ok")

                # Apply gear-shifting logic after each step
                self._apply_gear_shift(ctx)

                # Early Termination Check 1: Exact title match (score 11.0 from direct lookup)
                # If first result is an exact match, skip additional processing
                if ctx.retrieved_data:
                    first_score = ctx.retrieved_data[0].get('score', 0)
                    if first_score >= 10.0 and len(ctx.retrieved_data) >= 1:
                        ctx.log(f"✅ Early termination: Exact title match (score={first_score:.1f})")
                        break

                # Early Termination Check 2: High quality results with good coverage
                if (ctx.signals.get("highest_source_score", 0) >= config.HIGH_QUALITY_THRESHOLD
                    and ctx.signals.get("coverage_ratio", 0) >= config.MIN_COVERAGE_THRESHOLD
                    and len(ctx.retrieved_data) >= config.MIN_RESULTS_FOR_EARLY_EXIT):
                    ctx.log(f"✅ Early termination: High quality results found ({ctx.signals['highest_source_score']:.1f} score, {ctx.signals['coverage_ratio']:.0%} coverage)")
                    break

                # Safety check with complexity-aware step limit
                max_steps = getattr(ctx, 'complexity', None)
                max_steps = max_steps.max_steps if max_steps else config.MAX_ORCHESTRATION_STEPS
                if ctx.signals["step_counter"] >= max_steps:
                    ctx.log(f"🛑 Safety limit reached ({max_steps} steps for {getattr(ctx.complexity, 'level', 'default')} query)")
                    break

            # Log final state
            ctx.log(f"✓ Orchestration complete. Retrieved {len(ctx.retrieved_data)} results")
            return ctx.retrieved_data[:top_k]
        finally:
            # Expose compact observability surfaces for callers/UI diagnostics,
            # including partial/error runs.
            self.last_orchestration_status = ctx.orchestration_status_report()
            self.last_orchestration_snapshot = ctx.residue_snapshot(limit=20)

            if config.DEBUG:
                debug_print("=== ORCHESTRATION LOG ===")
                for log in ctx.logs:
                    debug_print(log)
                debug_print(f"Final signals: {ctx.signals}")
                debug_print(f"Orchestration status: {self.last_orchestration_status}")

    def _retrieve_without_orchestration(self, query: str, top_k: int = 10) -> List[Dict]:
        """Call ``retrieve`` with orchestration temporarily disabled.

        Centralizing this guard avoids subtle flag-leak bugs when retrieval
        throws inside orchestration sub-steps.
        """
        old_flag = config.USE_ORCHESTRATION
        config.USE_ORCHESTRATION = False
        try:
            return self.retrieve(query, top_k=top_k)
        finally:
            config.USE_ORCHESTRATION = old_flag
    
    def _dispatch_orchestration_step(self, ctx, step: str) -> bool:
        """Dispatch orchestration step handlers from a single lookup table.

        Returns ``True`` when a known step handler was executed, else ``False``
        after recording a skipped residue entry.
        """
        handler_name = self._ORCHESTRATION_STEP_HANDLERS.get(step)
        if not handler_name:
            ctx.log(f"⚠ Unknown step '{step}', skipping")
            ctx.add_residue(step, "skipped", "unknown step")
            return False

        handler = getattr(self, handler_name, None)
        if not callable(handler):
            ctx.log(f"⚠ Missing handler '{handler_name}' for step '{step}', skipping")
            ctx.add_residue(step, "skipped", "missing handler")
            return False

        handler(ctx)
        return True

    def _result_identity(self, result: Dict[str, Any]) -> str:
        """Build a stable dedup identity for retrieved items."""
        metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
        title = str(metadata.get("title", "")).strip().lower()
        source = str(metadata.get("source", "")).strip().lower()
        chunk = str(metadata.get("chunk_id", "")).strip().lower()

        if title:
            return f"{source}|{title}|{chunk}"
        return repr(result)

    def _merge_unique_results(self, ctx: HermitContext, results: List[Dict[str, Any]]) -> int:
        """Merge retrieval results into context while preventing duplicate records."""
        existing = {self._result_identity(r) for r in ctx.retrieved_data}
        added = 0

        for result in results:
            identity = self._result_identity(result)
            if identity in existing:
                continue
            ctx.retrieved_data.append(result)
            existing.add(identity)
            added += 1

        return added

    def _orchestrate_extract(self, ctx) -> None:
        """Extract entities from query andupdate ambiguity score."""
        if not self.use_joints or not hasattr(self, 'entity_joint'):
            ctx.log("⚠ Entity extraction disabled, using query as-is")
            ctx.signals["ambiguity_score"] = 0.0
            return
            
        try:
            entity_info = self.entity_joint.extract(ctx.original_query)
            ctx.extracted_entities = entity_info
            
            # Calculate ambiguity score
            is_comparison = entity_info.get('is_comparison', False)
            num_entities = len(entity_info.get('entities', []))
            
            if is_comparison:
                ctx.signals["ambiguity_score"] = 0.8  # Comparisons are complex
            elif num_entities > 3:
                ctx.signals["ambiguity_score"] = 0.6  # Multiple entities = moderate complexity
            else:
                ctx.signals["ambiguity_score"] = 0.2  # Simple query
                
            ctx.log(f"  Extracted {num_entities} entities, ambiguity={ctx.signals['ambiguity_score']:.2f}")
            
        except Exception as e:
            ctx.log(f"  ⚠ Entity extraction failed: {e}")
            ctx.signals["ambiguity_score"] = 0.5

    def _build_slot_followup_queries(self, ctx: HermitContext, resolved_entity: str) -> List[str]:
        """Build targeted follow-up lookups for biography/slot-style questions.

        Keeps this deterministic and model-agnostic so small models can still
        trigger richer retrieval passes.
        """
        query = (ctx.original_query or "").lower()
        base = (resolved_entity or "").strip()
        if not base:
            return []

        candidates: List[str] = [base, f"{base} biography"]

        # Education-style slots
        education_hints = ["university", "attend", "education", "study", "school", "degree"]
        if any(h in query for h in education_hints):
            candidates.extend([
                f"{base} education",
                f"{base} university",
                f"{base} studied",
                f"{base} attended",
            ])

        # Role/employment-style slots
        employment_hints = ["role", "position", "job", "worked", "work", "at "]
        if any(h in query for h in employment_hints):
            candidates.extend([
                f"{base} career",
                f"{base} employment",
                f"{base} role",
                f"{base} dropbox",
                f"{base} dropbox role",
            ])

        # Dedupe while preserving order
        seen = set()
        ordered: List[str] = []
        for c in candidates:
            k = c.strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            ordered.append(c.strip())
        return ordered[:8]

    def _direct_title_probe(self, title: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Attempt exact/near-exact title retrieval without full candidate generation."""
        if not title:
            return []

        probes = [title.strip()]
        probes.append(title.replace('_', ' ').strip())
        probes.append(title.replace(' ', '_').strip())

        seen = set()
        results: List[Dict[str, Any]] = []
        for probe in probes:
            key = probe.lower()
            if not probe or key in seen:
                continue
            seen.add(key)
            try:
                hits = self.search_by_title(probe)[:top_k]
                for h in hits:
                    h.setdefault('score', 9.5)
                results.extend(hits)
            except Exception:
                continue

        return results

    def _orchestrate_resolve(self, ctx) -> None:
        """Resolve indirect entity references using multi-hop resolution."""
        # Skip for simple queries (optimization)
        complexity = getattr(ctx, 'complexity', None)
        if complexity and complexity.skip_multi_hop:
            ctx.log("  ⏭ Skipping multi-hop resolver (simple query)")
            return

        if not self.use_joints or not hasattr(self, 'resolver_joint'):
            ctx.log("⚠ Multi-hop resolver not available")
            return

        if not ctx.extracted_entities or not ctx.retrieved_data:
            ctx.log("  No entities or data to resolve")
            return
            
        try:
            entities = ctx.extracted_entities.get('entities', [])
            resolution = self.resolver_joint.process(
                ctx.original_query,
                entities,
                ctx.retrieved_data
            )
            
            if resolution:
                resolved_entity = resolution.get('resolved_entity')
                base_entity = resolution.get('base_entity')
                search_terms = resolution.get('suggest_search', [])
                
                ctx.log(f"  ✓ Resolved '{base_entity}' → '{resolved_entity}'")
                
                # Store resolution for later reference
                ctx.iteration_results['resolved_entity'] = resolved_entity
                ctx.iteration_results['multi_hop_searches'] = search_terms
                
                # Inject search for resolved entity + slot-oriented probes
                follow_up_terms = []
                follow_up_terms.extend(search_terms[:3])
                follow_up_terms.extend(self._build_slot_followup_queries(ctx, resolved_entity))

                # First, do direct title probes for resolved entity forms.
                total_added = 0
                direct_hits = self._direct_title_probe(resolved_entity, top_k=6)
                if direct_hits:
                    added = self._merge_unique_results(ctx, direct_hits)
                    total_added += added
                    ctx.record_excursion(
                        step="resolve",
                        mode="retrieval_tool",
                        query=resolved_entity,
                        status="ok",
                        note="direct title probe for resolved entity",
                        payload={"result_count": len(direct_hits), "unique_added": added},
                    )
                    ctx.log(f"  Direct probe for '{resolved_entity}' added {added} unique results")

                seen_terms = set()
                attempts = 0
                for term in follow_up_terms:
                    norm = term.strip().lower()
                    if not norm or norm in seen_terms:
                        continue
                    seen_terms.add(norm)
                    attempts += 1

                    results = self._retrieve_without_orchestration(term, top_k=4)
                    ctx.record_excursion(
                        step="resolve",
                        mode="retrieval_tool",
                        query=term,
                        status="ok",
                        note="multi-hop follow-up query",
                        payload={"result_count": len(results), "resolved_entity": resolved_entity},
                    )
                    if results:
                        added = self._merge_unique_results(ctx, results)
                        total_added += added
                        ctx.log(f"  Retrieved {len(results)} articles for '{term}' ({added} unique)")

                    # Stop early once we have enough unique records for downstream scoring.
                    if len(ctx.retrieved_data) >= 8 or total_added >= 5:
                        break

                ctx.log(f"  Multi-hop follow-up attempts: {attempts}, total unique additions: {total_added}")
            else:
                ctx.log("  No indirect references detected")
                
        except Exception as e:
            ctx.record_excursion(
                step="resolve",
                mode="retrieval_tool",
                query=ctx.original_query,
                status="error",
                note=f"multi-hop resolution failed: {e}",
            )
            ctx.log(f"  ⚠ Multi-hop resolution failed: {e}")

    def _orchestrate_search(self, ctx) -> None:
        """Execute title-based search using existing retrieval."""
        try:
            # Use existing retrieve() but with orchestration disabled to avoid recursion
            results = self._retrieve_without_orchestration(ctx.original_query, top_k=10)
            ctx.record_excursion(
                step="search",
                mode="local_retrieval",
                query=ctx.original_query,
                status="ok",
                note="primary retrieval pass",
                payload={"result_count": len(results)},
            )

            added = self._merge_unique_results(ctx, results)
            ctx.log(f"  Retrieved {len(results)} articles ({added} unique)")
            
        except Exception as e:
            ctx.record_excursion(
                step="search",
                mode="local_retrieval",
                query=ctx.original_query,
                status="error",
                note=f"primary retrieval failed: {e}",
            )
            ctx.log(f"  ⚠ Search failed: {e}")

    def _orchestrate_score(self, ctx) -> None:
        """Score retrieved articles and update highest_source_score signal."""
        if not self.use_joints or not hasattr(self, 'scorer_joint'):
            ctx.log("⚠ Scoring disabled")
            ctx.signals["highest_source_score"] = 5.0  # Assume moderate quality
            return
            
        if not ctx.retrieved_data or not ctx.extracted_entities:
            ctx.log("  No data to score")
            ctx.signals["highest_source_score"] = 0.0
            return
            
        try:
            titles = [r.get('metadata', {}).get('title', '') for r in ctx.retrieved_data]
            scored_results = self.scorer_joint.score(
                ctx.original_query,
                ctx.extracted_entities,
                titles,
                top_k=10
            )
            
            if scored_results:
                # Update scores in the actual result objects
                # scored_results is list of (title, score)
                score_map = {t: s for t, s in scored_results}
                
                highest_score = 0.0
                for res in ctx.retrieved_data:
                    t = res.get('metadata', {}).get('title', '')
                    if t in score_map:
                        new_score = score_map[t]
                        res['score'] = new_score
                        if new_score > highest_score:
                            highest_score = new_score
                            
                ctx.signals["highest_source_score"] = highest_score
                ctx.log(f"  Highest score: {highest_score:.1f}/10")
            else:
                ctx.signals["highest_source_score"] = 0.0
                
        except Exception as e:
            ctx.log(f"  ⚠ Scoring failed: {e}")
            ctx.signals["highest_source_score"] = 3.0

    def _orchestrate_verify(self, ctx) -> None:
        """Verify entity coverage and update coverage_ratio signal."""
        # Skip for simple queries (optimization)
        complexity = getattr(ctx, 'complexity', None)
        if complexity and complexity.skip_coverage:
            ctx.log("  ⏭ Skipping coverage verifier (simple query)")
            ctx.signals["coverage_ratio"] = 1.0  # Assume complete for simple queries
            return

        if not self.use_joints or not hasattr(self, 'coverage_joint'):
            ctx.log("⚠ Coverage verification disabled")
            ctx.signals["coverage_ratio"] = 1.0  # Assume complete
            return

        if not ctx.extracted_entities or not ctx.retrieved_data:
            ctx.log("  No entities or data to verify")
            ctx.signals["coverage_ratio"] = 0.0
            return
            
        try:
            coverage_result = self.coverage_joint.verify_coverage(
                ctx.extracted_entities,
                ctx.retrieved_data
            )
            
            covered = list(coverage_result.get('covered', []))
            missing = list(coverage_result.get('missing', []))

            # Normalize indirect placeholders if resolver already produced a concrete entity.
            resolved_entity = str(ctx.iteration_results.get('resolved_entity') or '').strip()
            if resolved_entity and missing:
                lowered_titles = [
                    str(r.get('metadata', {}).get('title', '')).lower()
                    for r in (ctx.retrieved_data or [])
                ]
                resolved_seen = any(resolved_entity.lower() in t for t in lowered_titles)
                if resolved_seen:
                    rewritten_missing = []
                    for m in missing:
                        ml = str(m).lower()
                        if ('creator of' in ml or 'inventor of' in ml or 'founder of' in ml):
                            covered.append(resolved_entity)
                            continue
                        rewritten_missing.append(m)
                    missing = rewritten_missing

            total_entities = len(covered) + len(missing)
            covered_entities = len(covered)

            if total_entities > 0:
                ctx.signals["coverage_ratio"] = covered_entities / total_entities
            else:
                ctx.signals["coverage_ratio"] = 1.0

            ctx.log(f"  Coverage: {covered_entities}/{total_entities} entities ({ctx.signals['coverage_ratio']:.0%})")

            # Store missing entities for targeted search
            ctx.iteration_results['missing_entities'] = missing
            ctx.iteration_results['suggested_searches'] = coverage_result.get('suggested_searches', [])
            
        except Exception as e:
            ctx.log(f"  ⚠ Coverage verification failed: {e}")
            ctx.signals["coverage_ratio"] = 0.5

    def _orchestrate_expand(self, ctx) -> None:
        """Generate query expansions when initial results are poor."""
        if not hasattr(self, 'entity_joint'):
            ctx.log("  ⚠ Query expansion not available")
            return
            
        try:
            failed_terms = [ctx.original_query]
            expansions = self.entity_joint.suggest_expansion(ctx.original_query, failed_terms)
            
            if expansions:
                # Search for each expansion
                total_added = 0
                for term in expansions[:3]:  # Limit to 3 expansions
                    results = self._retrieve_without_orchestration(term, top_k=3)
                    added = self._merge_unique_results(ctx, results)
                    total_added += added
                    ctx.record_excursion(
                        step="expand",
                        mode="retrieval_tool",
                        query=term,
                        status="ok",
                        note="query expansion attempt",
                        payload={"result_count": len(results), "unique_added": added},
                    )
                ctx.log(
                    f"  Expanded search with {len(expansions[:3])} alternative queries "
                    f"({total_added} unique additions)"
                )
            else:
                ctx.log("  No expansions generated")
                
        except Exception as e:
            ctx.record_excursion(
                step="expand",
                mode="retrieval_tool",
                query=ctx.original_query,
                status="error",
                note=f"query expansion failed: {e}",
            )
            ctx.log(f"  ⚠ Query expansion failed: {e}")

    def _orchestrate_targeted(self, ctx) -> None:
        """Search for specific missing entities."""
        missing = ctx.iteration_results.get('missing_entities', [])
        suggested = ctx.iteration_results.get('suggested_searches', [])
        
        if not missing:
            ctx.log("  No missing entities to target")
            return
            
        try:
            # Use suggested searches if available, otherwise use entity names
            search_terms = suggested[:5] if suggested else missing[:3]

            # If resolver found a concrete entity, prioritize slot probes for it.
            resolved_entity = str(ctx.iteration_results.get('resolved_entity') or '').strip()
            if resolved_entity:
                slot_terms = self._build_slot_followup_queries(ctx, resolved_entity)
                search_terms = slot_terms + search_terms

                direct_hits = self._direct_title_probe(resolved_entity, top_k=6)
                if direct_hits:
                    added = self._merge_unique_results(ctx, direct_hits)
                    ctx.record_excursion(
                        step="targeted_search",
                        mode="retrieval_tool",
                        query=resolved_entity,
                        status="ok",
                        note="direct title probe during targeted search",
                        payload={"result_count": len(direct_hits), "unique_added": added},
                    )

            # dedupe terms while preserving order
            deduped_terms = []
            seen_terms = set()
            for term in search_terms:
                norm = str(term).strip().lower()
                if not norm or norm in seen_terms:
                    continue
                seen_terms.add(norm)
                deduped_terms.append(str(term).strip())

            total_added = 0
            for term in deduped_terms[:8]:
                results = self._retrieve_without_orchestration(term, top_k=2)
                added = self._merge_unique_results(ctx, results)
                total_added += added
                ctx.record_excursion(
                    step="targeted_search",
                    mode="retrieval_tool",
                    query=term,
                    status="ok",
                    note="coverage gap targeted lookup",
                    payload={
                        "result_count": len(results),
                        "unique_added": added,
                        "missing_count": len(missing),
                    },
                )
            ctx.log(
                f"  Targeted search executed across {len(deduped_terms[:8])} terms "
                f"({total_added} unique additions)"
            )
            
        except Exception as e:
            ctx.record_excursion(
                step="targeted_search",
                mode="retrieval_tool",
                query=ctx.original_query,
                status="error",
                note=f"targeted search failed: {e}",
                payload={"missing_count": len(missing)},
            )
            ctx.log(f"  ⚠ Targeted search failed: {e}")

    def _apply_gear_shift(self, ctx) -> None:
        """
        Apply gear-shifting logic based on current signals.
        Injects corrective steps into the plan when thresholds are not met.
        """
        # Gear 1.5: High Ambiguity → Multi-Hop Resolution
        # Trigger if ambiguity is high and we haven't tried resolving yet
        if (config.ENABLE_MULTI_HOP_RESOLUTION
            and ctx.signals.get("ambiguity_score", 0) >= config.MULTI_HOP_AMBIGUITY_THRESHOLD
            and "resolve" not in ctx.current_plan
            and not ctx.iteration_results.get('multi_hop_attempted')
            and ctx.signals["step_counter"] < 4):
            ctx.add_step("resolve", priority="high")
            ctx.iteration_results['multi_hop_attempted'] = True
            ctx.log(f"  🔄 GEAR 1.5: High ambiguity ({ctx.signals['ambiguity_score']:.2f}), adding multi-hop resolution")

        # Gear 2: Low source scores → expand query
        if (ctx.signals.get("highest_source_score", 0) < config.MIN_SOURCE_SCORE_THRESHOLD 
            and "expand" not in ctx.current_plan 
            and ctx.signals["step_counter"] < 7):
            ctx.add_step("expand", priority="normal")
            ctx.log(f"  🔄 GEAR 2: Low score ({ctx.signals['highest_source_score']:.1f}), adding query expansion")
        
        # Gear 3: Incomplete coverage → targeted search
        if (ctx.signals.get("coverage_ratio", 1.0) < config.MIN_COVERAGE_THRESHOLD 
            and "targeted_search" not in ctx.current_plan
            and ctx.signals["step_counter"] < 8):
            ctx.add_step("targeted_search", priority="normal")
            # Re-verify after targeted search
            ctx.add_step("verify", priority="normal")
            ctx.log(f"  🔄 GEAR 3: Incomplete coverage ({ctx.signals['coverage_ratio']:.0%}), adding targeted search")
