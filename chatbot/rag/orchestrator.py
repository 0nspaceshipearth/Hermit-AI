
import time
import re
from typing import List, Dict, Optional, Any
from chatbot import config
from chatbot.debug_utils import debug_print
from chatbot.state import HermitContext
from chatbot.intent import classify_query_complexity, QueryComplexity, should_use_pioneer, content_has_numerical_data

class OrchestrationModule:
    """
    Mixin for RAGSystem that handles dynamic orchestration (Gear Shifting).
    Includes:
    - retrieve_with_orchestration()
    - Implementation of orchestration steps (extract, resolve, search, score, etc.)
    """

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
        ctx.log(f"🚀 Starting orchestrated retrieval for: '{query}'")
        ctx.log(f"📊 Query complexity: {complexity.level} (max_steps={complexity.max_steps})")
        
        # Add pioneer_scan if configured
        if getattr(config, 'USE_PIONEER_JOINT', False):
            ctx.add_step("pioneer_scan", priority="low")
        
        # Processing loop
        while not ctx.is_complete():
            step = ctx.pop_step()
            if not step:
                break
                
            ctx.log(f"▶ Executing step: {step}")
            
            # Dispatch to appropriate handler
            if step == "extract":
                self._orchestrate_extract(ctx)
            elif step == "resolve":
                self._orchestrate_resolve(ctx)
            elif step == "search":
                self._orchestrate_search(ctx)
            elif step == "score":
                self._orchestrate_score(ctx)
            elif step == "verify":
                self._orchestrate_verify(ctx)
            elif step == "expand":
                self._orchestrate_expand(ctx)
            elif step == "targeted_search":
                 self._orchestrate_targeted(ctx)
            elif step == "pioneer_scan":
                self._orchestrate_pioneer(ctx)
            else:
                ctx.log(f"⚠ Unknown step '{step}', skipping")
                
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
        if config.DEBUG:
            debug_print("=== ORCHESTRATION LOG ===")
            for log in ctx.logs:
                debug_print(log)
            debug_print(f"Final signals: {ctx.signals}")
        
        return ctx.retrieved_data[:top_k]
    
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
                
                # Inject search for resolved entity
                old_flag = config.USE_ORCHESTRATION
                config.USE_ORCHESTRATION = False
                
                for term in search_terms[:2]:  # Try top 2 variations
                    results = self.retrieve(term, top_k=3)
                    if results:
                        ctx.retrieved_data.extend(results)
                        ctx.log(f"  Retrieved {len(results)} articles for '{term}'")
                        break
                
                config.USE_ORCHESTRATION = old_flag
            else:
                ctx.log("  No indirect references detected")
                
        except Exception as e:
            ctx.log(f"  ⚠ Multi-hop resolution failed: {e}")

    def _orchestrate_search(self, ctx) -> None:
        """Execute title-based search using existing retrieval."""
        try:
            # Use existing retrieve() but with orchestration disabled to avoid recursion
            old_flag = config.USE_ORCHESTRATION
            config.USE_ORCHESTRATION = False
            
            results = self.retrieve(ctx.original_query, top_k=10)
            
            config.USE_ORCHESTRATION = old_flag
            
            # Merge new results with existing (avoid duplicates)
            existing_titles = {r.get('metadata', {}).get('title') for r in ctx.retrieved_data}
            for result in results:
                title = result.get('metadata', {}).get('title')
                if title not in existing_titles:
                    ctx.retrieved_data.append(result)
                    
            ctx.log(f"  Retrieved {len(results)} articles")
            
        except Exception as e:
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
            
            total_entities = len(ctx.extracted_entities.get('entities', []))
            covered_entities = len(coverage_result.get('covered', []))
            
            if total_entities > 0:
                ctx.signals["coverage_ratio"] = covered_entities / total_entities
            else:
                ctx.signals["coverage_ratio"] = 1.0
                
            ctx.log(f"  Coverage: {covered_entities}/{total_entities} entities ({ctx.signals['coverage_ratio']:.0%})")
            
            # Store missing entities for targeted search
            ctx.iteration_results['missing_entities'] = coverage_result.get('missing', [])
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
                old_flag = config.USE_ORCHESTRATION
                config.USE_ORCHESTRATION = False
                
                for term in expansions[:3]:  # Limit to 3 expansions
                    results = self.retrieve(term, top_k=3)
                    ctx.retrieved_data.extend(results)
                    
                config.USE_ORCHESTRATION = old_flag
                ctx.log(f"  Expanded search with {len(expansions[:3])} alternative queries")
            else:
                ctx.log("  No expansions generated")
                
        except Exception as e:
            ctx.log(f"  ⚠ Query expansion failed: {e}")

    def _orchestrate_targeted(self, ctx) -> None:
        """Search for specific missing entities."""
        missing = ctx.iteration_results.get('missing_entities', [])
        suggested = ctx.iteration_results.get('suggested_searches', [])
        
        if not missing:
            ctx.log("  No missing entities to target")
            return
            
        try:
            old_flag = config.USE_ORCHESTRATION
            config.USE_ORCHESTRATION = False
            
            # Use suggested searches if available, otherwise use entity names
            search_terms = suggested[:5] if suggested else missing[:3]
            
            for term in search_terms:
                results = self.retrieve(term, top_k=2)
                ctx.retrieved_data.extend(results)
                
            config.USE_ORCHESTRATION = old_flag
            ctx.log(f"  Targeted search for {len(search_terms)} missing entities")
            
        except Exception as e:
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

    def _orchestrate_pioneer(self, ctx) -> None:
        """Execute Pioneer Neuro-Symbolic Scan on retrieved data."""
        if not getattr(config, 'USE_PIONEER_JOINT', False):
            return
        if not should_use_pioneer(ctx.original_query):
            ctx.log("  Pioneer: Skipped (query not data/math related)")
            return
        has_data = False
        for res in ctx.retrieved_data[:3]:
            if content_has_numerical_data(res.get('text', '')):
                has_data = True
                break
        if not has_data:
            ctx.log("  Pioneer: Skipped (no numerical data)")
            return
        if not self.pioneer_joint:
            ctx.log("  Pioneer: Joint not available")
            return
        ctx.log("  Pioneer: Analyzing numerical patterns...")
        try:
            for res in ctx.retrieved_data:
                text = res.get('text', '')
                if not content_has_numerical_data(text):
                    continue
                laws = self.pioneer_joint.analyze_content(text)
                if laws:
                    ctx.log(f"    Found law: {laws}")
                    law_block = "\n".join([f"discovered_law: {l}" for l in laws])
                    res['text'] = f"*** PIONEER DISCOVERY ***\n{law_block}\n***********************\n{text}"
                    res['score'] = 20.0
                    ctx.signals["highest_source_score"] = 20.0
        except Exception as e:
            ctx.log(f"  Pioneer scan failed: {e}")
