"""
Salience Calculator - Core engine for computing salience scores.

This calculator works with ANY user-defined configuration. It has ZERO hardcoded logic
about what components should exist or what they should be called.
"""

import time
import random
import re
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

from memlayer.config.salience import (
    TenantSalienceConfig,
    SalienceComponent,
    ScoringFunctionType,
    ThresholdStrategy,
    DecisionRule,
)


class SalienceCalculator:
    """
    Computes salience scores based on fully customizable user configurations.

    This class implements a flexible scoring engine that:
    1. Computes scores for each user-defined component
    2. Calculates weighted salience score
    3. Computes adaptive threshold
    4. Applies user-defined decision rules
    5. Logs all computation details for auditability

    No hardcoded assumptions about component names or weights.
    """

    def __init__(
        self,
        config: TenantSalienceConfig,
        fact: str,
        tenant_id: str,
        fact_id: Optional[str] = None,
        embedding_model: Optional[Any] = None,
    ):
        """
        Initialize the salience calculator.

        Args:
            config: The tenant's salience configuration
            fact: The fact text to score
            tenant_id: Tenant identifier
            fact_id: Optional fact ID for logging
            embedding_model: Optional embedding model for similarity calculations
        """
        self.config = config
        self.fact = fact
        self.tenant_id = tenant_id
        self.fact_id = fact_id
        self.embedding_model = embedding_model

        # Computation state
        self.component_scores: Dict[str, float] = {}
        self.weighted_score: float = 0.0
        self.threshold: float = 0.0
        self.decision: str = "SKIP"
        self.matched_rule: Optional[str] = None
        self.reasoning: str = ""

        # Detailed logging for debugging and audit
        self.log: Dict[str, Any] = {
            "fact": fact,
            "fact_id": fact_id,
            "tenant_id": tenant_id,
            "config_name": config.config_name,
            "computation_steps": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

    def compute_salience(self) -> Tuple[float, str, Dict[str, Any]]:
        """
        Main computation method. Returns (score, decision, full_log).

        Steps:
        1. Compute each component score
        2. Calculate weighted salience score
        3. Compute adaptive threshold
        4. Apply decision rules (if any)
        5. Make final decision (STORE/SKIP)

        Returns:
            Tuple of (final_salience_score, decision, computation_log)
        """
        start_time = time.time()

        try:
            # Step 1: Compute component scores
            self._log_step("Starting component score computation")
            for component in self.config.components:
                score = self._compute_component(component)
                self.component_scores[component.name] = score
                self._log_step(
                    f"Component '{component.name}': {score:.4f} (weight: {component.weight})"
                )

            # Step 2: Calculate weighted salience score
            self.weighted_score = sum(
                self.component_scores[c.name] * c.weight
                for c in self.config.components
            )
            self._log_step(f"Weighted salience score: {self.weighted_score:.4f}")

            # Step 3: Compute threshold
            self.threshold = self._compute_threshold()
            self._log_step(
                f"Threshold ({self.config.threshold_config.strategy}): {self.threshold:.4f}"
            )

            # Step 4: Apply decision rules (in priority order)
            rule_decision = self._apply_decision_rules(
                self.weighted_score, self.threshold
            )

            if rule_decision:
                self.decision = rule_decision
                self.reasoning = f"Decision rule '{self.matched_rule}' matched"
                self._log_step(f"Decision rule matched: {self.matched_rule} → {self.decision}")
            else:
                # Fallback: threshold-based decision
                if self.weighted_score >= self.threshold:
                    self.decision = "STORE"
                    self.reasoning = (
                        f"Salience score {self.weighted_score:.4f} >= "
                        f"threshold {self.threshold:.4f}"
                    )
                else:
                    self.decision = "SKIP"
                    self.reasoning = (
                        f"Salience score {self.weighted_score:.4f} < "
                        f"threshold {self.threshold:.4f}"
                    )
                self._log_step(f"Threshold-based decision: {self.decision}")

            computation_time_ms = (time.time() - start_time) * 1000

            # Build final log
            self.log.update({
                "component_scores": self.component_scores,
                "weighted_score": self.weighted_score,
                "threshold": self.threshold,
                "decision": self.decision,
                "matched_rule": self.matched_rule,
                "reasoning": self.reasoning,
                "computation_time_ms": computation_time_ms,
            })

            return (self.weighted_score, self.decision, self.log)

        except Exception as e:
            self._log_step(f"ERROR: {str(e)}")
            self.log["error"] = str(e)
            raise

    def _compute_component(self, component: SalienceComponent) -> float:
        """
        Compute score for a single component based on its scoring function.

        Dispatches to the appropriate scoring function implementation.
        """
        func_type = component.scoring_function
        config = component.scoring_config

        try:
            if func_type == ScoringFunctionType.EMBEDDING_SIMILARITY:
                return self._score_embedding_similarity(config)

            elif func_type == ScoringFunctionType.KEYWORD_MATCH:
                return self._score_keyword_match(config)

            elif func_type == ScoringFunctionType.LENGTH_BONUS:
                return self._score_length_bonus(config)

            elif func_type == ScoringFunctionType.FREQUENCY:
                return self._score_frequency(config)

            elif func_type == ScoringFunctionType.LIVENESS:
                return self._score_liveness(config)

            elif func_type == ScoringFunctionType.LLM_SCORED:
                return self._score_llm(config)

            elif func_type == ScoringFunctionType.CUSTOM_PYTHON:
                return self._score_custom_python(config)

            elif func_type == ScoringFunctionType.WEIGHTED_SUM:
                return self._score_weighted_sum(config)

            elif func_type == ScoringFunctionType.BOOLEAN_GATE:
                return self._score_boolean_gate(config)

            else:
                raise ValueError(f"Unknown scoring function type: {func_type}")

        except Exception as e:
            self._log_step(
                f"Error computing component '{component.name}' "
                f"(function: {func_type}): {str(e)}"
            )
            # Return 0.0 on error (fail gracefully)
            return 0.0

    # ========================================================================
    # Scoring Function Implementations
    # ========================================================================

    def _score_embedding_similarity(self, config: Dict[str, Any]) -> float:
        """
        Score based on embedding similarity to existing memories.

        Config params:
        - target: "existing_memories" (what to compare against)
        - similarity_threshold: optional threshold

        TODO: Implement actual embedding similarity once vector storage is integrated.
        For now, returns a stub value.
        """
        # Stub implementation - returns random score for now
        # In production: compute fact embedding, compare to existing memories
        return random.uniform(0.3, 0.9)

    def _score_keyword_match(self, config: Dict[str, Any]) -> float:
        """
        Score based on keyword matching.

        Config params:
        - keywords: List[str] - keywords to match
        - case_sensitive: bool (default False)

        Returns: (matched_count / total_keywords)
        """
        keywords = config.get("keywords", [])
        if not keywords:
            return 0.0

        case_sensitive = config.get("case_sensitive", False)
        fact_text = self.fact if case_sensitive else self.fact.lower()

        matched = 0
        for keyword in keywords:
            search_keyword = keyword if case_sensitive else keyword.lower()
            if search_keyword in fact_text:
                matched += 1

        score = matched / len(keywords) if keywords else 0.0
        return min(score, 1.0)

    def _score_length_bonus(self, config: Dict[str, Any]) -> float:
        """
        Score based on text length (longer = higher score).

        Config params:
        - min_length: minimum length for score 0.0
        - max_length: maximum length for score 1.0
        """
        min_length = config.get("min_length", 10)
        max_length = config.get("max_length", 500)

        fact_length = len(self.fact)

        if fact_length <= min_length:
            return 0.0
        elif fact_length >= max_length:
            return 1.0
        else:
            # Linear interpolation
            return (fact_length - min_length) / (max_length - min_length)

    def _score_frequency(self, config: Dict[str, Any]) -> float:
        """
        Score based on term frequency (how often certain terms appear).

        Config params:
        - target_terms: List[str] - terms to count

        Returns: normalized frequency score
        """
        target_terms = config.get("target_terms", [])
        if not target_terms:
            return 0.5

        fact_lower = self.fact.lower()
        total_count = sum(fact_lower.count(term.lower()) for term in target_terms)

        # Normalize by fact length
        normalized = total_count / max(len(self.fact.split()), 1)
        return min(normalized, 1.0)

    def _score_liveness(self, config: Dict[str, Any]) -> float:
        """
        Score based on recency/freshness indicators (e.g., temporal keywords).

        Config params:
        - temporal_keywords: List[str] - keywords indicating recency
          (e.g., ["today", "recently", "just", "now", "current"])

        Returns: 1.0 if any temporal keyword found, else 0.0
        """
        temporal_keywords = config.get(
            "temporal_keywords",
            ["today", "recently", "just", "now", "currently", "this week", "this month"]
        )

        fact_lower = self.fact.lower()
        for keyword in temporal_keywords:
            if keyword.lower() in fact_lower:
                return 1.0

        return 0.0

    def _score_llm(self, config: Dict[str, Any]) -> float:
        """
        Score using an LLM prompt.

        Config params:
        - prompt: The prompt to send to the LLM
        - model: Optional model name

        TODO: Implement actual LLM scoring.
        For now, returns a stub value.
        """
        # Stub implementation
        # In production: call LLM with prompt, parse response
        return random.uniform(0.4, 0.8)

    def _score_custom_python(self, config: Dict[str, Any]) -> float:
        """
        Execute custom Python code to compute score.

        Config params:
        - code: Python code string that computes a score
        - The code has access to: fact, tenant_id

        SECURITY WARNING: eval() is used. Only allow trusted users!
        """
        code = config.get("code", "")
        if not code:
            return 0.5

        try:
            # Create safe context with limited scope
            context = {
                "fact": self.fact,
                "tenant_id": self.tenant_id,
                "len": len,
                "min": min,
                "max": max,
                "sum": sum,
            }

            # Execute code and get result
            # The code should assign to a variable called 'score'
            exec(code, context)
            score = context.get("score", 0.5)

            return float(max(0.0, min(1.0, score)))

        except Exception as e:
            self._log_step(f"Custom Python scoring error: {e}")
            return 0.0

    def _score_weighted_sum(self, config: Dict[str, Any]) -> float:
        """
        Compute a weighted sum of other component scores.

        Config params:
        - components: Dict[str, float] - {component_name: weight}

        Returns: weighted sum of specified components
        """
        component_weights = config.get("components", {})

        total = 0.0
        for comp_name, weight in component_weights.items():
            if comp_name in self.component_scores:
                total += self.component_scores[comp_name] * weight

        return min(total, 1.0)

    def _score_boolean_gate(self, config: Dict[str, Any]) -> float:
        """
        Boolean gate: returns 1.0 or 0.0 based on a condition.

        Config params:
        - condition: Boolean expression (e.g., "length > 100")

        Returns: 1.0 if true, 0.0 if false
        """
        condition = config.get("condition", "")
        if not condition:
            return 0.0

        try:
            # Safe eval context
            context = {
                "fact": self.fact,
                "length": len(self.fact),
                "tenant_id": self.tenant_id,
            }

            result = eval(condition, {"__builtins__": {}}, context)
            return 1.0 if result else 0.0

        except Exception as e:
            self._log_step(f"Boolean gate evaluation error: {e}")
            return 0.0

    # ========================================================================
    # Threshold Computation
    # ========================================================================

    def _compute_threshold(self) -> float:
        """
        Compute the salience threshold based on configured strategy.
        """
        strategy = self.config.threshold_config.strategy
        tc = self.config.threshold_config  # shorthand

        try:
            if strategy == ThresholdStrategy.MEAN_MULTIPLIER:
                recent_scores = self._get_recent_salience(tc.recent_facts_window)
                if not recent_scores:
                    return tc.absolute_threshold  # Fallback
                mean = sum(recent_scores) / len(recent_scores)
                return mean * tc.multiplier

            elif strategy == ThresholdStrategy.PERCENTILE:
                recent_scores = self._get_recent_salience(tc.recent_facts_window)
                if not recent_scores:
                    return tc.absolute_threshold  # Fallback
                return self._calculate_percentile(recent_scores, tc.percentile)

            elif strategy == ThresholdStrategy.ABSOLUTE:
                return tc.absolute_threshold

            elif strategy == ThresholdStrategy.EXPONENTIAL_DECAY:
                # Threshold decays over time (encourages more storage as time passes)
                # Stub: return fixed value
                return tc.absolute_threshold * (1.0 - tc.decay_rate)

            elif strategy == ThresholdStrategy.DYNAMIC_PERCENTILE:
                # Similar to percentile but adjusts window dynamically
                recent_scores = self._get_recent_salience(tc.recent_facts_window)
                if not recent_scores:
                    return tc.absolute_threshold
                return self._calculate_percentile(recent_scores, tc.percentile)

            elif strategy == ThresholdStrategy.CONFIDENCE_WEIGHTED:
                # Weighted by confidence (stub implementation)
                return tc.absolute_threshold

            else:
                # Unknown strategy - fallback to absolute
                return tc.absolute_threshold

        except Exception as e:
            self._log_step(f"Error computing threshold: {e}")
            return tc.absolute_threshold  # Safe fallback

    def _get_recent_salience(self, window: int) -> List[float]:
        """
        Get recent salience scores for adaptive thresholding.

        TODO: Integrate with actual database to fetch recent scores.
        For now, returns stub data.
        """
        # Stub implementation - returns random scores
        # In production: query salience_computation_log table
        return [random.uniform(0.3, 0.8) for _ in range(min(window, 50))]

    def _calculate_percentile(self, scores: List[float], percentile: float) -> float:
        """Calculate percentile value from a list of scores."""
        if not scores:
            return 0.5

        sorted_scores = sorted(scores)
        index = int(len(sorted_scores) * (percentile / 100.0))
        index = max(0, min(index, len(sorted_scores) - 1))
        return sorted_scores[index]

    # ========================================================================
    # Decision Rules
    # ========================================================================

    def _apply_decision_rules(
        self, salience: float, threshold: float
    ) -> Optional[str]:
        """
        Apply user-defined decision rules in priority order.

        Rules are evaluated with a safe eval() context containing:
        - salience_score
        - threshold
        - All component scores (by component name)

        Returns: "STORE" or "SKIP" if a rule matches, None otherwise
        """
        if not self.config.decision_rules:
            return None

        # Build evaluation context
        context = {
            "salience_score": salience,
            "threshold": threshold,
            **self.component_scores,  # Add all component scores
        }

        # Evaluate rules in priority order (already sorted by config model)
        for rule in self.config.decision_rules:
            try:
                # Evaluate condition with safe context
                result = eval(
                    rule.condition,
                    {"__builtins__": {}},  # No builtins for security
                    context
                )

                if result:
                    # Rule matched!
                    self.matched_rule = rule.name
                    self._log_step(
                        f"Decision rule '{rule.name}' matched "
                        f"(condition: {rule.condition}) → {rule.action}"
                    )
                    return rule.action

            except Exception as e:
                self._log_step(
                    f"Error evaluating rule '{rule.name}' "
                    f"(condition: {rule.condition}): {e}"
                )
                # Continue to next rule on error
                continue

        # No rules matched
        return None

    # ========================================================================
    # Logging
    # ========================================================================

    def _log_step(self, message: str):
        """Add a computation step to the log."""
        self.log["computation_steps"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
        })
