"""
Salience Calculator Service

Computes salience scores for facts based on flexible, user-defined configurations.
Supports custom components, threshold strategies, and decision rules.
"""

import math
import statistics
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import numpy as np

from memlayer.config.salience import (
    TenantSalienceConfig,
    SalienceComponent,
    ScoringFunctionType,
    ThresholdStrategy,
)


class SalienceCalculator:
    """
    Calculates salience scores for facts using flexible configurations.

    A salience score determines whether a fact is important enough to store
    in memory. The system computes a weighted sum of custom components,
    applies a threshold strategy, and evaluates decision rules.

    Example:
        config = TenantSalienceConfig(
            tenant_id="user-123",
            config_name="my_config",
            components=[...],
            threshold_config=...
        )
        calc = SalienceCalculator(config, "This is a fact", "user-123")
        score, decision, details = calc.compute_salience()
    """

    def __init__(
        self,
        config: TenantSalienceConfig,
        fact: str,
        tenant_id: str,
        recent_salience_history: Optional[List[float]] = None,
    ):
        """
        Initialize the calculator with a configuration and fact.

        Args:
            config: TenantSalienceConfig with all scoring components and rules
            fact: The text of the fact to score
            tenant_id: Tenant identifier for multi-tenancy
            recent_salience_history: Previous salience scores (for threshold calculation)
        """
        self.config = config
        self.fact = fact
        self.tenant_id = tenant_id
        self.recent_salience_history = recent_salience_history or []

        # Computation log for debugging
        self.log: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "config_name": config.config_name,
            "fact": fact[:200],  # Log first 200 chars
            "timestamp": datetime.utcnow().isoformat(),
            "component_scores": {},
            "threshold_calculation": {},
            "decision_rules_evaluated": [],
            "final_decision": None,
            "errors": [],
        }

    def compute_salience(self) -> Tuple[float, str, Dict[str, Any]]:
        """
        Compute complete salience score for the fact.

        Returns:
            Tuple of (salience_score, decision, component_scores)
            - salience_score: float (0-1)
            - decision: "STORE" or "SKIP"
            - component_scores: dict mapping component name to score
        """
        try:
            # Step 1: Compute each component's score
            component_scores = self._compute_all_components()
            self.log["component_scores"] = component_scores

            # Step 2: Weighted sum
            weighted_score = self._compute_weighted_score(component_scores)
            self.log["weighted_score"] = weighted_score

            # Step 3: Compute threshold
            threshold = self._compute_threshold()
            self.log["threshold"] = threshold

            # Step 4: Apply decision rules
            decision = self._apply_decision_rules(weighted_score, threshold, component_scores)
            self.log["final_decision"] = decision

            return weighted_score, decision, component_scores

        except Exception as e:
            self.log["errors"].append(str(e))
            raise

    def _compute_all_components(self) -> Dict[str, float]:
        """
        Compute scores for all components in the configuration.

        Returns:
            Dict mapping component name to computed score
        """
        scores = {}
        for component in self.config.components:
            try:
                score = self._compute_component(component)
                # Clamp to [0, 1]
                score = max(0.0, min(1.0, score))
                scores[component.name] = score
            except Exception as e:
                error_msg = f"Error computing {component.name}: {str(e)}"
                self.log["errors"].append(error_msg)
                # Use 0.5 as default if computation fails
                scores[component.name] = 0.5

        return scores

    def _compute_component(self, component: SalienceComponent) -> float:
        """
        Compute score for a single component based on its scoring function.

        Args:
            component: SalienceComponent with scoring function type and config

        Returns:
            Score between 0 and 1
        """
        scoring_func = component.scoring_function

        if scoring_func == ScoringFunctionType.EMBEDDING_SIMILARITY:
            return self._score_embedding_similarity(component)
        elif scoring_func == ScoringFunctionType.KEYWORD_MATCH:
            return self._score_keyword_match(component)
        elif scoring_func == ScoringFunctionType.LENGTH_BONUS:
            return self._score_length_bonus(component)
        elif scoring_func == ScoringFunctionType.FREQUENCY:
            return self._score_frequency(component)
        elif scoring_func == ScoringFunctionType.LIVENESS:
            return self._score_liveness(component)
        elif scoring_func == ScoringFunctionType.LLM_SCORED:
            return self._score_llm_scored(component)
        elif scoring_func == ScoringFunctionType.CUSTOM_PYTHON:
            return self._score_custom_python(component)
        elif scoring_func == ScoringFunctionType.WEIGHTED_SUM:
            return self._score_weighted_sum(component)
        elif scoring_func == ScoringFunctionType.BOOLEAN_GATE:
            return self._score_boolean_gate(component)
        else:
            raise ValueError(f"Unknown scoring function: {scoring_func}")

    def _score_embedding_similarity(self, component: SalienceComponent) -> float:
        """
        Score based on embedding similarity to reference embeddings.

        Stub implementation - in production would use actual embeddings.
        """
        config = component.scoring_config
        target = config.get("target", "existing_memories")

        # Stub: return random score based on fact length
        # In production: compute actual cosine similarity with embeddings
        return min(1.0, len(self.fact) / 500)

    def _score_keyword_match(self, component: SalienceComponent) -> float:
        """
        Score based on presence of keywords in the fact.

        Keyword match: count of matching keywords / total keywords
        """
        config = component.scoring_config
        keywords = config.get("keywords", [])

        if not keywords:
            return 0.5  # Default if no keywords specified

        fact_lower = self.fact.lower()
        matches = sum(1 for kw in keywords if kw.lower() in fact_lower)

        return matches / len(keywords)

    def _score_length_bonus(self, component: SalienceComponent) -> float:
        """
        Score based on length of the fact.

        Longer facts (up to min_length) get higher scores.
        """
        config = component.scoring_config
        min_length = config.get("min_length", 100)
        max_length = config.get("max_length", 1000)

        length = len(self.fact)

        if length < min_length:
            return 0.0
        elif length > max_length:
            return 1.0
        else:
            # Linear scale between min and max
            return (length - min_length) / (max_length - min_length)

    def _score_frequency(self, component: SalienceComponent) -> float:
        """
        Score based on frequency of occurrence in recent history.

        Stub implementation - would compare fact to recent facts.
        """
        config = component.scoring_config

        # Stub: return score based on word count (more common words = higher)
        word_count = len(self.fact.split())
        return min(1.0, word_count / 50)

    def _score_liveness(self, component: SalienceComponent) -> float:
        """
        Score based on recency/liveness of the fact.

        Stub implementation - would use timestamp.
        """
        config = component.scoring_config
        days = config.get("days", 180)

        # Stub: always return 1.0 (assuming current fact is "live")
        # In production: compute days since creation and decay
        return 1.0

    def _score_llm_scored(self, component: SalienceComponent) -> float:
        """
        Score by querying an LLM.

        Stub implementation - would call LLM with custom prompt.
        """
        config = component.scoring_config
        prompt_template = config.get("prompt", "Score this 0-1: {fact}")

        # Stub: return score based on fact length (more is better)
        return min(1.0, len(self.fact) / 500)

    def _score_custom_python(self, component: SalienceComponent) -> float:
        """
        Score using custom Python code provided by user.

        The user provides a function_code that will be evaluated.
        """
        config = component.scoring_config
        function_code = config.get("function_code", "def score(fact): return 0.5")

        try:
            # Create a safe namespace
            namespace = {}
            exec(function_code, namespace)
            score_func = namespace.get("score")

            if not score_func:
                raise ValueError("Function code must define a 'score' function")

            return score_func(self.fact)
        except Exception as e:
            raise ValueError(f"Error executing custom Python: {str(e)}")

    def _score_weighted_sum(self, component: SalienceComponent) -> float:
        """
        Score as a weighted sum of sub-components.

        Stub implementation - would recursively score sub-components.
        """
        config = component.scoring_config
        sub_weights = config.get("sub_weights", {})

        if not sub_weights:
            return 0.5

        # Stub: equal weight for each sub-component
        return 0.5

    def _score_boolean_gate(self, component: SalienceComponent) -> float:
        """
        Score based on boolean condition.

        Returns 1.0 if condition is true, 0.0 if false.
        """
        config = component.scoring_config
        condition = config.get("condition", "True")

        try:
            # Create safe evaluation context
            context = {"fact": self.fact, "len": len(self.fact), "words": len(self.fact.split())}
            result = eval(condition, {"__builtins__": {}}, context)
            return 1.0 if result else 0.0
        except Exception as e:
            raise ValueError(f"Error evaluating boolean gate condition: {str(e)}")

    def _compute_weighted_score(self, component_scores: Dict[str, float]) -> float:
        """
        Compute final weighted score from component scores.

        weighted_score = sum(component_scores[c.name] * c.weight for c in components)
        """
        weighted_score = 0.0
        for component in self.config.components:
            score = component_scores.get(component.name, 0.5)
            weighted_score += score * component.weight

        return min(1.0, max(0.0, weighted_score))

    def _compute_threshold(self) -> float:
        """
        Compute threshold using the configured strategy.

        Returns:
            Threshold value (float between 0 and 1)
        """
        strategy = self.config.threshold_config.strategy

        if strategy == ThresholdStrategy.MEAN_MULTIPLIER:
            return self._threshold_mean_multiplier()
        elif strategy == ThresholdStrategy.PERCENTILE:
            return self._threshold_percentile()
        elif strategy == ThresholdStrategy.EXPONENTIAL_DECAY:
            return self._threshold_exponential_decay()
        elif strategy == ThresholdStrategy.ABSOLUTE:
            return self._threshold_absolute()
        elif strategy == ThresholdStrategy.DYNAMIC_PERCENTILE:
            return self._threshold_dynamic_percentile()
        elif strategy == ThresholdStrategy.CONFIDENCE_WEIGHTED:
            return self._threshold_confidence_weighted()
        else:
            raise ValueError(f"Unknown threshold strategy: {strategy}")

    def _threshold_mean_multiplier(self) -> float:
        """
        Mean multiplier: mean(recent_salience) * multiplier

        Returns:
            Threshold value
        """
        window = self.config.threshold_config.recent_facts_window
        multiplier = self.config.threshold_config.multiplier

        if not self.recent_salience_history:
            # Default to 0.5 if no history
            threshold = 0.5 * multiplier
        else:
            recent = self.recent_salience_history[-window:]
            mean_score = statistics.mean(recent) if recent else 0.5
            threshold = mean_score * multiplier

        self.log["threshold_calculation"]["strategy"] = "mean_multiplier"
        self.log["threshold_calculation"]["mean"] = statistics.mean(self.recent_salience_history[-window:]) if self.recent_salience_history else 0.5
        self.log["threshold_calculation"]["multiplier"] = multiplier

        return min(1.0, max(0.0, threshold))

    def _threshold_percentile(self) -> float:
        """
        Percentile: nth percentile of recent salience scores

        Returns:
            Threshold value
        """
        window = self.config.threshold_config.recent_facts_window
        percentile = self.config.threshold_config.percentile

        if not self.recent_salience_history:
            threshold = 0.5
        else:
            recent = self.recent_salience_history[-window:]
            if len(recent) < 2:
                threshold = statistics.mean(recent) if recent else 0.5
            else:
                # Use numpy percentile for precise calculation
                threshold = float(np.percentile(recent, percentile))

        self.log["threshold_calculation"]["strategy"] = "percentile"
        self.log["threshold_calculation"]["percentile"] = percentile
        self.log["threshold_calculation"]["window_size"] = len(self.recent_salience_history[-window:]) if self.recent_salience_history else 0

        return min(1.0, max(0.0, threshold))

    def _threshold_exponential_decay(self) -> float:
        """
        Exponential decay: older scores get exponentially less weight

        Returns:
            Threshold value
        """
        decay_rate = self.config.threshold_config.decay_rate

        if not self.recent_salience_history:
            threshold = 0.5
        else:
            # Weight recent scores more heavily
            n = len(self.recent_salience_history)
            weights = [math.exp(-decay_rate * (n - i - 1)) for i in range(n)]
            total_weight = sum(weights)
            weighted_sum = sum(s * w for s, w in zip(self.recent_salience_history, weights))
            threshold = weighted_sum / total_weight if total_weight > 0 else 0.5

        self.log["threshold_calculation"]["strategy"] = "exponential_decay"
        self.log["threshold_calculation"]["decay_rate"] = decay_rate

        return min(1.0, max(0.0, threshold))

    def _threshold_absolute(self) -> float:
        """
        Absolute: fixed threshold from configuration

        Returns:
            Threshold value
        """
        threshold = self.config.threshold_config.absolute_threshold
        self.log["threshold_calculation"]["strategy"] = "absolute"
        self.log["threshold_calculation"]["threshold"] = threshold
        return threshold

    def _threshold_dynamic_percentile(self) -> float:
        """
        Dynamic percentile: adjust percentile based on recent variance

        Returns:
            Threshold value
        """
        window = self.config.threshold_config.recent_facts_window
        percentile = self.config.threshold_config.percentile

        if not self.recent_salience_history or len(self.recent_salience_history) < 2:
            threshold = 0.5
        else:
            recent = self.recent_salience_history[-window:]

            # Adjust percentile based on variance
            std_dev = statistics.stdev(recent) if len(recent) > 1 else 0
            adjusted_percentile = percentile + (std_dev * 10)  # Adjust by variance
            adjusted_percentile = min(100, max(0, adjusted_percentile))

            threshold = float(np.percentile(recent, adjusted_percentile))

        self.log["threshold_calculation"]["strategy"] = "dynamic_percentile"
        self.log["threshold_calculation"]["base_percentile"] = percentile

        return min(1.0, max(0.0, threshold))

    def _threshold_confidence_weighted(self) -> float:
        """
        Confidence weighted: weight by confidence in predictions

        Stub implementation - would use model confidence scores.
        """
        # Stub: return mean of recent scores
        if not self.recent_salience_history:
            threshold = 0.5
        else:
            threshold = statistics.mean(self.recent_salience_history)

        self.log["threshold_calculation"]["strategy"] = "confidence_weighted"

        return min(1.0, max(0.0, threshold))

    def _apply_decision_rules(
        self,
        salience_score: float,
        threshold: float,
        component_scores: Dict[str, float]
    ) -> str:
        """
        Evaluate decision rules in priority order.

        The first rule whose condition evaluates to True determines the action.
        Falls back to "SKIP" if score < threshold or no rules match.

        Args:
            salience_score: Computed weighted salience score
            threshold: Computed threshold
            component_scores: Dict of component name -> score

        Returns:
            "STORE" or "SKIP"
        """
        if not self.config.decision_rules:
            # No rules - use threshold comparison
            decision = "STORE" if salience_score >= threshold else "SKIP"
            self.log["decision_rules_evaluated"].append({
                "type": "default_threshold",
                "score": salience_score,
                "threshold": threshold,
                "result": decision
            })
            return decision

        # Sort rules by priority (lower priority wins)
        sorted_rules = sorted(self.config.decision_rules, key=lambda r: r.priority)

        # Build evaluation context
        context = {
            "salience_score": salience_score,
            "threshold": threshold,
            "word_count": len(self.fact.split()),
            "fact_length": len(self.fact),
        }
        # Add all component scores to context
        context.update(component_scores)

        # Evaluate rules in priority order
        for rule in sorted_rules:
            try:
                # Evaluate condition with safe builtins
                result = eval(rule.condition, {"__builtins__": {}}, context)

                self.log["decision_rules_evaluated"].append({
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "condition": rule.condition,
                    "matched": bool(result),
                    "action": rule.action if result else None
                })

                if result:
                    return rule.action

            except Exception as e:
                error_msg = f"Error evaluating rule {rule.rule_id}: {str(e)}"
                self.log["errors"].append(error_msg)
                self.log["decision_rules_evaluated"].append({
                    "rule_id": rule.rule_id,
                    "error": error_msg
                })
                continue

        # Default: compare to threshold
        decision = "STORE" if salience_score >= threshold else "SKIP"
        self.log["decision_rules_evaluated"].append({
            "type": "default_threshold",
            "score": salience_score,
            "threshold": threshold,
            "result": decision
        })

        return decision

    def get_log(self) -> Dict[str, Any]:
        """
        Get the complete computation log for debugging.

        Returns:
            Dict with all computation details
        """
        return self.log
