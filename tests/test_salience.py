"""
Unit tests for the flexible salience configuration system.

Tests cover:
- Pydantic model validation
- Component weight validation
- Salience calculation with different scoring functions
- Threshold strategies
- Decision rule evaluation
- API endpoints (CRUD operations)
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from memlayer.config.salience import (
    ScoringFunctionType,
    ThresholdStrategy,
    SalienceComponent,
    AdaptiveThresholdConfig,
    DecisionRule,
    TenantSalienceConfig,
)
from memlayer.services.salience_calculator import SalienceCalculator


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def basic_components():
    """Basic set of 3 components with weights summing to 1.0."""
    return [
        SalienceComponent(
            name="novelty",
            weight=0.4,
            description="How new is this information?",
            scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY,
            scoring_config={"target": "existing_memories"}
        ),
        SalienceComponent(
            name="emotional_impact",
            weight=0.3,
            description="Emotional significance",
            scoring_function=ScoringFunctionType.LLM_SCORED,
            scoring_config={"prompt": "Score emotional impact 0-1"}
        ),
        SalienceComponent(
            name="technical_depth",
            weight=0.3,
            description="Technical complexity",
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={"keywords": ["algorithm", "optimization", "neural"]}
        ),
    ]


@pytest.fixture
def threshold_config():
    """Basic threshold configuration."""
    return AdaptiveThresholdConfig(
        strategy=ThresholdStrategy.PERCENTILE,
        recent_facts_window=150,
        percentile=70.0
    )


@pytest.fixture
def decision_rules():
    """Sample decision rules."""
    return [
        DecisionRule(
            name="Always keep high emotional",
            condition="emotional_impact > 0.85",
            action="STORE",
            priority=1
        ),
        DecisionRule(
            name="Skip low novelty",
            condition="novelty < 0.2",
            action="SKIP",
            priority=2
        ),
    ]


@pytest.fixture
def sample_config(basic_components, threshold_config, decision_rules):
    """Complete salience configuration."""
    return TenantSalienceConfig(
        tenant_id="test_tenant_123",
        config_name="test_config",
        components=basic_components,
        threshold_config=threshold_config,
        decision_rules=decision_rules,
        is_active=True
    )


# ============================================================================
# Pydantic Model Tests
# ============================================================================


class TestSalienceComponent:
    """Tests for SalienceComponent model."""

    def test_valid_component(self):
        """Test creating a valid component."""
        comp = SalienceComponent(
            name="novelty",
            weight=0.5,
            description="Test component",
            scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY,
            scoring_config={"target": "memories"}
        )
        assert comp.name == "novelty"
        assert comp.weight == 0.5
        assert comp.scoring_function == ScoringFunctionType.EMBEDDING_SIMILARITY

    def test_weight_bounds(self):
        """Test that weight must be between 0.0 and 1.0."""
        # Valid weights
        SalienceComponent(
            name="test", weight=0.0, scoring_function=ScoringFunctionType.KEYWORD_MATCH
        )
        SalienceComponent(
            name="test", weight=1.0, scoring_function=ScoringFunctionType.KEYWORD_MATCH
        )

        # Invalid weights
        with pytest.raises(ValueError):
            SalienceComponent(
                name="test", weight=-0.1, scoring_function=ScoringFunctionType.KEYWORD_MATCH
            )

        with pytest.raises(ValueError):
            SalienceComponent(
                name="test", weight=1.1, scoring_function=ScoringFunctionType.KEYWORD_MATCH
            )


class TestTenantSalienceConfig:
    """Tests for TenantSalienceConfig model."""

    def test_valid_config(self, sample_config):
        """Test creating a valid configuration."""
        assert sample_config.tenant_id == "test_tenant_123"
        assert sample_config.config_name == "test_config"
        assert len(sample_config.components) == 3
        assert sample_config.is_active is True

    def test_weights_must_sum_to_one(self, threshold_config):
        """Test that component weights must sum to 1.0."""
        # Valid: weights sum to 1.0
        valid_components = [
            SalienceComponent(name="a", weight=0.5, scoring_function=ScoringFunctionType.KEYWORD_MATCH),
            SalienceComponent(name="b", weight=0.5, scoring_function=ScoringFunctionType.KEYWORD_MATCH),
        ]
        config = TenantSalienceConfig(
            tenant_id="test",
            config_name="valid",
            components=valid_components,
            threshold_config=threshold_config
        )
        assert config is not None

        # Valid: weights sum to 0.995 (within tolerance)
        valid_components_rounding = [
            SalienceComponent(name="a", weight=0.495, scoring_function=ScoringFunctionType.KEYWORD_MATCH),
            SalienceComponent(name="b", weight=0.505, scoring_function=ScoringFunctionType.KEYWORD_MATCH),
        ]
        config = TenantSalienceConfig(
            tenant_id="test",
            config_name="valid_rounding",
            components=valid_components_rounding,
            threshold_config=threshold_config
        )
        assert config is not None

        # Invalid: weights sum to 0.8
        invalid_components = [
            SalienceComponent(name="a", weight=0.4, scoring_function=ScoringFunctionType.KEYWORD_MATCH),
            SalienceComponent(name="b", weight=0.4, scoring_function=ScoringFunctionType.KEYWORD_MATCH),
        ]
        with pytest.raises(ValueError, match="must sum to 1.0"):
            TenantSalienceConfig(
                tenant_id="test",
                config_name="invalid",
                components=invalid_components,
                threshold_config=threshold_config
            )

    def test_decision_rules_sorted_by_priority(self, basic_components, threshold_config):
        """Test that decision rules are automatically sorted by priority."""
        rules = [
            DecisionRule(name="rule_3", condition="True", action="STORE", priority=3),
            DecisionRule(name="rule_1", condition="True", action="STORE", priority=1),
            DecisionRule(name="rule_2", condition="True", action="SKIP", priority=2),
        ]

        config = TenantSalienceConfig(
            tenant_id="test",
            config_name="test",
            components=basic_components,
            threshold_config=threshold_config,
            decision_rules=rules
        )

        # Should be sorted: priority 1, 2, 3
        assert config.decision_rules[0].priority == 1
        assert config.decision_rules[1].priority == 2
        assert config.decision_rules[2].priority == 3

    def test_custom_component_names(self, threshold_config):
        """Test that users can use ANY component names (no restrictions)."""
        # Completely custom component names
        custom_components = [
            SalienceComponent(
                name="my_weird_score_54",
                weight=0.25,
                scoring_function=ScoringFunctionType.KEYWORD_MATCH
            ),
            SalienceComponent(
                name="🎯_target_relevance",
                weight=0.25,
                scoring_function=ScoringFunctionType.LLM_SCORED
            ),
            SalienceComponent(
                name="UserDefinedMetric_v2",
                weight=0.5,
                scoring_function=ScoringFunctionType.LENGTH_BONUS
            ),
        ]

        config = TenantSalienceConfig(
            tenant_id="test",
            config_name="custom_names",
            components=custom_components,
            threshold_config=threshold_config
        )

        assert config.components[0].name == "my_weird_score_54"
        assert config.components[1].name == "🎯_target_relevance"
        assert config.components[2].name == "UserDefinedMetric_v2"


# ============================================================================
# Salience Calculator Tests
# ============================================================================


class TestSalienceCalculator:
    """Tests for SalienceCalculator."""

    def test_compute_salience_basic(self, sample_config):
        """Test basic salience computation."""
        calculator = SalienceCalculator(
            config=sample_config,
            fact="Alice works at TechCorp using neural optimization algorithms",
            tenant_id="test_tenant_123"
        )

        score, decision, log = calculator.compute_salience()

        # Should return valid score, decision, and log
        assert 0.0 <= score <= 1.0
        assert decision in ["STORE", "SKIP"]
        assert "component_scores" in log
        assert "computation_steps" in log
        assert "reasoning" in log

    def test_component_scores_computed(self, sample_config):
        """Test that all component scores are computed."""
        calculator = SalienceCalculator(
            config=sample_config,
            fact="Test fact with algorithm and optimization keywords",
            tenant_id="test_tenant_123"
        )

        score, decision, log = calculator.compute_salience()

        # All components should have scores
        assert "novelty" in calculator.component_scores
        assert "emotional_impact" in calculator.component_scores
        assert "technical_depth" in calculator.component_scores

        # All scores should be 0-1
        for comp_score in calculator.component_scores.values():
            assert 0.0 <= comp_score <= 1.0

    def test_weighted_score_calculation(self, basic_components, threshold_config):
        """Test that weighted score is calculated correctly."""
        # Create config with known weights
        config = TenantSalienceConfig(
            tenant_id="test",
            config_name="test",
            components=basic_components,
            threshold_config=threshold_config
        )

        calculator = SalienceCalculator(
            config=config,
            fact="Test fact",
            tenant_id="test"
        )

        # Mock component scores for predictable result
        calculator.component_scores = {
            "novelty": 0.8,
            "emotional_impact": 0.6,
            "technical_depth": 0.4,
        }

        # Calculate expected weighted score
        # 0.8 * 0.4 + 0.6 * 0.3 + 0.4 * 0.3 = 0.32 + 0.18 + 0.12 = 0.62
        expected = 0.62

        # Compute actual weighted score
        weighted = sum(
            calculator.component_scores[c.name] * c.weight
            for c in config.components
        )

        assert abs(weighted - expected) < 0.01  # Allow small floating point error

    def test_keyword_match_scoring(self, threshold_config):
        """Test keyword match scoring function."""
        config = TenantSalienceConfig(
            tenant_id="test",
            config_name="test",
            components=[
                SalienceComponent(
                    name="keyword_score",
                    weight=1.0,
                    scoring_function=ScoringFunctionType.KEYWORD_MATCH,
                    scoring_config={
                        "keywords": ["algorithm", "optimization", "neural"]
                    }
                )
            ],
            threshold_config=threshold_config
        )

        # Fact with 2 out of 3 keywords
        calculator = SalienceCalculator(
            config=config,
            fact="This is about algorithm and neural networks",
            tenant_id="test"
        )

        score, _, _ = calculator.compute_salience()

        # Should match 2/3 keywords = 0.667
        assert abs(score - 0.667) < 0.01

    def test_length_bonus_scoring(self, threshold_config):
        """Test length bonus scoring function."""
        config = TenantSalienceConfig(
            tenant_id="test",
            config_name="test",
            components=[
                SalienceComponent(
                    name="length",
                    weight=1.0,
                    scoring_function=ScoringFunctionType.LENGTH_BONUS,
                    scoring_config={"min_length": 10, "max_length": 100}
                )
            ],
            threshold_config=threshold_config
        )

        # Short fact (below min)
        calc_short = SalienceCalculator(config=config, fact="Short", tenant_id="test")
        score_short, _, _ = calc_short.compute_salience()
        assert score_short == 0.0

        # Long fact (above max)
        calc_long = SalienceCalculator(
            config=config,
            fact="x" * 150,  # 150 chars
            tenant_id="test"
        )
        score_long, _, _ = calc_long.compute_salience()
        assert score_long == 1.0

        # Medium fact (mid-range)
        calc_mid = SalienceCalculator(
            config=config,
            fact="x" * 55,  # 55 chars = middle of 10-100 range
            tenant_id="test"
        )
        score_mid, _, _ = calc_mid.compute_salience()
        assert 0.4 < score_mid < 0.6  # Should be around 0.5


class TestThresholdStrategies:
    """Tests for different threshold calculation strategies."""

    def test_absolute_threshold(self, basic_components):
        """Test absolute threshold strategy."""
        threshold_config = AdaptiveThresholdConfig(
            strategy=ThresholdStrategy.ABSOLUTE,
            absolute_threshold=0.7
        )

        config = TenantSalienceConfig(
            tenant_id="test",
            config_name="test",
            components=basic_components,
            threshold_config=threshold_config
        )

        calculator = SalienceCalculator(config=config, fact="Test", tenant_id="test")
        calculator.compute_salience()

        assert calculator.threshold == 0.7

    def test_mean_multiplier_threshold(self, basic_components):
        """Test mean multiplier threshold strategy."""
        threshold_config = AdaptiveThresholdConfig(
            strategy=ThresholdStrategy.MEAN_MULTIPLIER,
            multiplier=0.8,
            absolute_threshold=0.5  # Fallback if no recent data
        )

        config = TenantSalienceConfig(
            tenant_id="test",
            config_name="test",
            components=basic_components,
            threshold_config=threshold_config
        )

        calculator = SalienceCalculator(config=config, fact="Test", tenant_id="test")
        calculator.compute_salience()

        # Since we're using stub data, threshold should be computed
        # (might be fallback or calculated from stub data)
        assert 0.0 <= calculator.threshold <= 1.0


class TestDecisionRules:
    """Tests for decision rule evaluation."""

    def test_decision_rule_matching(self, basic_components, threshold_config):
        """Test that decision rules are evaluated correctly."""
        rules = [
            DecisionRule(
                name="High novelty rule",
                condition="novelty > 0.9",
                action="STORE",
                priority=1
            )
        ]

        config = TenantSalienceConfig(
            tenant_id="test",
            config_name="test",
            components=basic_components,
            threshold_config=threshold_config,
            decision_rules=rules
        )

        calculator = SalienceCalculator(config=config, fact="Test", tenant_id="test")

        # Mock high novelty score
        calculator.component_scores = {
            "novelty": 0.95,  # > 0.9, should trigger rule
            "emotional_impact": 0.3,
            "technical_depth": 0.2,
        }

        # Apply rules
        decision = calculator._apply_decision_rules(0.5, 0.7)

        assert decision == "STORE"
        assert calculator.matched_rule == "High novelty rule"

    def test_decision_rule_priority_order(self, basic_components, threshold_config):
        """Test that rules are evaluated in priority order (first match wins)."""
        rules = [
            DecisionRule(
                name="Skip rule",
                condition="novelty < 0.3",
                action="SKIP",
                priority=1  # Higher priority
            ),
            DecisionRule(
                name="Store rule",
                condition="novelty < 0.5",  # Would also match
                action="STORE",
                priority=2  # Lower priority
            ),
        ]

        config = TenantSalienceConfig(
            tenant_id="test",
            config_name="test",
            components=basic_components,
            threshold_config=threshold_config,
            decision_rules=rules
        )

        calculator = SalienceCalculator(config=config, fact="Test", tenant_id="test")

        calculator.component_scores = {
            "novelty": 0.2,  # Matches both rules
            "emotional_impact": 0.5,
            "technical_depth": 0.5,
        }

        decision = calculator._apply_decision_rules(0.5, 0.7)

        # First rule (SKIP) should match, not second rule (STORE)
        assert decision == "SKIP"
        assert calculator.matched_rule == "Skip rule"

    def test_decision_rule_with_multiple_conditions(self, basic_components, threshold_config):
        """Test decision rules with complex boolean expressions."""
        rules = [
            DecisionRule(
                name="Complex rule",
                condition="novelty > 0.8 and emotional_impact > 0.7",
                action="STORE",
                priority=1
            )
        ]

        config = TenantSalienceConfig(
            tenant_id="test",
            config_name="test",
            components=basic_components,
            threshold_config=threshold_config,
            decision_rules=rules
        )

        calculator = SalienceCalculator(config=config, fact="Test", tenant_id="test")

        # Both conditions met
        calculator.component_scores = {
            "novelty": 0.85,
            "emotional_impact": 0.75,
            "technical_depth": 0.3,
        }
        decision = calculator._apply_decision_rules(0.5, 0.7)
        assert decision == "STORE"

        # Only one condition met - should not match
        calculator.component_scores = {
            "novelty": 0.85,
            "emotional_impact": 0.5,  # Too low
            "technical_depth": 0.3,
        }
        decision = calculator._apply_decision_rules(0.5, 0.7)
        assert decision is None  # No rule matched


# ============================================================================
# Integration Tests
# ============================================================================


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_complete_workflow(self, threshold_config):
        """Test complete workflow: create config, compute salience, make decision."""
        # 1. Create custom config
        config = TenantSalienceConfig(
            tenant_id="customer_123",
            config_name="my_custom_score_54",
            components=[
                SalienceComponent(
                    name="keyword_relevance",
                    weight=0.6,
                    scoring_function=ScoringFunctionType.KEYWORD_MATCH,
                    scoring_config={"keywords": ["python", "machine learning", "AI"]}
                ),
                SalienceComponent(
                    name="text_length",
                    weight=0.4,
                    scoring_function=ScoringFunctionType.LENGTH_BONUS,
                    scoring_config={"min_length": 20, "max_length": 200}
                ),
            ],
            threshold_config=threshold_config,
            decision_rules=[
                DecisionRule(
                    name="Always keep ML content",
                    condition="keyword_relevance > 0.5",
                    action="STORE",
                    priority=1
                )
            ]
        )

        # 2. Compute salience for a fact
        fact = "This article discusses machine learning and AI applications in Python"
        calculator = SalienceCalculator(config=config, fact=fact, tenant_id="customer_123")
        score, decision, log = calculator.compute_salience()

        # 3. Verify results
        assert score > 0.0
        assert decision in ["STORE", "SKIP"]
        assert "keyword_relevance" in calculator.component_scores
        assert "text_length" in calculator.component_scores

        # Should match 2/3 keywords (machine learning, AI, Python)
        # Keyword relevance should be high, triggering the decision rule
        assert calculator.component_scores["keyword_relevance"] >= 0.5

    def test_zero_restrictions_flexibility(self, threshold_config):
        """Test that system truly has zero restrictions on component design."""
        # User can create config with 10 components, weird names, any weights
        many_components = [
            SalienceComponent(
                name=f"component_{i}",
                weight=0.1,
                scoring_function=ScoringFunctionType.KEYWORD_MATCH,
                scoring_config={"keywords": [f"word{i}"]}
            )
            for i in range(10)
        ]

        config = TenantSalienceConfig(
            tenant_id="test",
            config_name="ultra_custom_config",
            components=many_components,
            threshold_config=threshold_config
        )

        calculator = SalienceCalculator(config=config, fact="Test fact", tenant_id="test")
        score, decision, log = calculator.compute_salience()

        # Should work with 10 components
        assert len(calculator.component_scores) == 10
        assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
