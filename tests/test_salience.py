"""
Comprehensive tests for the flexible salience configuration system.

Tests cover:
- Pydantic model validation
- SalienceCalculator computation
- Component scoring functions
- Threshold calculation strategies
- Decision rule evaluation
- Storage operations
- API endpoints
"""

import pytest
from datetime import datetime
from memlayer.config.salience import (
    ScoringFunctionType,
    ThresholdStrategy,
    SalienceComponent,
    AdaptiveThresholdConfig,
    DecisionRule,
    TenantSalienceConfig,
)
from memlayer.services.salience_calculator import SalienceCalculator
from memlayer.storage.salience_config_store import InMemorySalienceConfigStore, FileBasedSalienceConfigStore
import tempfile
import shutil


class TestSalienceComponentModels:
    """Test Pydantic models for salience configuration."""

    def test_create_salience_component(self):
        """Test creating a salience component."""
        component = SalienceComponent(
            name="novelty",
            weight=0.5,
            description="How new is this fact?",
            scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY,
            scoring_config={"target": "existing_memories"}
        )

        assert component.name == "novelty"
        assert component.weight == 0.5
        assert component.scoring_function == ScoringFunctionType.EMBEDDING_SIMILARITY

    def test_create_threshold_config(self):
        """Test creating threshold configuration."""
        threshold_config = AdaptiveThresholdConfig(
            strategy=ThresholdStrategy.PERCENTILE,
            recent_facts_window=150,
            percentile=70
        )

        assert threshold_config.strategy == ThresholdStrategy.PERCENTILE
        assert threshold_config.percentile == 70
        assert threshold_config.recent_facts_window == 150

    def test_create_decision_rule(self):
        """Test creating a decision rule."""
        rule = DecisionRule(
            rule_id="rule_1",
            name="High emotional content",
            condition="emotional_impact > 0.9",
            action="STORE",
            priority=1
        )

        assert rule.rule_id == "rule_1"
        assert rule.condition == "emotional_impact > 0.9"
        assert rule.action == "STORE"
        assert rule.priority == 1

    def test_decision_rule_invalid_action(self):
        """Test that invalid action raises error."""
        with pytest.raises(ValueError):
            DecisionRule(
                rule_id="rule_1",
                condition="novelty > 0.5",
                action="INVALID"  # Must be STORE or SKIP
            )


class TestTenantSalienceConfigValidation:
    """Test validation of TenantSalienceConfig."""

    def test_weights_must_sum_to_one(self):
        """Test that component weights must sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            TenantSalienceConfig(
                tenant_id="tenant-123",
                config_name="bad_weights",
                components=[
                    SalienceComponent(
                        name="comp1",
                        weight=0.3,
                        scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY
                    ),
                    SalienceComponent(
                        name="comp2",
                        weight=0.3,
                        scoring_function=ScoringFunctionType.KEYWORD_MATCH
                    ),
                    # Missing 0.4 weight
                ]
            )

    def test_weights_sum_with_float_rounding(self):
        """Test that weights are validated with float rounding tolerance."""
        # Should succeed with 0.99-1.01 range
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="rounding_test",
            components=[
                SalienceComponent(
                    name="comp1",
                    weight=0.333333,
                    scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY
                ),
                SalienceComponent(
                    name="comp2",
                    weight=0.333333,
                    scoring_function=ScoringFunctionType.KEYWORD_MATCH
                ),
                SalienceComponent(
                    name="comp3",
                    weight=0.333334,
                    scoring_function=ScoringFunctionType.LLM_SCORED
                ),
            ]
        )
        assert len(config.components) == 3

    def test_unique_component_names(self):
        """Test that component names must be unique."""
        with pytest.raises(ValueError, match="must be unique"):
            TenantSalienceConfig(
                tenant_id="tenant-123",
                config_name="duplicate_names",
                components=[
                    SalienceComponent(
                        name="novelty",
                        weight=0.5,
                        scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY
                    ),
                    SalienceComponent(
                        name="novelty",  # Duplicate!
                        weight=0.5,
                        scoring_function=ScoringFunctionType.KEYWORD_MATCH
                    ),
                ]
            )

    def test_unique_rule_ids(self):
        """Test that decision rule IDs must be unique."""
        with pytest.raises(ValueError, match="must be unique"):
            TenantSalienceConfig(
                tenant_id="tenant-123",
                config_name="duplicate_rules",
                components=[
                    SalienceComponent(
                        name="score",
                        weight=1.0,
                        scoring_function=ScoringFunctionType.LLM_SCORED
                    ),
                ],
                decision_rules=[
                    DecisionRule(rule_id="rule_1", condition="score > 0.9", action="STORE", priority=1),
                    DecisionRule(rule_id="rule_1", condition="score < 0.5", action="SKIP", priority=2),
                ]
            )

    def test_custom_mode_54_example(self):
        """Test the custom mode 54 example from requirements."""
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="my_custom_score_54",
            components=[
                SalienceComponent(
                    name="novelty",
                    weight=0.4,
                    description="How new?",
                    scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY,
                    scoring_config={"target": "existing_memories"}
                ),
                SalienceComponent(
                    name="emotional_impact",
                    weight=0.3,
                    description="How emotionally significant?",
                    scoring_function=ScoringFunctionType.LLM_SCORED,
                    scoring_config={"prompt": "Score emotional impact..."}
                ),
                SalienceComponent(
                    name="technical_depth",
                    weight=0.3,
                    description="Technical complexity",
                    scoring_function=ScoringFunctionType.KEYWORD_MATCH,
                    scoring_config={"keywords": ["algorithm", "optimization"]}
                ),
            ],
            threshold_config=AdaptiveThresholdConfig(
                strategy=ThresholdStrategy.PERCENTILE,
                recent_facts_window=150,
                percentile=70
            ),
            decision_rules=[
                DecisionRule(
                    rule_id="rule_1",
                    name="Always keep high emotional",
                    condition="emotional_impact > 0.85",
                    action="STORE",
                    priority=1
                )
            ]
        )

        assert len(config.components) == 3
        assert config.config_name == "my_custom_score_54"
        assert config.decision_rules[0].condition == "emotional_impact > 0.85"


class TestSalienceCalculator:
    """Test the SalienceCalculator service."""

    def test_compute_salience_basic(self):
        """Test basic salience computation."""
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="test_config",
            components=[
                SalienceComponent(
                    name="novelty",
                    weight=0.5,
                    scoring_function=ScoringFunctionType.KEYWORD_MATCH,
                    scoring_config={"keywords": ["novel", "new", "amazing"]}
                ),
                SalienceComponent(
                    name="importance",
                    weight=0.5,
                    scoring_function=ScoringFunctionType.LENGTH_BONUS,
                    scoring_config={"min_length": 10, "max_length": 200}
                ),
            ],
            threshold_config=AdaptiveThresholdConfig(
                strategy=ThresholdStrategy.ABSOLUTE,
                absolute_threshold=0.5
            )
        )

        fact = "This is a novel and amazing discovery with significant importance."

        calculator = SalienceCalculator(config, fact, "tenant-123")
        salience_score, decision, component_scores = calculator.compute_salience()

        assert 0 <= salience_score <= 1
        assert decision in ["STORE", "SKIP"]
        assert "novelty" in component_scores
        assert "importance" in component_scores
        assert 0 <= component_scores["novelty"] <= 1
        assert 0 <= component_scores["importance"] <= 1

    def test_keyword_match_scoring(self):
        """Test keyword match scoring function."""
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="keyword_test",
            components=[
                SalienceComponent(
                    name="technical",
                    weight=1.0,
                    scoring_function=ScoringFunctionType.KEYWORD_MATCH,
                    scoring_config={"keywords": ["algorithm", "optimization", "performance"]}
                ),
            ]
        )

        # Fact with 2 out of 3 keywords
        fact = "Algorithm optimization is critical for performance tuning."
        calculator = SalienceCalculator(config, fact, "tenant-123")
        _, _, scores = calculator.compute_salience()

        # Should be 3/3 = 1.0 (all keywords match)
        assert scores["technical"] == 1.0

    def test_keyword_match_no_matches(self):
        """Test keyword match with no matching keywords."""
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="keyword_test",
            components=[
                SalienceComponent(
                    name="technical",
                    weight=1.0,
                    scoring_function=ScoringFunctionType.KEYWORD_MATCH,
                    scoring_config={"keywords": ["blockchain", "quantum", "cryptocurrency"]}
                ),
            ]
        )

        # Fact with no matching keywords
        fact = "I had a nice lunch today at my favorite restaurant."
        calculator = SalienceCalculator(config, fact, "tenant-123")
        _, _, scores = calculator.compute_salience()

        # Should be 0/3 = 0.0
        assert scores["technical"] == 0.0

    def test_length_bonus_scoring(self):
        """Test length bonus scoring function."""
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="length_test",
            components=[
                SalienceComponent(
                    name="depth",
                    weight=1.0,
                    scoring_function=ScoringFunctionType.LENGTH_BONUS,
                    scoring_config={"min_length": 50, "max_length": 150}
                ),
            ]
        )

        # Test below minimum
        calc_short = SalienceCalculator(config, "Short fact", "tenant-123")
        _, _, scores = calc_short.compute_salience()
        assert scores["depth"] == 0.0

        # Test at maximum
        long_fact = "x" * 150
        calc_long = SalienceCalculator(config, long_fact, "tenant-123")
        _, _, scores = calc_long.compute_salience()
        assert scores["depth"] == 1.0

        # Test in middle
        medium_fact = "x" * 100  # Between 50 and 150
        calc_medium = SalienceCalculator(config, medium_fact, "tenant-123")
        _, _, scores = calc_medium.compute_salience()
        # (100 - 50) / (150 - 50) = 50 / 100 = 0.5
        assert scores["depth"] == 0.5

    def test_threshold_mean_multiplier(self):
        """Test mean_multiplier threshold strategy."""
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="mean_mult_test",
            components=[
                SalienceComponent(
                    name="score",
                    weight=1.0,
                    scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY
                ),
            ],
            threshold_config=AdaptiveThresholdConfig(
                strategy=ThresholdStrategy.MEAN_MULTIPLIER,
                multiplier=0.8,
                recent_facts_window=100
            )
        )

        recent_scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        calculator = SalienceCalculator(config, "Test fact", "tenant-123", recent_scores)

        _, _, _ = calculator.compute_salience()
        log = calculator.get_log()

        # Mean of [0.5, 0.6, 0.7, 0.8, 0.9] = 0.7
        # Threshold = 0.7 * 0.8 = 0.56
        assert log["threshold"] == pytest.approx(0.56, rel=0.01)

    def test_threshold_percentile(self):
        """Test percentile threshold strategy."""
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="percentile_test",
            components=[
                SalienceComponent(
                    name="score",
                    weight=1.0,
                    scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY
                ),
            ],
            threshold_config=AdaptiveThresholdConfig(
                strategy=ThresholdStrategy.PERCENTILE,
                percentile=60,
                recent_facts_window=100
            )
        )

        recent_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        calculator = SalienceCalculator(config, "Test fact", "tenant-123", recent_scores)

        _, _, _ = calculator.compute_salience()
        log = calculator.get_log()

        # 60th percentile should be around 0.54
        assert 0.5 <= log["threshold"] <= 0.6

    def test_threshold_absolute(self):
        """Test absolute threshold strategy."""
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="absolute_test",
            components=[
                SalienceComponent(
                    name="score",
                    weight=1.0,
                    scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY
                ),
            ],
            threshold_config=AdaptiveThresholdConfig(
                strategy=ThresholdStrategy.ABSOLUTE,
                absolute_threshold=0.75
            )
        )

        calculator = SalienceCalculator(config, "Test fact", "tenant-123")
        _, _, _ = calculator.compute_salience()
        log = calculator.get_log()

        assert log["threshold"] == 0.75

    def test_decision_rules_priority_order(self):
        """Test that decision rules are evaluated in priority order."""
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="rule_test",
            components=[
                SalienceComponent(
                    name="score1",
                    weight=0.5,
                    scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY
                ),
                SalienceComponent(
                    name="score2",
                    weight=0.5,
                    scoring_function=ScoringFunctionType.KEYWORD_MATCH,
                    scoring_config={"keywords": ["important"]}
                ),
            ],
            decision_rules=[
                DecisionRule(
                    rule_id="rule_high_priority",
                    condition="score2 > 0.5",
                    action="STORE",
                    priority=1  # Evaluated first
                ),
                DecisionRule(
                    rule_id="rule_low_priority",
                    condition="score1 < 0.3",
                    action="SKIP",
                    priority=2  # Evaluated second
                ),
            ]
        )

        # Fact with "important" keyword -> score2 will be high
        fact = "This is important information."
        calculator = SalienceCalculator(config, fact, "tenant-123")
        _, decision, _ = calculator.compute_salience()

        # Should match first rule and STORE
        assert decision == "STORE"

    def test_decision_rules_condition_context(self):
        """Test that decision rules have access to component scores."""
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="context_test",
            components=[
                SalienceComponent(
                    name="emotional",
                    weight=1.0,
                    scoring_function=ScoringFunctionType.KEYWORD_MATCH,
                    scoring_config={"keywords": ["love", "hate", "amazing", "terrible"]}
                ),
            ],
            decision_rules=[
                DecisionRule(
                    rule_id="emotional_rule",
                    condition="emotional > 0.5 and word_count > 5",
                    action="STORE",
                    priority=1
                ),
            ]
        )

        # Fact with emotional keywords
        fact = "I love this amazing discovery"
        calculator = SalienceCalculator(config, fact, "tenant-123")
        _, decision, _ = calculator.compute_salience()

        # emotional score = 2/4 = 0.5, word_count = 5
        # Condition: 0.5 > 0.5 and 5 > 5 = False
        # So it should SKIP
        assert decision == "SKIP"

    def test_decision_rules_fallback_to_threshold(self):
        """Test fallback to threshold if no rules match."""
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="fallback_test",
            components=[
                SalienceComponent(
                    name="score",
                    weight=1.0,
                    scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY
                ),
            ],
            threshold_config=AdaptiveThresholdConfig(
                strategy=ThresholdStrategy.ABSOLUTE,
                absolute_threshold=0.5
            ),
            decision_rules=[
                DecisionRule(
                    rule_id="never_matches",
                    condition="False",  # Never matches
                    action="SKIP",
                    priority=1
                ),
            ]
        )

        calculator = SalienceCalculator(config, "Test fact", "tenant-123")
        salience_score, decision, _ = calculator.compute_salience()

        # Should fall back to threshold comparison
        expected_decision = "STORE" if salience_score >= 0.5 else "SKIP"
        assert decision == expected_decision

    def test_unknown_scoring_function(self):
        """Test that unknown scoring function raises error."""
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="bad_func",
            components=[
                SalienceComponent(
                    name="unknown",
                    weight=1.0,
                    scoring_function="unknown_function"  # type: ignore
                ),
            ]
        )

        calculator = SalienceCalculator(config, "Test", "tenant-123")
        with pytest.raises(ValueError, match="Unknown scoring function"):
            calculator.compute_salience()

    def test_computation_log(self):
        """Test that computation log contains detailed information."""
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="log_test",
            components=[
                SalienceComponent(
                    name="novelty",
                    weight=1.0,
                    scoring_function=ScoringFunctionType.KEYWORD_MATCH,
                    scoring_config={"keywords": ["new", "novel"]}
                ),
            ]
        )

        fact = "This is a novel approach."
        calculator = SalienceCalculator(config, fact, "tenant-123")
        _, _, _ = calculator.compute_salience()

        log = calculator.get_log()

        assert "tenant_id" in log
        assert "config_name" in log
        assert "component_scores" in log
        assert "threshold" in log
        assert "final_decision" in log
        assert "decision_rules_evaluated" in log
        assert log["tenant_id"] == "tenant-123"
        assert log["config_name"] == "log_test"


class TestSalienceConfigStore:
    """Test storage backends for salience configurations."""

    def test_in_memory_store_create_config(self):
        """Test creating config in memory store."""
        store = InMemorySalienceConfigStore()
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="test_config",
            components=[
                SalienceComponent(
                    name="score",
                    weight=1.0,
                    scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY
                ),
            ]
        )

        # Create should be async
        import asyncio
        config_id = asyncio.run(store.create_config(config))

        assert config_id is not None
        assert len(config_id) > 0

    def test_in_memory_store_get_config(self):
        """Test retrieving config from memory store."""
        store = InMemorySalienceConfigStore()
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="test_config",
            components=[
                SalienceComponent(
                    name="score",
                    weight=1.0,
                    scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY
                ),
            ]
        )

        import asyncio

        asyncio.run(store.create_config(config))
        retrieved = asyncio.run(store.get_config("tenant-123", "test_config"))

        assert retrieved is not None
        assert retrieved.config_name == "test_config"
        assert len(retrieved.components) == 1

    def test_in_memory_store_list_configs(self):
        """Test listing configs from memory store."""
        store = InMemorySalienceConfigStore()

        import asyncio

        for i in range(3):
            config = TenantSalienceConfig(
                tenant_id="tenant-123",
                config_name=f"config_{i}",
                components=[
                    SalienceComponent(
                        name="score",
                        weight=1.0,
                        scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY
                    ),
                ]
            )
            asyncio.run(store.create_config(config))

        configs = asyncio.run(store.list_configs("tenant-123"))
        assert len(configs) == 3

    def test_in_memory_store_update_config(self):
        """Test updating config in memory store."""
        store = InMemorySalienceConfigStore()
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="test_config",
            components=[
                SalienceComponent(
                    name="score",
                    weight=1.0,
                    scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY
                ),
            ]
        )

        import asyncio

        asyncio.run(store.create_config(config))

        # Update
        config.components[0].weight = 0.8
        success = asyncio.run(store.update_config("tenant-123", "test_config", config))

        assert success
        retrieved = asyncio.run(store.get_config("tenant-123", "test_config"))
        assert retrieved.components[0].weight == 0.8

    def test_in_memory_store_delete_config(self):
        """Test deleting config from memory store."""
        store = InMemorySalienceConfigStore()
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="test_config",
            components=[
                SalienceComponent(
                    name="score",
                    weight=1.0,
                    scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY
                ),
            ]
        )

        import asyncio

        asyncio.run(store.create_config(config))
        success = asyncio.run(store.delete_config("tenant-123", "test_config"))

        assert success
        retrieved = asyncio.run(store.get_config("tenant-123", "test_config"))
        assert retrieved is None

    def test_file_based_store_persistence(self):
        """Test that file-based store persists to disk."""
        temp_dir = tempfile.mkdtemp()

        try:
            store = FileBasedSalienceConfigStore(temp_dir)
            config = TenantSalienceConfig(
                tenant_id="tenant-123",
                config_name="persistent_config",
                components=[
                    SalienceComponent(
                        name="score",
                        weight=1.0,
                        scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY
                    ),
                ]
            )

            import asyncio

            asyncio.run(store.create_config(config))

            # Create new store instance from same path
            store2 = FileBasedSalienceConfigStore(temp_dir)
            retrieved = asyncio.run(store2.get_config("tenant-123", "persistent_config"))

            assert retrieved is not None
            assert retrieved.config_name == "persistent_config"

        finally:
            shutil.rmtree(temp_dir)

    def test_duplicate_config_name_raises_error(self):
        """Test that creating duplicate config names raises error."""
        store = InMemorySalienceConfigStore()
        config = TenantSalienceConfig(
            tenant_id="tenant-123",
            config_name="test_config",
            components=[
                SalienceComponent(
                    name="score",
                    weight=1.0,
                    scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY
                ),
            ]
        )

        import asyncio

        asyncio.run(store.create_config(config))

        with pytest.raises(ValueError, match="already exists"):
            asyncio.run(store.create_config(config))


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_research_mode_example(self):
        """Test the research mode example from requirements."""
        config = TenantSalienceConfig(
            tenant_id="research-lab",
            config_name="research_neural_nets",
            components=[
                SalienceComponent(
                    name="novelty",
                    weight=0.2,
                    scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY,
                    scoring_config={}
                ),
                SalienceComponent(
                    name="relevance_to_task",
                    weight=0.3,
                    scoring_function=ScoringFunctionType.KEYWORD_MATCH,
                    scoring_config={"keywords": ["neural", "network", "deep", "learning"]}
                ),
                SalienceComponent(
                    name="technical_depth",
                    weight=0.2,
                    scoring_function=ScoringFunctionType.LENGTH_BONUS,
                    scoring_config={"min_length": 100}
                ),
                SalienceComponent(
                    name="citation_importance",
                    weight=0.15,
                    scoring_function=ScoringFunctionType.FREQUENCY,
                    scoring_config={}
                ),
                SalienceComponent(
                    name="recency_6_months",
                    weight=0.15,
                    scoring_function=ScoringFunctionType.LIVENESS,
                    scoring_config={"days": 180}
                ),
            ]
        )

        fact = "Deep neural networks with convolutional architectures have revolutionized computer vision research."
        calculator = SalienceCalculator(config, fact, "research-lab")
        salience_score, decision, component_scores = calculator.compute_salience()

        assert 0 <= salience_score <= 1
        assert decision in ["STORE", "SKIP"]
        assert len(component_scores) == 5
        assert "novelty" in component_scores
        assert "relevance_to_task" in component_scores

    def test_full_workflow_with_storage(self):
        """Test complete workflow: create config, compute, log."""
        store = InMemorySalienceConfigStore()

        config = TenantSalienceConfig(
            tenant_id="user-1",
            config_name="my_config",
            components=[
                SalienceComponent(
                    name="importance",
                    weight=1.0,
                    scoring_function=ScoringFunctionType.KEYWORD_MATCH,
                    scoring_config={"keywords": ["critical", "important", "urgent"]}
                ),
            ],
            threshold_config=AdaptiveThresholdConfig(
                strategy=ThresholdStrategy.ABSOLUTE,
                absolute_threshold=0.3
            )
        )

        import asyncio

        config_id = asyncio.run(store.create_config(config))
        assert config_id

        retrieved = asyncio.run(store.get_config("user-1", "my_config"))
        assert retrieved

        # Compute salience
        calculator = SalienceCalculator(retrieved, "This is a critical issue", "user-1")
        score, decision, scores = calculator.compute_salience()

        # Log decision
        log_id = asyncio.run(store.log_decision(
            tenant_id="user-1",
            fact_id="fact-1",
            config_id=config_id,
            config_snapshot=retrieved.model_dump(),
            component_scores=scores,
            final_salience_score=score,
            threshold_value=0.3,
            decision_rule_matched=None,
            decision=decision,
            reasoning="Matched keyword match threshold"
        ))

        assert log_id

        # Retrieve logs
        logs = asyncio.run(store.get_audit_logs("user-1", "my_config"))
        assert len(logs) == 1
        assert logs[0].decision in ["STORE", "SKIP"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
