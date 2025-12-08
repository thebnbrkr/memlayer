"""
Flexible Salience Configuration System for Memlayer

This module provides a completely customizable salience configuration system
with zero presets or templates. Users can define ANY components, weights,
threshold calculation strategies, and decision rules.

Example configurations included as docstrings demonstrate the flexibility.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class ScoringFunctionType(str, Enum):
    """Available scoring function types for salience components."""
    EMBEDDING_SIMILARITY = "embedding_similarity"
    KEYWORD_MATCH = "keyword_match"
    LENGTH_BONUS = "length_bonus"
    FREQUENCY = "frequency"
    LIVENESS = "liveness"
    LLM_SCORED = "llm_scored"
    CUSTOM_PYTHON = "custom_python"
    WEIGHTED_SUM = "weighted_sum"
    BOOLEAN_GATE = "boolean_gate"


class ThresholdStrategy(str, Enum):
    """Available threshold calculation strategies."""
    MEAN_MULTIPLIER = "mean_multiplier"
    PERCENTILE = "percentile"
    EXPONENTIAL_DECAY = "exponential_decay"
    ABSOLUTE = "absolute"
    DYNAMIC_PERCENTILE = "dynamic_percentile"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


class SalienceComponent(BaseModel):
    """
    A single component in the salience score calculation.

    Users can define unlimited components with custom names, weights, and
    scoring functions. The system is agnostic to component purpose.

    Example:
        {
            "name": "emotional_impact",
            "weight": 0.3,
            "description": "How emotionally significant is this fact?",
            "scoring_function": "llm_scored",
            "scoring_config": {
                "prompt": "Rate the emotional significance 0-1: {fact}"
            }
        }
    """
    name: str = Field(..., description="User-defined component name (e.g., 'novelty', 'emotional_impact')")
    weight: float = Field(..., ge=0.0, le=1.0, description="Weight in final score (0.0-1.0)")
    description: str = Field(default="", description="Human-readable description of this component")
    scoring_function: ScoringFunctionType = Field(..., description="How to compute this component's score")
    scoring_config: Dict[str, Any] = Field(default_factory=dict, description="Config specific to scoring function")


class AdaptiveThresholdConfig(BaseModel):
    """
    Configuration for adaptive threshold calculation.

    Different strategies can be combined or used independently.
    The system uses the strategy specified to compute the final threshold.
    """
    strategy: ThresholdStrategy = Field(
        default=ThresholdStrategy.MEAN_MULTIPLIER,
        description="Which threshold strategy to use"
    )
    recent_facts_window: int = Field(
        default=100,
        ge=1,
        description="Number of recent facts to consider for mean/percentile"
    )
    multiplier: float = Field(
        default=0.8,
        description="Multiplier for mean_multiplier strategy"
    )
    percentile: float = Field(
        default=60,
        ge=0.0,
        le=100.0,
        description="Percentile for percentile strategy (0-100)"
    )
    absolute_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fixed threshold for absolute strategy"
    )
    decay_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Decay rate for exponential_decay strategy"
    )


class DecisionRule(BaseModel):
    """
    A rule that determines whether to STORE or SKIP a fact.

    Rules are evaluated in priority order. The first matching rule wins.
    Conditions are boolean expressions with access to computed scores.

    Available variables in condition:
    - salience_score: float (final weighted score)
    - threshold: float (computed threshold)
    - novelty, importance, etc.: float (component scores - names from components)
    - word_count: int (length of fact)

    Example:
        {
            "rule_id": "high_emotional",
            "name": "Always store high emotional facts",
            "condition": "emotional_impact > 0.9",
            "action": "STORE",
            "priority": 1
        }
    """
    rule_id: str = Field(..., description="Unique identifier for this rule")
    name: str = Field(default="", description="Human-readable rule name")
    condition: str = Field(
        ...,
        description="Boolean expression (e.g., 'novelty > 0.9 and importance > 0.5')"
    )
    action: str = Field(
        ...,
        pattern="^(STORE|SKIP)$",
        description="Action if condition matches: STORE or SKIP"
    )
    priority: int = Field(
        default=100,
        ge=1,
        description="Lower priority wins (evaluated first)"
    )


class TenantSalienceConfig(BaseModel):
    """
    Complete salience configuration for a tenant.

    This is the root configuration object. Users can have multiple configs
    per tenant, and switch between them or use different ones for different
    fact types.

    Validation:
    - Component weights must sum to 1.0 (allowing 0.99-1.01 for float rounding)
    - All component names must be unique
    - All decision rule IDs must be unique

    Example - Custom Mode 54:
        {
            "tenant_id": "tenant-123",
            "config_name": "my_custom_score_54",
            "components": [
                {
                    "name": "novelty",
                    "weight": 0.4,
                    "description": "How new compared to existing memories?",
                    "scoring_function": "embedding_similarity",
                    "scoring_config": {"target": "existing_memories"}
                },
                {
                    "name": "emotional_impact",
                    "weight": 0.3,
                    "description": "How emotionally significant?",
                    "scoring_function": "llm_scored",
                    "scoring_config": {"prompt": "Score emotional impact 0-1"}
                },
                {
                    "name": "technical_depth",
                    "weight": 0.3,
                    "description": "Technical complexity and depth",
                    "scoring_function": "keyword_match",
                    "scoring_config": {"keywords": ["algorithm", "optimization", "architecture"]}
                }
            ],
            "threshold_config": {
                "strategy": "percentile",
                "recent_facts_window": 150,
                "percentile": 70
            },
            "decision_rules": [
                {
                    "rule_id": "rule_1",
                    "name": "Always keep high emotional facts",
                    "condition": "emotional_impact > 0.85",
                    "action": "STORE",
                    "priority": 1
                },
                {
                    "rule_id": "rule_2",
                    "name": "Skip low novelty",
                    "condition": "novelty < 0.2",
                    "action": "SKIP",
                    "priority": 2
                }
            ],
            "is_active": true
        }

    Example - Research Mode:
        {
            "config_name": "research_neural_nets",
            "components": [
                {
                    "name": "novelty",
                    "weight": 0.2,
                    "scoring_function": "embedding_similarity",
                    "scoring_config": {}
                },
                {
                    "name": "relevance_to_task",
                    "weight": 0.3,
                    "scoring_function": "llm_scored",
                    "scoring_config": {"task": "neural network research"}
                },
                {
                    "name": "technical_depth",
                    "weight": 0.2,
                    "scoring_function": "length_bonus",
                    "scoring_config": {"min_length": 100}
                },
                {
                    "name": "citation_importance",
                    "weight": 0.15,
                    "scoring_function": "frequency",
                    "scoring_config": {}
                },
                {
                    "name": "recency_6_months",
                    "weight": 0.15,
                    "scoring_function": "liveness",
                    "scoring_config": {"days": 180}
                }
            ],
            "threshold_config": {
                "strategy": "dynamic_percentile",
                "recent_facts_window": 200,
                "percentile": 50
            },
            "decision_rules": []
        }
    """
    tenant_id: str = Field(..., description="Tenant identifier")
    config_name: str = Field(..., description="User-defined configuration name (e.g., 'my_custom_score_54')")
    components: List[SalienceComponent] = Field(..., min_items=1, description="List of scoring components")
    threshold_config: AdaptiveThresholdConfig = Field(
        default_factory=AdaptiveThresholdConfig,
        description="Threshold calculation configuration"
    )
    decision_rules: List[DecisionRule] = Field(
        default_factory=list,
        description="List of decision rules (optional)"
    )
    is_active: bool = Field(default=True, description="Whether this config is currently active")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Last update timestamp")

    @field_validator("components")
    @classmethod
    def validate_unique_component_names(cls, components: List[SalienceComponent]) -> List[SalienceComponent]:
        """Ensure all component names are unique."""
        names = [c.name for c in components]
        if len(names) != len(set(names)):
            raise ValueError("Component names must be unique")
        return components

    @field_validator("components")
    @classmethod
    def validate_weights_sum_to_one(cls, components: List[SalienceComponent]) -> List[SalienceComponent]:
        """Validate that component weights sum to 1.0 (allow 0.99-1.01 for float rounding)."""
        total_weight = sum(c.weight for c in components)
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(
                f"Component weights must sum to 1.0 (allowing 0.99-1.01 for float rounding). "
                f"Got sum of {total_weight:.4f}"
            )
        return components

    @field_validator("decision_rules")
    @classmethod
    def validate_unique_rule_ids(cls, rules: List[DecisionRule]) -> List[DecisionRule]:
        """Ensure all decision rule IDs are unique."""
        if not rules:
            return rules
        ids = [r.rule_id for r in rules]
        if len(ids) != len(set(ids)):
            raise ValueError("Decision rule IDs must be unique")
        return rules

    @field_validator("decision_rules")
    @classmethod
    def validate_rules_sorted_by_priority(cls, rules: List[DecisionRule]) -> List[DecisionRule]:
        """Validate that rules are sorted by priority (lower first)."""
        if len(rules) > 1:
            sorted_rules = sorted(rules, key=lambda r: r.priority)
            # Just validate the rule is valid, don't enforce sorting
            # (rules are sorted during evaluation)
        return rules

    class Config:
        json_schema_extra = {
            "example": {
                "tenant_id": "tenant-xyz",
                "config_name": "custom_config_001",
                "components": [
                    {
                        "name": "novelty",
                        "weight": 0.5,
                        "description": "Novelty score",
                        "scoring_function": "embedding_similarity",
                        "scoring_config": {}
                    },
                    {
                        "name": "importance",
                        "weight": 0.5,
                        "description": "Importance score",
                        "scoring_function": "llm_scored",
                        "scoring_config": {}
                    }
                ],
                "threshold_config": {
                    "strategy": "mean_multiplier",
                    "multiplier": 0.8
                },
                "decision_rules": [],
                "is_active": True
            }
        }
