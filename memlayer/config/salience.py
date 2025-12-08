"""
Fully flexible salience configuration system for Memlayer.

This module provides ZERO presets and ZERO templates - complete freedom for users
to define their own salience scoring components, thresholds, and decision rules.

Example 1: Custom Mode 54
--------------------------
{
  "config_name": "my_custom_score_54",
  "components": [
    {
      "name": "novelty",
      "weight": 0.4,
      "description": "How new is this information?",
      "scoring_function": "embedding_similarity",
      "scoring_config": {"target": "existing_memories"}
    },
    {
      "name": "emotional_impact",
      "weight": 0.3,
      "scoring_function": "llm_scored",
      "scoring_config": {"prompt": "Score emotional impact 0-1..."}
    },
    {
      "name": "technical_depth",
      "weight": 0.3,
      "scoring_function": "keyword_match",
      "scoring_config": {"keywords": ["algorithm", "optimization", "neural"]}
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
      "name": "Always keep high emotional",
      "condition": "emotional_impact > 0.85",
      "action": "STORE",
      "priority": 1
    }
  ]
}

Example 2: Research Mode
-------------------------
{
  "config_name": "research_neural_nets",
  "components": [
    {"name": "novelty", "weight": 0.2, "scoring_function": "embedding_similarity", ...},
    {"name": "relevance_to_task", "weight": 0.3, "scoring_function": "llm_scored", ...},
    {"name": "technical_depth", "weight": 0.2, "scoring_function": "keyword_match", ...},
    {"name": "citation_importance", "weight": 0.15, "scoring_function": "frequency", ...},
    {"name": "recency_6_months", "weight": 0.15", "scoring_function": "liveness", ...}
  ],
  "threshold_config": {
    "strategy": "mean_multiplier",
    "recent_facts_window": 100,
    "multiplier": 0.8
  },
  "decision_rules": []
}
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
import uuid


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
    A single component in the salience scoring system.

    Users can define ANY component with ANY name and ANY scoring function.
    No restrictions, complete freedom.
    """

    name: str = Field(
        ...,
        description="User-defined name for this component (e.g., 'novelty', 'emotional_impact', 'my_custom_score')"
    )

    weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weight for this component (0.0-1.0). All weights must sum to 1.0."
    )

    description: str = Field(
        "",
        description="Human-readable description of what this component measures"
    )

    scoring_function: ScoringFunctionType = Field(
        ...,
        description="The function used to compute this component's score"
    )

    scoring_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration parameters for the scoring function (e.g., keywords, prompts, thresholds)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "novelty",
                "weight": 0.4,
                "description": "Measures how new this information is compared to existing memories",
                "scoring_function": "embedding_similarity",
                "scoring_config": {
                    "target": "existing_memories",
                    "similarity_threshold": 0.85
                }
            }
        }


class AdaptiveThresholdConfig(BaseModel):
    """
    Configuration for adaptive threshold calculation.

    Users can choose ANY threshold strategy and configure it ANY way they want.
    """

    strategy: ThresholdStrategy = Field(
        ...,
        description="The strategy to use for calculating the salience threshold"
    )

    recent_facts_window: int = Field(
        default=100,
        ge=1,
        description="Number of recent facts to consider for adaptive strategies"
    )

    multiplier: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Multiplier for mean_multiplier strategy (default 0.8 = 80% of mean)"
    )

    percentile: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Percentile value for percentile-based strategies (0-100)"
    )

    absolute_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fixed threshold value for absolute strategy (0.0-1.0)"
    )

    decay_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Decay rate for exponential_decay strategy"
    )

    confidence_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for confidence in confidence_weighted strategy"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "strategy": "percentile",
                "recent_facts_window": 150,
                "percentile": 70.0
            }
        }


class DecisionRule(BaseModel):
    """
    A decision rule that can override the threshold-based decision.

    Rules are evaluated in priority order. Users can write ANY boolean expression.
    """

    rule_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this rule"
    )

    name: str = Field(
        ...,
        description="Human-readable name for this rule"
    )

    condition: str = Field(
        ...,
        description=(
            "Boolean expression to evaluate (e.g., 'emotional_impact > 0.85 and novelty > 0.5'). "
            "Available variables: salience_score, threshold, and all component names."
        )
    )

    action: str = Field(
        ...,
        pattern="^(STORE|SKIP)$",
        description="Action to take if condition is true: STORE or SKIP"
    )

    priority: int = Field(
        ...,
        ge=1,
        description="Priority order (lower number = higher priority, evaluated first)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "rule_id": "rule_1",
                "name": "Always keep high emotional impact",
                "condition": "emotional_impact > 0.85",
                "action": "STORE",
                "priority": 1
            }
        }


class TenantSalienceConfig(BaseModel):
    """
    Complete salience configuration for a tenant.

    This is the top-level config that users create. ZERO restrictions on what they can configure.
    They can have 1 component or 100 components, use any weights, any thresholds, any rules.
    The ONLY validation: weights must sum to 1.0.
    """

    config_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this configuration"
    )

    tenant_id: str = Field(
        ...,
        description="Tenant identifier (for multi-tenancy)"
    )

    config_name: str = Field(
        ...,
        description="User-defined name for this config (e.g., 'my_custom_score_54', 'research_mode')"
    )

    components: List[SalienceComponent] = Field(
        ...,
        min_length=1,
        description="List of salience components. Can have ANY number of components with ANY names."
    )

    threshold_config: AdaptiveThresholdConfig = Field(
        ...,
        description="Configuration for threshold calculation"
    )

    decision_rules: List[DecisionRule] = Field(
        default_factory=list,
        description="Optional decision rules that can override threshold-based decisions"
    )

    is_active: bool = Field(
        default=True,
        description="Whether this configuration is currently active"
    )

    created_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when this config was created"
    )

    updated_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when this config was last updated"
    )

    created_by_user_id: Optional[str] = Field(
        default=None,
        description="User ID of the creator"
    )

    @field_validator('components')
    @classmethod
    def validate_weights_sum(cls, components: List[SalienceComponent]) -> List[SalienceComponent]:
        """Validate that component weights sum to 1.0 (allowing 0.99-1.01 for float rounding)."""
        total_weight = sum(c.weight for c in components)

        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(
                f"Component weights must sum to 1.0 (got {total_weight:.4f}). "
                f"Allowed range: 0.99-1.01 for float rounding tolerance."
            )

        return components

    @field_validator('decision_rules')
    @classmethod
    def validate_unique_priorities(cls, rules: List[DecisionRule]) -> List[DecisionRule]:
        """Warn if multiple rules have the same priority (not an error, but might be unintended)."""
        if rules:
            priorities = [r.priority for r in rules]
            if len(priorities) != len(set(priorities)):
                # Just a warning, not an error - users might intentionally want this
                pass
        return rules

    @model_validator(mode='after')
    def sort_decision_rules(self) -> 'TenantSalienceConfig':
        """Sort decision rules by priority (lowest number first)."""
        if self.decision_rules:
            self.decision_rules = sorted(self.decision_rules, key=lambda r: r.priority)
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "tenant_id": "customer_123",
                "config_name": "my_custom_score_54",
                "components": [
                    {
                        "name": "novelty",
                        "weight": 0.4,
                        "description": "How new?",
                        "scoring_function": "embedding_similarity",
                        "scoring_config": {"target": "existing_memories"}
                    },
                    {
                        "name": "emotional_impact",
                        "weight": 0.3,
                        "scoring_function": "llm_scored",
                        "scoring_config": {"prompt": "Score emotional impact..."}
                    },
                    {
                        "name": "technical_depth",
                        "weight": 0.3,
                        "scoring_function": "keyword_match",
                        "scoring_config": {"keywords": ["algorithm", "optimization"]}
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
                        "name": "Always keep high emotional",
                        "condition": "emotional_impact > 0.85",
                        "action": "STORE",
                        "priority": 1
                    }
                ],
                "is_active": True
            }
        }


class SalienceComputationLog(BaseModel):
    """
    Log entry for a single salience computation.

    Stores complete audit trail of how a decision was made.
    """

    log_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this log entry"
    )

    tenant_id: str = Field(..., description="Tenant identifier")

    fact_id: Optional[str] = Field(None, description="ID of the fact being scored")

    fact_text: str = Field(..., description="The actual fact text that was scored")

    config_id: str = Field(..., description="ID of the config that was used")

    config_snapshot: Dict[str, Any] = Field(
        ...,
        description="Complete snapshot of the config at time of computation"
    )

    component_scores: Dict[str, float] = Field(
        ...,
        description="Individual scores for each component {component_name: score}"
    )

    final_salience_score: float = Field(
        ...,
        description="Final weighted salience score (0.0-1.0)"
    )

    threshold_value: float = Field(
        ...,
        description="Threshold value that was computed/used"
    )

    decision_rule_matched: Optional[str] = Field(
        None,
        description="Name of the decision rule that was matched (if any)"
    )

    decision: str = Field(
        ...,
        pattern="^(STORE|SKIP)$",
        description="Final decision: STORE or SKIP"
    )

    reasoning: str = Field(
        ...,
        description="Human-readable explanation of why this decision was made"
    )

    computation_time_ms: float = Field(
        ...,
        description="Time taken to compute salience (milliseconds)"
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of computation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "tenant_id": "customer_123",
                "fact_text": "Alice works at TechCorp as a software engineer",
                "config_id": "cfg_abc123",
                "component_scores": {
                    "novelty": 0.85,
                    "emotional_impact": 0.3,
                    "technical_depth": 0.6
                },
                "final_salience_score": 0.625,
                "threshold_value": 0.55,
                "decision": "STORE",
                "reasoning": "Salience score 0.625 exceeds threshold 0.55"
            }
        }
