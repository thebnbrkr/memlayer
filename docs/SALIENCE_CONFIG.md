# Flexible Salience Configuration System

A fully customizable salience scoring system with **ZERO presets** and **ZERO templates**. Complete freedom to define your own components, weights, thresholds, and decision rules.

## Overview

The salience configuration system allows you to create custom scoring logic for determining which facts are worth storing in memory. You have complete control over:

- **Components**: Define ANY components with ANY names
- **Weights**: Set ANY weights (must sum to 1.0)
- **Scoring Functions**: Choose from 9 different scoring functions or write custom Python
- **Thresholds**: 6 different threshold calculation strategies
- **Decision Rules**: Boolean expressions for override logic

## Quick Start

### 1. Create a Custom Configuration

```python
from memlayer import TenantSalienceConfig
from memlayer.config.salience import (
    SalienceComponent,
    ScoringFunctionType,
    AdaptiveThresholdConfig,
    ThresholdStrategy,
    DecisionRule,
)

# Define your custom components
config = TenantSalienceConfig(
    tenant_id="customer_123",
    config_name="my_custom_score_54",
    components=[
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
            scoring_function=ScoringFunctionType.LLM_SCORED,
            scoring_config={"prompt": "Score emotional impact 0-1"}
        ),
        SalienceComponent(
            name="technical_depth",
            weight=0.3,
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={"keywords": ["algorithm", "optimization", "neural"]}
        ),
    ],
    threshold_config=AdaptiveThresholdConfig(
        strategy=ThresholdStrategy.PERCENTILE,
        recent_facts_window=150,
        percentile=70
    ),
    decision_rules=[
        DecisionRule(
            name="Always keep high emotional",
            condition="emotional_impact > 0.85",
            action="STORE",
            priority=1
        )
    ]
)
```

### 2. Compute Salience for a Fact

```python
from memlayer import SalienceCalculator

calculator = SalienceCalculator(
    config=config,
    fact="Alice works at TechCorp as a software engineer",
    tenant_id="customer_123"
)

score, decision, log = calculator.compute_salience()

print(f"Salience score: {score:.3f}")
print(f"Decision: {decision}")  # STORE or SKIP
print(f"Component scores: {log['component_scores']}")
print(f"Reasoning: {log['reasoning']}")
```

### 3. Use the API (Optional)

If you're running the FastAPI server:

```python
from fastapi import FastAPI
from memlayer import salience_router

app = FastAPI()
app.include_router(salience_router)

# Run with: uvicorn main:app --reload
```

API endpoints:
- `POST /api/config/salience` - Create config
- `GET /api/config/salience?tenant_id=X` - List configs
- `GET /api/config/salience/{config_name}` - Get config
- `PUT /api/config/salience/{config_name}` - Update config
- `DELETE /api/config/salience/{config_name}` - Delete config
- `POST /api/config/salience/{config_name}/test` - Test config
- `GET /api/config/salience/{config_name}/audit-log` - Get audit trail

## Scoring Functions

### Available Functions

1. **embedding_similarity** - Cosine similarity to existing memories
2. **keyword_match** - Keyword matching (returns matched/total ratio)
3. **length_bonus** - Score based on text length
4. **frequency** - Term frequency scoring
5. **liveness** - Temporal recency indicators
6. **llm_scored** - Score using an LLM prompt
7. **custom_python** - Execute custom Python code
8. **weighted_sum** - Weighted sum of other components
9. **boolean_gate** - Returns 1.0 or 0.0 based on condition

### Example: Keyword Match

```python
SalienceComponent(
    name="keyword_relevance",
    weight=0.5,
    scoring_function=ScoringFunctionType.KEYWORD_MATCH,
    scoring_config={
        "keywords": ["python", "machine learning", "AI"],
        "case_sensitive": False
    }
)
```

### Example: Length Bonus

```python
SalienceComponent(
    name="text_length",
    weight=0.3,
    scoring_function=ScoringFunctionType.LENGTH_BONUS,
    scoring_config={
        "min_length": 20,  # Score 0.0 below this
        "max_length": 200  # Score 1.0 above this
    }
)
```

### Example: Custom Python

```python
SalienceComponent(
    name="custom_scorer",
    weight=0.2,
    scoring_function=ScoringFunctionType.CUSTOM_PYTHON,
    scoring_config={
        "code": """
# Code has access to: fact, tenant_id
word_count = len(fact.split())
score = min(word_count / 50, 1.0)  # Max score at 50 words
"""
    }
)
```

## Threshold Strategies

### 1. Absolute

Fixed threshold value (simplest):

```python
AdaptiveThresholdConfig(
    strategy=ThresholdStrategy.ABSOLUTE,
    absolute_threshold=0.6  # Fixed at 0.6
)
```

### 2. Mean Multiplier

Threshold = mean of recent scores × multiplier:

```python
AdaptiveThresholdConfig(
    strategy=ThresholdStrategy.MEAN_MULTIPLIER,
    recent_facts_window=100,
    multiplier=0.8  # 80% of recent mean
)
```

### 3. Percentile

Threshold = Nth percentile of recent scores:

```python
AdaptiveThresholdConfig(
    strategy=ThresholdStrategy.PERCENTILE,
    recent_facts_window=150,
    percentile=70  # 70th percentile
)
```

### 4. Exponential Decay

Threshold decreases over time:

```python
AdaptiveThresholdConfig(
    strategy=ThresholdStrategy.EXPONENTIAL_DECAY,
    absolute_threshold=0.7,
    decay_rate=0.1
)
```

## Decision Rules

Decision rules are boolean expressions that can override the threshold-based decision. They're evaluated in priority order (lower number = higher priority).

### Example Rules

```python
decision_rules=[
    # High priority: Always store important content
    DecisionRule(
        name="Always keep high novelty",
        condition="novelty > 0.9",
        action="STORE",
        priority=1
    ),

    # Medium priority: Skip low-quality content
    DecisionRule(
        name="Skip low quality",
        condition="novelty < 0.2 and emotional_impact < 0.2",
        action="SKIP",
        priority=2
    ),

    # Complex condition
    DecisionRule(
        name="Keep technical papers",
        condition="technical_depth > 0.7 and keyword_relevance > 0.5",
        action="STORE",
        priority=3
    ),
]
```

### Available Variables in Conditions

- `salience_score` - Final weighted salience score
- `threshold` - Computed threshold value
- All component names (e.g., `novelty`, `emotional_impact`, etc.)

### Operators

You can use any Python boolean operators:
- Comparison: `>`, `<`, `>=`, `<=`, `==`, `!=`
- Logical: `and`, `or`, `not`
- Parentheses: `(condition1) and (condition2)`

## Complete Examples

### Example 1: Research Paper Scoring

```python
config = TenantSalienceConfig(
    tenant_id="research_team",
    config_name="research_neural_nets",
    components=[
        SalienceComponent(
            name="novelty",
            weight=0.2,
            scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY
        ),
        SalienceComponent(
            name="relevance_to_task",
            weight=0.3,
            scoring_function=ScoringFunctionType.LLM_SCORED,
            scoring_config={
                "prompt": "Score relevance to neural networks research 0-1"
            }
        ),
        SalienceComponent(
            name="technical_depth",
            weight=0.2,
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={
                "keywords": [
                    "neural network", "deep learning", "backpropagation",
                    "gradient descent", "optimization", "architecture"
                ]
            }
        ),
        SalienceComponent(
            name="citation_importance",
            weight=0.15,
            scoring_function=ScoringFunctionType.FREQUENCY,
            scoring_config={
                "target_terms": ["citation", "reference", "paper"]
            }
        ),
        SalienceComponent(
            name="recency_6_months",
            weight=0.15,
            scoring_function=ScoringFunctionType.LIVENESS,
            scoring_config={
                "temporal_keywords": [
                    "2024", "recent", "latest", "new", "current"
                ]
            }
        ),
    ],
    threshold_config=AdaptiveThresholdConfig(
        strategy=ThresholdStrategy.MEAN_MULTIPLIER,
        recent_facts_window=100,
        multiplier=0.8
    ),
    decision_rules=[
        DecisionRule(
            name="Always keep highly relevant papers",
            condition="relevance_to_task > 0.85",
            action="STORE",
            priority=1
        ),
        DecisionRule(
            name="Keep novel + technical",
            condition="novelty > 0.7 and technical_depth > 0.6",
            action="STORE",
            priority=2
        ),
    ]
)
```

### Example 2: Personal Memory Assistant

```python
config = TenantSalienceConfig(
    tenant_id="user_alice",
    config_name="personal_assistant",
    components=[
        SalienceComponent(
            name="emotional_significance",
            weight=0.4,
            scoring_function=ScoringFunctionType.LLM_SCORED,
            scoring_config={
                "prompt": "Score emotional significance for a personal memory 0-1"
            }
        ),
        SalienceComponent(
            name="future_usefulness",
            weight=0.3,
            scoring_function=ScoringFunctionType.LLM_SCORED,
            scoring_config={
                "prompt": "How useful will this be to remember later? 0-1"
            }
        ),
        SalienceComponent(
            name="contains_details",
            weight=0.3,
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={
                "keywords": [
                    "name", "date", "time", "place", "number", "email",
                    "address", "phone", "appointment", "deadline"
                ]
            }
        ),
    ],
    threshold_config=AdaptiveThresholdConfig(
        strategy=ThresholdStrategy.PERCENTILE,
        recent_facts_window=200,
        percentile=60
    ),
    decision_rules=[
        DecisionRule(
            name="Always keep important dates/events",
            condition="contains_details > 0.5 and emotional_significance > 0.7",
            action="STORE",
            priority=1
        ),
    ]
)
```

### Example 3: 100 Component Freedom Test

You can have as many components as you want:

```python
# Create 100 custom components
components = [
    SalienceComponent(
        name=f"metric_{i}",
        weight=0.01,  # 100 components × 0.01 = 1.0
        scoring_function=ScoringFunctionType.KEYWORD_MATCH,
        scoring_config={"keywords": [f"keyword_{i}"]}
    )
    for i in range(100)
]

config = TenantSalienceConfig(
    tenant_id="power_user",
    config_name="ultra_custom_100_metrics",
    components=components,
    threshold_config=AdaptiveThresholdConfig(
        strategy=ThresholdStrategy.ABSOLUTE,
        absolute_threshold=0.5
    )
)
```

## Database Schema

The system uses PostgreSQL with JSONB for maximum flexibility:

```sql
-- Stores configs
CREATE TABLE tenant_salience_config (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    config_name VARCHAR(255) NOT NULL,
    components JSONB NOT NULL,
    threshold_config JSONB NOT NULL,
    decision_rules JSONB NOT NULL DEFAULT '[]',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    UNIQUE(tenant_id, config_name)
);

-- Stores audit trail
CREATE TABLE salience_computation_log (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL,
    fact_id UUID,
    fact_text TEXT NOT NULL,
    config_id UUID NOT NULL,
    config_snapshot JSONB NOT NULL,
    component_scores JSONB NOT NULL,
    final_salience_score FLOAT NOT NULL,
    threshold_value FLOAT NOT NULL,
    decision_rule_matched VARCHAR(255),
    decision VARCHAR(10) NOT NULL CHECK (decision IN ('STORE', 'SKIP')),
    reasoning TEXT NOT NULL,
    computation_time_ms FLOAT,
    created_at TIMESTAMP
);
```

## Testing

Run the comprehensive test suite:

```bash
pytest tests/test_salience.py -v
```

Test coverage includes:
- ✅ Component weight validation
- ✅ Config creation with custom names
- ✅ All scoring functions
- ✅ All threshold strategies
- ✅ Decision rule evaluation
- ✅ Priority-based rule matching
- ✅ Complete end-to-end workflows

## API Usage Examples

### Create Config via API

```bash
curl -X POST "http://localhost:8000/api/config/salience" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "customer_123",
    "config_name": "my_config",
    "components": [
      {
        "name": "novelty",
        "weight": 0.5,
        "scoring_function": "keyword_match",
        "scoring_config": {"keywords": ["new", "innovation"]}
      },
      {
        "name": "importance",
        "weight": 0.5,
        "scoring_function": "llm_scored",
        "scoring_config": {"prompt": "Score importance 0-1"}
      }
    ],
    "threshold_config": {
      "strategy": "absolute",
      "absolute_threshold": 0.6
    },
    "decision_rules": []
  }'
```

### Test Config via API

```bash
curl -X POST "http://localhost:8000/api/config/salience/my_config/test?tenant_id=customer_123" \
  -H "Content-Type: application/json" \
  -d '{
    "fact": "This is a new innovation in AI technology"
  }'
```

Response:
```json
{
  "salience_score": 0.675,
  "threshold": 0.6,
  "decision": "STORE",
  "component_scores": {
    "novelty": 1.0,
    "importance": 0.35
  },
  "full_log": {
    "computation_steps": [...],
    "reasoning": "Salience score 0.675 >= threshold 0.6"
  },
  "computation_time_ms": 45.2
}
```

## Key Principles

1. **ZERO Restrictions**: Define ANY components with ANY names
2. **Complete Freedom**: Use ANY scoring functions, ANY weights, ANY thresholds
3. **Full Transparency**: Complete audit trail of all decisions
4. **Production Ready**: Error handling, logging, and comprehensive tests
5. **Type Safety**: Full Pydantic validation
6. **Flexible Storage**: JSONB for maximum schema flexibility

## Migration

To set up the database:

```bash
psql -U postgres -d your_database -f migrations/001_create_salience_tables.sql
```

## License

MIT License - Same as Memlayer
