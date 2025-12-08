# Memlayer Flexible Salience Configuration - Interactive Demo
# ============================================================
# Google Colab notebook to test custom salience configurations
#
# This notebook demonstrates:
# 1. Creating custom salience configurations with ANY components
# 2. Testing different scoring functions
# 3. Comparing threshold strategies
# 4. Using decision rules
# 5. Visualizing salience scores

# ============================================================
# CELL 1: Setup and Installation
# ============================================================

# Install Memlayer (if not already installed)
!pip install -q git+https://github.com/thebnbrkr/memlayer.git@claude/hosted-memory-service-01CkroWwJc2EpkTcQ72hS2im

# Import required libraries
import os
from memlayer import TenantSalienceConfig, SalienceCalculator
from memlayer.config.salience import (
    SalienceComponent,
    ScoringFunctionType,
    AdaptiveThresholdConfig,
    ThresholdStrategy,
    DecisionRule,
)

print("✅ Installation complete!")
print("=" * 60)


# ============================================================
# CELL 2: Example 1 - Simple Keyword-Based Configuration
# ============================================================

print("\n📊 EXAMPLE 1: Simple Keyword-Based Salience")
print("=" * 60)

# Create a simple config with just keyword matching
simple_config = TenantSalienceConfig(
    tenant_id="demo_user",
    config_name="simple_keyword_config",
    components=[
        SalienceComponent(
            name="tech_relevance",
            weight=0.6,
            description="Relevance to technology topics",
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={
                "keywords": ["python", "AI", "machine learning", "algorithm", "neural"]
            }
        ),
        SalienceComponent(
            name="text_length",
            weight=0.4,
            description="Text length bonus",
            scoring_function=ScoringFunctionType.LENGTH_BONUS,
            scoring_config={
                "min_length": 20,
                "max_length": 200
            }
        ),
    ],
    threshold_config=AdaptiveThresholdConfig(
        strategy=ThresholdStrategy.ABSOLUTE,
        absolute_threshold=0.5
    ),
    decision_rules=[]
)

# Test with different facts
test_facts = [
    "Alice is learning Python and machine learning.",
    "Bob went to the store today.",
    "This article discusses neural networks and AI algorithms in depth.",
    "Hi!",
]

print("\nTesting facts with simple keyword config:")
print("-" * 60)

for fact in test_facts:
    calculator = SalienceCalculator(
        config=simple_config,
        fact=fact,
        tenant_id="demo_user"
    )

    score, decision, log = calculator.compute_salience()

    print(f"\n📝 Fact: {fact[:50]}...")
    print(f"   Score: {score:.3f} | Decision: {decision}")
    print(f"   Component Scores:")
    for comp_name, comp_score in calculator.component_scores.items():
        print(f"      - {comp_name}: {comp_score:.3f}")
    print(f"   Reasoning: {log['reasoning']}")


# ============================================================
# CELL 3: Example 2 - Custom Mode 54 (From Spec)
# ============================================================

print("\n\n📊 EXAMPLE 2: Custom Mode 54")
print("=" * 60)
print("A custom configuration with 3 components and decision rules")

custom_mode_54 = TenantSalienceConfig(
    tenant_id="demo_user",
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
            description="Emotional significance",
            scoring_function=ScoringFunctionType.LLM_SCORED,
            scoring_config={"prompt": "Score emotional impact 0-1"}
        ),
        SalienceComponent(
            name="technical_depth",
            weight=0.3,
            description="Technical complexity",
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={"keywords": ["algorithm", "optimization", "neural", "deep learning"]}
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
        ),
        DecisionRule(
            name="Keep novel + technical",
            condition="novelty > 0.7 and technical_depth > 0.6",
            action="STORE",
            priority=2
        ),
    ]
)

# Test facts
custom_test_facts = [
    "I just got engaged to my partner! We're getting married next year!",
    "The optimization algorithm uses deep learning for neural network training.",
    "I ate lunch today.",
]

print("\nTesting facts with Custom Mode 54:")
print("-" * 60)

for fact in custom_test_facts:
    calculator = SalienceCalculator(
        config=custom_mode_54,
        fact=fact,
        tenant_id="demo_user"
    )

    score, decision, log = calculator.compute_salience()

    print(f"\n📝 Fact: {fact}")
    print(f"   Score: {score:.3f} | Threshold: {calculator.threshold:.3f} | Decision: {decision}")
    print(f"   Component Scores:")
    for comp_name, comp_score in calculator.component_scores.items():
        print(f"      - {comp_name}: {comp_score:.3f}")
    if calculator.matched_rule:
        print(f"   🎯 Matched Rule: {calculator.matched_rule}")
    print(f"   Reasoning: {log['reasoning']}")


# ============================================================
# CELL 4: Example 3 - Comparing Threshold Strategies
# ============================================================

print("\n\n📊 EXAMPLE 3: Comparing Threshold Strategies")
print("=" * 60)

# Create configs with different threshold strategies
strategies = [
    ("Absolute (0.6)", ThresholdStrategy.ABSOLUTE, {"absolute_threshold": 0.6}),
    ("Mean Multiplier (0.8x)", ThresholdStrategy.MEAN_MULTIPLIER, {"multiplier": 0.8, "absolute_threshold": 0.5}),
    ("Percentile (70th)", ThresholdStrategy.PERCENTILE, {"percentile": 70, "absolute_threshold": 0.5}),
]

test_fact = "This is an important announcement about AI safety and machine learning ethics."

print(f"\nTest Fact: {test_fact}")
print("\nResults with different threshold strategies:")
print("-" * 60)

for strategy_name, strategy_type, strategy_params in strategies:
    config = TenantSalienceConfig(
        tenant_id="demo_user",
        config_name=f"threshold_test_{strategy_type.value}",
        components=[
            SalienceComponent(
                name="keyword_score",
                weight=1.0,
                scoring_function=ScoringFunctionType.KEYWORD_MATCH,
                scoring_config={"keywords": ["AI", "machine learning", "important", "ethics"]}
            )
        ],
        threshold_config=AdaptiveThresholdConfig(
            strategy=strategy_type,
            **strategy_params
        ),
        decision_rules=[]
    )

    calculator = SalienceCalculator(config=config, fact=test_fact, tenant_id="demo_user")
    score, decision, log = calculator.compute_salience()

    print(f"\n{strategy_name}:")
    print(f"   Score: {score:.3f} | Threshold: {calculator.threshold:.3f} | Decision: {decision}")


# ============================================================
# CELL 5: Example 4 - Decision Rules Priority Test
# ============================================================

print("\n\n📊 EXAMPLE 4: Decision Rules Priority")
print("=" * 60)
print("Testing how decision rules override threshold-based decisions")

# Config with multiple decision rules
priority_config = TenantSalienceConfig(
    tenant_id="demo_user",
    config_name="priority_test",
    components=[
        SalienceComponent(
            name="importance",
            weight=0.5,
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={"keywords": ["critical", "urgent", "important"]}
        ),
        SalienceComponent(
            name="length",
            weight=0.5,
            scoring_function=ScoringFunctionType.LENGTH_BONUS,
            scoring_config={"min_length": 10, "max_length": 100}
        ),
    ],
    threshold_config=AdaptiveThresholdConfig(
        strategy=ThresholdStrategy.ABSOLUTE,
        absolute_threshold=0.7  # High threshold
    ),
    decision_rules=[
        DecisionRule(
            name="Always store critical messages",
            condition="importance > 0.8",
            action="STORE",
            priority=1  # Highest priority
        ),
        DecisionRule(
            name="Skip very short messages",
            condition="length < 0.2",
            action="SKIP",
            priority=2  # Lower priority
        ),
    ]
)

priority_test_facts = [
    "Critical alert: System failure detected!",  # Should trigger rule 1 (STORE)
    "Hi",  # Should trigger rule 2 (SKIP)
    "This is a normal message about regular things.",  # No rules, use threshold
]

print("\nTesting decision rule priority:")
print("-" * 60)

for fact in priority_test_facts:
    calculator = SalienceCalculator(
        config=priority_config,
        fact=fact,
        tenant_id="demo_user"
    )

    score, decision, log = calculator.compute_salience()

    print(f"\n📝 Fact: {fact}")
    print(f"   Score: {score:.3f} | Threshold: {calculator.threshold:.3f}")
    print(f"   Component Scores: importance={calculator.component_scores['importance']:.3f}, "
          f"length={calculator.component_scores['length']:.3f}")
    if calculator.matched_rule:
        print(f"   🎯 Matched Rule: '{calculator.matched_rule}' → {decision}")
    else:
        print(f"   No rule matched, threshold-based → {decision}")
    print(f"   Reasoning: {log['reasoning']}")


# ============================================================
# CELL 6: Example 5 - Extreme Flexibility (10 Components!)
# ============================================================

print("\n\n📊 EXAMPLE 5: Extreme Flexibility - 10 Custom Components")
print("=" * 60)
print("Demonstrating ZERO restrictions: 10 components with custom names")

# Create 10 custom components (completely arbitrary names)
flexible_config = TenantSalienceConfig(
    tenant_id="demo_user",
    config_name="ultra_flexible_10_components",
    components=[
        SalienceComponent(
            name="metric_alpha",
            weight=0.1,
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={"keywords": ["alpha"]}
        ),
        SalienceComponent(
            name="metric_beta",
            weight=0.1,
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={"keywords": ["beta"]}
        ),
        SalienceComponent(
            name="metric_gamma",
            weight=0.1,
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={"keywords": ["gamma"]}
        ),
        SalienceComponent(
            name="my_weird_score_54",
            weight=0.1,
            scoring_function=ScoringFunctionType.LENGTH_BONUS,
            scoring_config={"min_length": 10, "max_length": 100}
        ),
        SalienceComponent(
            name="UserDefinedMetric_v2",
            weight=0.1,
            scoring_function=ScoringFunctionType.FREQUENCY,
            scoring_config={"target_terms": ["the", "and"]}
        ),
        SalienceComponent(
            name="🎯_target_relevance",
            weight=0.1,
            scoring_function=ScoringFunctionType.LIVENESS,
            scoring_config={"temporal_keywords": ["today", "now", "recent"]}
        ),
        SalienceComponent(
            name="component_7",
            weight=0.1,
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={"keywords": ["test"]}
        ),
        SalienceComponent(
            name="component_8",
            weight=0.1,
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={"keywords": ["demo"]}
        ),
        SalienceComponent(
            name="component_9",
            weight=0.1,
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={"keywords": ["example"]}
        ),
        SalienceComponent(
            name="component_10",
            weight=0.1,
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={"keywords": ["colab"]}
        ),
    ],
    threshold_config=AdaptiveThresholdConfig(
        strategy=ThresholdStrategy.ABSOLUTE,
        absolute_threshold=0.3
    ),
    decision_rules=[]
)

test_fact = "This is a demo example for the Google Colab test with alpha beta gamma keywords today."

calculator = SalienceCalculator(
    config=flexible_config,
    fact=test_fact,
    tenant_id="demo_user"
)

score, decision, log = calculator.compute_salience()

print(f"\nTest Fact: {test_fact}")
print(f"\nFinal Score: {score:.3f} | Decision: {decision}")
print(f"\nAll 10 Component Scores:")
for comp_name, comp_score in calculator.component_scores.items():
    print(f"   {comp_name}: {comp_score:.3f}")

# Verify weights sum to 1.0
total_weight = sum(c.weight for c in flexible_config.components)
print(f"\n✅ Total weight: {total_weight:.2f} (valid: {0.99 <= total_weight <= 1.01})")
print(f"✅ Number of components: {len(flexible_config.components)}")


# ============================================================
# CELL 7: Interactive Test - Create Your Own Config!
# ============================================================

print("\n\n📊 EXAMPLE 6: Create Your Own Configuration")
print("=" * 60)
print("Now it's your turn! Customize this config and re-run the cell.")

# ⬇️ CUSTOMIZE THESE VALUES ⬇️
YOUR_CONFIG_NAME = "my_personal_config"
YOUR_KEYWORDS = ["AI", "technology", "innovation", "future"]
YOUR_KEYWORD_WEIGHT = 0.7
YOUR_LENGTH_WEIGHT = 0.3
YOUR_THRESHOLD = 0.6

# Create your custom config
your_config = TenantSalienceConfig(
    tenant_id="your_tenant",
    config_name=YOUR_CONFIG_NAME,
    components=[
        SalienceComponent(
            name="my_keyword_score",
            weight=YOUR_KEYWORD_WEIGHT,
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={"keywords": YOUR_KEYWORDS}
        ),
        SalienceComponent(
            name="my_length_score",
            weight=YOUR_LENGTH_WEIGHT,
            scoring_function=ScoringFunctionType.LENGTH_BONUS,
            scoring_config={"min_length": 20, "max_length": 150}
        ),
    ],
    threshold_config=AdaptiveThresholdConfig(
        strategy=ThresholdStrategy.ABSOLUTE,
        absolute_threshold=YOUR_THRESHOLD
    ),
    decision_rules=[]
)

# Test your config
your_test_facts = [
    "AI and technology are driving innovation for the future.",
    "I had coffee this morning.",
    "The future of innovation in AI technology looks promising with new developments.",
]

print(f"\nYour Config: '{YOUR_CONFIG_NAME}'")
print(f"Keywords: {YOUR_KEYWORDS}")
print(f"Weights: keyword={YOUR_KEYWORD_WEIGHT}, length={YOUR_LENGTH_WEIGHT}")
print(f"Threshold: {YOUR_THRESHOLD}")
print("\nTesting your config:")
print("-" * 60)

for fact in your_test_facts:
    calculator = SalienceCalculator(
        config=your_config,
        fact=fact,
        tenant_id="your_tenant"
    )

    score, decision, log = calculator.compute_salience()

    print(f"\n📝 {fact}")
    print(f"   Score: {score:.3f} | Decision: {decision}")
    print(f"   keyword={calculator.component_scores['my_keyword_score']:.3f}, "
          f"length={calculator.component_scores['my_length_score']:.3f}")


# ============================================================
# CELL 8: Visualization - Compare Multiple Configs
# ============================================================

print("\n\n📊 EXAMPLE 7: Comparing Multiple Configurations")
print("=" * 60)

# Create 3 different configs
configs = [
    ("Strict (threshold 0.8)", TenantSalienceConfig(
        tenant_id="demo",
        config_name="strict",
        components=[
            SalienceComponent(name="score", weight=1.0,
                            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
                            scoring_config={"keywords": ["important", "critical", "urgent"]})
        ],
        threshold_config=AdaptiveThresholdConfig(strategy=ThresholdStrategy.ABSOLUTE, absolute_threshold=0.8),
    )),
    ("Balanced (threshold 0.5)", TenantSalienceConfig(
        tenant_id="demo",
        config_name="balanced",
        components=[
            SalienceComponent(name="score", weight=1.0,
                            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
                            scoring_config={"keywords": ["important", "critical", "urgent"]})
        ],
        threshold_config=AdaptiveThresholdConfig(strategy=ThresholdStrategy.ABSOLUTE, absolute_threshold=0.5),
    )),
    ("Permissive (threshold 0.2)", TenantSalienceConfig(
        tenant_id="demo",
        config_name="permissive",
        components=[
            SalienceComponent(name="score", weight=1.0,
                            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
                            scoring_config={"keywords": ["important", "critical", "urgent"]})
        ],
        threshold_config=AdaptiveThresholdConfig(strategy=ThresholdStrategy.ABSOLUTE, absolute_threshold=0.2),
    )),
]

test_facts_comparison = [
    "This is a critical and urgent message!",
    "This is an important update.",
    "Just a regular message.",
]

print("\nComparison Results:")
print("-" * 60)

for fact in test_facts_comparison:
    print(f"\n📝 Fact: {fact}")
    print("   Config Results:")

    for config_name, config in configs:
        calc = SalienceCalculator(config=config, fact=fact, tenant_id="demo")
        score, decision, _ = calc.compute_salience()
        print(f"      {config_name}: score={score:.3f} → {decision}")


# ============================================================
# CELL 9: Summary and Key Takeaways
# ============================================================

print("\n\n" + "=" * 60)
print("🎉 SUMMARY - What You've Learned")
print("=" * 60)

print("""
✅ Component Flexibility
   - Create ANY number of components with ANY names
   - Use ANY weights (must sum to 1.0)
   - Choose from 9 different scoring functions

✅ Threshold Strategies
   - Absolute: Fixed threshold value
   - Percentile: Adapt based on recent scores
   - Mean Multiplier: Percentage of mean
   - And 3 more strategies!

✅ Decision Rules
   - Boolean expressions to override threshold
   - Priority-based evaluation (first match wins)
   - Access to all component scores in conditions

✅ Full Auditability
   - Every decision logged with reasoning
   - Component scores tracked
   - Complete transparency

✅ ZERO Restrictions
   - No presets, no templates
   - Complete freedom to design your own logic
   - Production-ready with type safety

🔗 Next Steps:
   1. Modify the configs above and re-run cells
   2. Create your own custom scoring logic
   3. Test with your own facts
   4. Check docs/SALIENCE_CONFIG.md for full API docs

Happy customizing! 🚀
""")

print("=" * 60)
