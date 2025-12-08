# Memlayer Integrated Salience System - End-to-End Demo
# =======================================================
# This notebook demonstrates the INTEGRATED salience system working
# with the actual memory flow. You can see:
# - Custom salience configs deciding what to store
# - Real chat interactions
# - Salience logs showing WHY decisions were made
# - What got stored vs. what got skipped

# ============================================================
# CELL 1: Setup and Installation
# ============================================================

!pip install -q git+https://github.com/thebnbrkr/memlayer.git@claude/hosted-memory-service-01CkroWwJc2EpkTcQ72hS2im

import os
from memlayer import OpenAI, TenantSalienceConfig, GraphVisualizer, MemoryBrowser
from memlayer.config.salience import (
    SalienceComponent,
    ScoringFunctionType,
    AdaptiveThresholdConfig,
    ThresholdStrategy,
    DecisionRule,
)

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with your actual key

print("✅ Installation complete!")
print("="*60)


# ============================================================
# CELL 2: Create Custom Salience Config
# ============================================================

print("\n📊 Creating Custom Salience Configuration")
print("="*60)

# Define a custom salience config that:
# - Values technical content (60%)
# - Values length (40%)
# - Has a rule: Always store if mentions "important"

my_salience_config = TenantSalienceConfig(
    tenant_id="alice",
    config_name="tech_focused_config",
    components=[
        SalienceComponent(
            name="technical_relevance",
            weight=0.6,
            description="How technical/informative is this?",
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={
                "keywords": ["python", "code", "algorithm", "data", "AI", "technical", "project"]
            }
        ),
        SalienceComponent(
            name="sufficient_length",
            weight=0.4,
            description="Long enough to be meaningful",
            scoring_function=ScoringFunctionType.LENGTH_BONUS,
            scoring_config={
                "min_length": 15,
                "max_length": 150
            }
        ),
    ],
    threshold_config=AdaptiveThresholdConfig(
        strategy=ThresholdStrategy.ABSOLUTE,
        absolute_threshold=0.5  # Need score >= 0.5 to store
    ),
    decision_rules=[
        DecisionRule(
            name="Always keep important messages",
            condition="technical_relevance > 0.8",
            action="STORE",
            priority=1
        ),
    ]
)

print(f"✅ Created config: {my_salience_config.config_name}")
print(f"   Components: {len(my_salience_config.components)}")
print(f"   Threshold: {my_salience_config.threshold_config.absolute_threshold}")
print(f"   Decision rules: {len(my_salience_config.decision_rules)}")


# ============================================================
# CELL 3: Initialize Client with Custom Salience
# ============================================================

print("\n🚀 Initializing OpenAI Client with Custom Salience Config")
print("="*60)

client = OpenAI(
    model="gpt-4o-mini",
    user_id="Alice",
    tenant_id="alice",
    storage_path="./integrated_memory",
    operation_mode="online",
    salience_config=my_salience_config,  # ← NEW: Use custom config!
    salience_threshold=0.0  # Ignored when salience_config is provided
)

print("✅ Client initialized with custom salience configuration!")
print("   Now chat messages will be scored by YOUR custom logic.")


# ============================================================
# CELL 4: Test Chat - See Salience in Action
# ============================================================

print("\n\n💬 Testing Chat with Different Messages")
print("="*60)
print("Let's send 4 messages and see which ones get stored:\n")

test_messages = [
    "My name is Alice and I work at TechCorp.",  # Should STORE (long + technical)
    "Hi!",  # Should SKIP (too short)
    "I'm working on a Python data analysis project using algorithms.",  # Should STORE (very technical)
    "I had coffee today.",  # Should SKIP (not technical enough)
]

for i, msg in enumerate(test_messages, 1):
    print(f"\n[{i}/4] Sending: \"{msg}\"")
    print("-"*60)

    response = client.chat([
        {"role": "user", "content": msg}
    ])

    print(f"✓ Response: {response['choices'][0]['message']['content'][:100]}...")

print("\n✅ All messages sent!")


# ============================================================
# CELL 5: Check Salience Logs
# ============================================================

print("\n\n📊 Salience Decision Logs")
print("="*60)
print("Let's see what the custom salience config decided:\n")

logs = client.get_salience_logs()

for i, log in enumerate(logs, 1):
    print(f"\n[Log {i}]")
    print(f"Fact: \"{log['fact'][:60]}...\"")
    print(f"Decision: {log['decision']}")
    print(f"Salience Score: {log['score']:.3f} (threshold: {log['threshold']:.3f})")
    print(f"Component Scores:")
    for comp_name, comp_score in log['component_scores'].items():
        print(f"   - {comp_name}: {comp_score:.3f}")
    if log['matched_rule']:
        print(f"🎯 Matched Rule: '{log['matched_rule']}'")
    print(f"💡 Reasoning: {log['reasoning']}")
    print("-"*60)


# ============================================================
# CELL 6: View What Got Stored
# ============================================================

print("\n\n📦 What Got Stored in Memory?")
print("="*60)

browser = MemoryBrowser(client.storage)
browser.show_all(user_id="Alice", max_items=20)


# ============================================================
# CELL 7: View Knowledge Graph
# ============================================================

print("\n\n🕸️ Knowledge Graph (Entities & Relationships)")
print("="*60)

viz = GraphVisualizer(client.graph_storage)
viz.print_summary()

# Show visual graph
print("\nGenerating graph visualization...")
viz.show()


# ============================================================
# CELL 8: Test Different Salience Config
# ============================================================

print("\n\n🔄 Testing Different Salience Config")
print("="*60)
print("Let's create a PERMISSIVE config that stores everything:")

# Create a very permissive config
permissive_config = TenantSalienceConfig(
    tenant_id="bob",
    config_name="store_everything",
    components=[
        SalienceComponent(
            name="always_high",
            weight=1.0,
            scoring_function=ScoringFunctionType.LENGTH_BONUS,
            scoring_config={"min_length": 1, "max_length": 10}  # Even "Hi!" scores well
        )
    ],
    threshold_config=AdaptiveThresholdConfig(
        strategy=ThresholdStrategy.ABSOLUTE,
        absolute_threshold=0.1  # Very low threshold
    )
)

# Create new client with permissive config
client2 = OpenAI(
    model="gpt-4o-mini",
    user_id="Bob",
    tenant_id="bob",
    storage_path="./permissive_memory",
    operation_mode="online",
    salience_config=permissive_config
)

print(f"✅ Created permissive config with threshold {permissive_config.threshold_config.absolute_threshold}")

# Test with same messages
print("\nSending same messages with permissive config:")
for msg in ["Hi!", "I had coffee today."]:
    print(f"\nSending: \"{msg}\"")
    client2.chat([{"role": "user", "content": msg}])

# Check logs
print("\nSalience logs for permissive config:")
logs2 = client2.get_salience_logs()
for log in logs2:
    print(f"   {log['fact'][:40]}... → {log['decision']} (score: {log['score']:.3f})")


# ============================================================
# CELL 9: Compare Results
# ============================================================

print("\n\n📊 Comparison: Strict vs. Permissive")
print("="*60)

print("\nSTRICT CONFIG (technical + length):")
print(f"   Total messages: {len(logs)}")
stored = [l for l in logs if l['decision'] == 'STORE']
skipped = [l for l in logs if l['decision'] == 'SKIP']
print(f"   Stored: {len(stored)}")
print(f"   Skipped: {len(skipped)}")

print("\nPERMISSIVE CONFIG (low threshold):")
print(f"   Total messages: {len(logs2)}")
stored2 = [l for l in logs2 if l['decision'] == 'STORE']
skipped2 = [l for l in logs2 if l['decision'] == 'SKIP']
print(f"   Stored: {len(stored2)}")
print(f"   Skipped: {len(skipped2)}")

print("\n💡 This shows how custom salience configs control what gets stored!")


# ============================================================
# CELL 10: Summary
# ============================================================

print("\n\n" + "="*60)
print("🎉 END-TO-END INTEGRATION COMPLETE")
print("="*60)

print("""
✅ What We Demonstrated:

1. Created custom salience configuration
   - User-defined components (technical_relevance, sufficient_length)
   - Custom weights (60/40 split)
   - Absolute threshold (0.5)
   - Decision rule (always store if technical_relevance > 0.8)

2. Integrated with real memory flow
   - OpenAI client uses the custom config
   - Chat messages scored by YOUR logic
   - Salience calculator decides STORE vs SKIP

3. Full transparency
   - Get salience logs showing all decisions
   - See component scores for each message
   - Understand WHY each decision was made

4. Compared two configs
   - Strict config: Only stored technical content
   - Permissive config: Stored everything

🎯 KEY TAKEAWAY:
   You now have FULL CONTROL over what gets stored in memory!
   No hardcoded rules, no presets - YOUR custom logic decides.

🚀 Next Steps:
   - Try your own custom components
   - Experiment with different thresholds
   - Add your own decision rules
   - Deploy to production with FastAPI!
""")

print("="*60)
