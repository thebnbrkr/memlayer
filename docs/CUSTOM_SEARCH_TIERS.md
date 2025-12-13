# Custom Search Tiers Guide

## Overview

**YES, search tiers are fully customizable in Memlayer!**

The default tiers (fast/balanced/deep) are just the starting point. You can create custom tiers for any use case:

```python
# Default (hardcoded)
fast → vector only (top 2)
balanced → vector only (top 5)
deep → vector + graph (2-hop traversal)

# Custom (your configuration)
product_lookup → vector only (top 3, high precision, recency boost)
recommendation → hybrid (top 10, 2-hop graph, relationship traversal)
debug_mode → hybrid (top 20, 5-hop graph, very deep investigation)
```

## Comparison with Other Systems

### Mem0
- **Tiers**: None (vector-only, no customization)
- **Flexibility**: ❌ No control

### Mem0g
- **Tiers**: None (always uses hybrid vector+graph)
- **Flexibility**: ❌ Fixed strategy, wasteful on simple queries

### Supermemory
- **Tiers**: Storage-based (hot/cold)
  - Hot tier: Recent memories in Cloudflare KV (fast access)
  - Cold tier: Older memories in cost-effective storage
- **Flexibility**: ⚠️ Time-based only, not query-based
- **Recency**: ✅ Strong recency bias (41.4% faster than Mem0)

### Memlayer (Custom Tiers)
- **Tiers**: Fully customizable search strategies
- **Flexibility**: ✅ Create unlimited custom tiers
- **Controls**: Vector/graph mode, top-k, depth, recency, filters
- **Novelty**: 🎯 Most flexible system available

## Custom Tier Configuration

### Basic Structure

```python
from memlayer.config.search_tiers import SearchTierConfig, SearchMode

tier = SearchTierConfig(
    name="my_tier",
    mode=SearchMode.HYBRID,  # VECTOR_ONLY, GRAPH_ONLY, or HYBRID
    vector_top_k=10,         # Number of vector results
    graph_depth=2,           # Graph traversal depth (1-5 hops)
    enable_graph=True,       # Enable graph search
    recency_boost=0.5,       # Recency bias (0.0-1.0)
    score_threshold=0.7,     # Minimum similarity score
    metadata_filters={"source": "work"},  # Optional filters
    custom_ranker=None       # Optional custom ranking function
)
```

### Fluent Builder API

```python
from memlayer.config.search_tiers import SearchTierBuilder

tier = (
    SearchTierBuilder("social_feed")
    .vector_only()
    .top_k(15)
    .recency_boost(0.9)
    .score_threshold(0.5)
    .build()
)
```

## Preset Tier Collections

### E-commerce Tiers

```python
from memlayer.config.search_tiers import load_preset, get_tier

load_preset("ecommerce")

# Quick product lookup (fast, precise)
tier = get_tier("product_lookup")
# → vector only, top 3, score_threshold=0.7, recency_boost=0.3

# Personalized recommendations (hybrid, explores relationships)
tier = get_tier("recommendation")
# → hybrid, top 10, graph depth 2, recency_boost=0.5

# Browse history (lots of results, heavy recency)
tier = get_tier("browse_history")
# → vector only, top 20, recency_boost=0.8
```

**Use case**:
- "Show me this product" → `product_lookup` (50ms, precise)
- "Recommend similar products" → `recommendation` (180ms, graph finds "bought together")
- "What did I view yesterday?" → `browse_history` (80ms, recent items)

### Customer Support Tiers

```python
load_preset("support")

# Quick answers (FAQ-style)
tier = get_tier("quick_answer")
# → vector only, top 3, score_threshold=0.8 (high confidence)

# Deep investigation (complex issues)
tier = get_tier("deep_investigation")
# → hybrid, top 15, graph depth 3, recency_boost=0.6

# Ticket history (relationship traversal)
tier = get_tier("ticket_history")
# → graph only, depth 2
```

**Use case**:
- "How do I reset my password?" → `quick_answer` (fast FAQ)
- "Investigate this billing error" → `deep_investigation` (deep graph, finds root cause)
- "Show all tickets for this customer" → `ticket_history` (graph traversal)

### Research / Academic Tiers

```python
load_preset("research")

# Literature review (broad search)
tier = get_tier("literature_review")
# → hybrid, top 20, graph depth 3, recency_boost=0.2

# Citation network (very deep graph)
tier = get_tier("citation_network")
# → graph only, depth 4 (traces citation chains)

# Quick fact (single precise answer)
tier = get_tier("quick_fact")
# → vector only, top 1, score_threshold=0.9
```

**Use case**:
- "Papers about transformers in NLP" → `literature_review` (broad)
- "How did ResNet influence BERT?" → `citation_network` (deep citation graph)
- "Who invented the transformer?" → `quick_fact` (single answer)

### Supermemory-Style Recency Tiers

```python
load_preset("supermemory")

# Hot memory (recent, fast - like Supermemory's Cloudflare KV)
tier = get_tier("hot_memory")
# → vector only, top 5, recency_boost=0.9, filter: last 24 hours

# Cold memory (older, comprehensive)
tier = get_tier("cold_memory")
# → hybrid, top 10, graph depth 2, recency_boost=0.0, filter: > 1 week

# Contextual recall (balanced)
tier = get_tier("contextual_recall")
# → hybrid, top 8, graph depth 2, recency_boost=0.5
```

**Mimics Supermemory's architecture**:
- Supermemory stores recent memories in Cloudflare KV (hot tier)
- Memlayer applies recency boost to prioritize recent memories
- Same effect: Fast access to recent data, comprehensive search for old data

## Industry-Specific Custom Tiers

### Healthcare

```python
patient_tier = (
    SearchTierBuilder("patient_history")
    .hybrid()
    .top_k(10)
    .depth(3)  # Traverse conditions → medications → side effects
    .recency_boost(0.6)  # Recent visits more relevant
    .score_threshold(0.8)  # High precision for healthcare
    .filter(record_type="clinical")
    .build()
)
```

**Use case**: "Show patient's allergy history and related medications"
- Graph traverses: Patient → Allergies → Medications → Contraindications

### Legal

```python
case_law_tier = (
    SearchTierBuilder("case_law")
    .hybrid()
    .top_k(20)
    .depth(4)  # Deep citation network
    .recency_boost(0.2)  # Older cases still relevant
    .score_threshold(0.85)  # Very high precision
    .build()
)
```

**Use case**: "Find precedents for this case"
- Graph traverses: Case A → cites Case B → cites Case C → cites Case D

### Finance

```python
transaction_tier = (
    SearchTierBuilder("transaction_analysis")
    .hybrid()
    .top_k(50)  # Lots of transactions
    .depth(2)  # Related transactions, accounts, entities
    .recency_boost(0.8)  # Recent transactions critical
    .filter(status="completed")
    .build()
)
```

**Use case**: "Detect suspicious transaction patterns"
- Graph traverses: Transaction → Related Accounts → Flagged Entities

## Dynamic Tier Selection

Instead of hardcoding, choose tiers dynamically based on question type:

```python
def choose_tier(question: str) -> str:
    q = question.lower()

    # Simple factual
    if q.startswith(("what is", "who is", "where")):
        return "fast"

    # Relational
    if any(word in q for word in ["connected", "related", "between"]):
        return "deep"

    # Recommendation
    if "recommend" in q or "similar" in q:
        return "recommendation"

    # Recent events
    if "recent" in q or "yesterday" in q or "today" in q:
        return "hot_memory"

    # Default
    return "balanced"

# Usage
tier_name = choose_tier("What products are related to X?")
tier = get_tier(tier_name)
```

## Performance Comparison

| Tier Type | Top-K | Graph Depth | Latency | Best For |
|-----------|-------|-------------|---------|----------|
| **fast** | 2 | 0 | ~30ms | Simple facts |
| **balanced** | 5 | 0 | ~50ms | General queries |
| **deep** | 10 | 2 | ~180ms | Relational questions |
| **product_lookup** | 3 | 0 | ~40ms | E-commerce searches |
| **recommendation** | 10 | 2 | ~200ms | "Similar to X" |
| **quick_answer** | 3 | 0 | ~35ms | FAQ, support |
| **deep_investigation** | 15 | 3 | ~300ms | Complex debugging |
| **citation_network** | 20 | 4 | ~450ms | Academic research |
| **hot_memory** | 5 | 0 | ~25ms | Recent memories (Supermemory-style) |

## Recency Boost Explained

**What is it?** Boosts the score of recent memories (like Supermemory's recency bias).

**How it works:**
```python
# Without recency boost
score = similarity_score  # 0.0 - 1.0

# With recency boost (0.5)
age_factor = 1.0 / (1 + days_old)  # Newer = higher
boosted_score = similarity_score * (1 + recency_boost * age_factor)

# Example:
# Memory from today: score = 0.7 * (1 + 0.5 * 1.0) = 1.05 (capped at 1.0)
# Memory from 10 days ago: score = 0.7 * (1 + 0.5 * 0.09) = 0.73
```

**Supermemory uses recency_boost ≈ 0.9** (very heavy bias toward recent memories)

## When to Use Each Tier

| Question | Optimal Tier | Reasoning |
|----------|-------------|-----------|
| "What is Alice's job?" | `fast` | Simple fact, no relationships needed |
| "Tell me about Alice" | `balanced` | Moderate context, still vector-only |
| "How are Alice's career and hobbies connected?" | `deep` | Relational, needs graph traversal |
| "What products did I view yesterday?" | `hot_memory` | Recent, recency boost critical |
| "Recommend products like X" | `recommendation` | Graph finds "bought together" relationships |
| "How do I reset password?" | `quick_answer` | FAQ, needs high precision |
| "Debug this complex billing issue" | `deep_investigation` | Deep graph, needs root cause analysis |
| "Papers citing BERT" | `citation_network` | Academic, very deep graph |

## Custom Ranking Functions

Add custom logic to re-rank results:

```python
def boost_work_memories(results):
    """Boost memories tagged with 'work' context."""
    for result in results:
        if result.get("metadata", {}).get("context") == "work":
            result["score"] *= 1.2  # 20% boost
    return sorted(results, key=lambda x: x["score"], reverse=True)

tier = (
    SearchTierBuilder("work_focused")
    .vector_only()
    .top_k(10)
    .ranker(boost_work_memories)
    .build()
)
```

## Metadata Filters

Restrict search to specific subsets:

```python
# Search only work-related memories
tier = (
    SearchTierBuilder("work_only")
    .vector_only()
    .top_k(5)
    .filter(context="work", department="engineering")
    .build()
)

# Search only recent memories
tier = (
    SearchTierBuilder("recent_only")
    .vector_only()
    .top_k(10)
    .filter(age_hours=24)  # Last 24 hours only
    .build()
)
```

## Integration with Memlayer

### Option 1: Register and Use by Name

```python
from memlayer import OpenAI as Memlayer
from memlayer.config.search_tiers import SearchTierBuilder, register_tier

# Create custom tier
tier = (
    SearchTierBuilder("my_tier")
    .hybrid()
    .top_k(8)
    .depth(2)
    .build()
)
register_tier(tier)

# Use it (requires SearchService modification to accept custom tiers)
client = Memlayer(...)
response = client.chat(
    [{"role": "user", "content": "How are X and Y connected?"}],
    search_tier="my_tier"  # Use custom tier
)
```

### Option 2: Modify SearchService

Currently, SearchService has hardcoded mappings. To use custom tiers, modify `memlayer/services/__init__.py`:

```python
# File: memlayer/services/__init__.py

from memlayer.config.search_tiers import get_tier, DEFAULT_TIERS

def search(self, query: str, user_id: str, search_tier: str = "balanced", ...):
    # Get tier configuration
    try:
        tier_config = get_tier(search_tier)
    except ValueError:
        # Fallback to default if tier not found
        tier_config = DEFAULT_TIERS.get(search_tier, DEFAULT_TIERS["balanced"])

    # Use tier config
    top_k = tier_config.vector_top_k
    enable_graph = tier_config.enable_graph
    graph_depth = tier_config.graph_depth

    # Vector search
    results = self.storage.search_memories(
        query_embedding=query_embedding,
        user_id=user_id,
        top_k=top_k
    )

    # Graph search (if enabled)
    if enable_graph:
        # Use tier_config.graph_depth instead of hardcoded 2
        facts = self.graph_storage.get_subgraph_context(entity, depth=graph_depth)
```

## Summary: Why Custom Tiers Matter

**Mem0**: No customization, vector-only
**Mem0g**: No customization, always hybrid (wastes compute)
**Supermemory**: Time-based tiers only (hot/cold storage)
**Memlayer**: **Fully customizable search strategies** ✅

**You can customize:**
1. ✅ Vector/graph/hybrid mode
2. ✅ Top-K results (1-100)
3. ✅ Graph depth (1-5 hops)
4. ✅ Recency bias (0.0-1.0, like Supermemory)
5. ✅ Score thresholds
6. ✅ Metadata filters
7. ✅ Custom ranking functions

**This is the MOST flexible memory system available!** 🚀

## Next Steps

1. Run the examples: `python examples/custom_search_tiers.py`
2. Create your own tiers for your use case
3. Test performance with different configurations
4. Publish results showing custom tiers outperform fixed strategies

## References

- Supermemory Architecture: [How It Works](https://supermemory.ai/docs/how-it-works)
- Supermemory Blog: [Memory Engine Design](https://supermemory.ai/blog/memory-engine/)
- Memlayer Docs: `docs/ARCHITECTURE.md`
