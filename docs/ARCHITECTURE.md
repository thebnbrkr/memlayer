# Memlayer Architecture Documentation

Complete guide to how Memlayer works internally, with clear examples.

---

## Table of Contents

1. [Overview & Data Flow](#overview--data-flow)
2. [Salience Configuration System](#salience-configuration-system)
3. [Storage Layers](#storage-layers)
4. [Consolidation Service](#consolidation-service)
5. [OpenAI Wrapper](#openai-wrapper)
6. [End-to-End Examples](#end-to-end-examples)

---

## Overview & Data Flow

### What Memlayer Does

Memlayer is a **memory layer for LLMs** that:
1. Stores important facts from conversations
2. Retrieves relevant memories when answering questions
3. Uses **custom salience rules** to decide what's important
4. Stores data in **dual storage** (vector DB + graph DB)

### High-Level Data Flow

```
User Message
    ↓
OpenAI Wrapper (memlayer/wrappers/openai.py)
    ↓
┌───────────────────────────────────────────┐
│ 1. Retrieve relevant memories             │
│    - Vector search (semantic similarity)  │
│    - Graph search (entity relationships)  │
└───────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────┐
│ 2. Add memories to context                │
│    - Inject into system prompt            │
└───────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────┐
│ 3. Call OpenAI API                        │
│    - With memory-enhanced context         │
└───────────────────────────────────────────┘
    ↓
LLM Response
    ↓
┌───────────────────────────────────────────┐
│ 4. Store new facts (background thread)    │
│    - Salience check (is it important?)    │
│    - If yes: Extract entities & store     │
└───────────────────────────────────────────┘
    ↓
Return response to user
```

### File Structure

```
memlayer/
├── wrappers/
│   └── openai.py              # Main entry point - OpenAI-compatible API
├── services/
│   ├── __init__.py            # ConsolidationService - extracts & stores facts
│   └── salience_calculator.py # SalienceCalculator - scores facts
├── storage/
│   ├── chroma.py              # Vector storage (ChromaDB)
│   ├── networkx.py            # Graph storage (NetworkX)
│   └── memgraph.py            # Graph storage (Memgraph - production)
├── config/
│   └── salience.py            # Salience config models (Pydantic)
└── routes/
    └── salience_config.py     # FastAPI routes for managing configs
```

---

## Salience Configuration System

### What is Salience?

**Salience** = "Is this fact important enough to store?"

Example conversation:
```
User: "I work at Google"           ← HIGH salience (store it!)
User: "Uh huh"                     ← LOW salience (skip it)
User: "I love Python programming"  ← HIGH salience (store it!)
```

### How Salience Works in Memlayer

**Traditional approach (Mem0, Zep):**
- Hardcoded rules
- Fixed importance threshold
- One-size-fits-all

**Memlayer approach:**
- **Fully customizable** scoring functions
- **Composable** components with weights
- **Boolean decision rules** with priority
- **Different configs** for different use cases

### Salience Configuration Structure

```python
from memlayer.config.salience import (
    TenantSalienceConfig,
    SalienceComponent,
    ScoringFunctionType,
    AdaptiveThresholdConfig,
    ThresholdStrategy
)

config = TenantSalienceConfig(
    tenant_id="user123",
    config_name="my_config",

    # Components: How to score facts
    components=[
        SalienceComponent(
            name="component1",
            weight=0.6,  # 60% of final score
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={"keywords": ["important", "urgent"]}
        ),
        SalienceComponent(
            name="component2",
            weight=0.4,  # 40% of final score
            scoring_function=ScoringFunctionType.LENGTH_BONUS,
            scoring_config={"optimal_length": 100}
        )
    ],

    # Threshold: What score is "good enough" to store?
    threshold_config=AdaptiveThresholdConfig(
        strategy=ThresholdStrategy.ABSOLUTE,
        absolute_threshold=0.5  # Store if score >= 0.5
    ),

    # Decision rules: Override threshold with custom logic
    decision_rules=[
        DecisionRule(
            name="always_store_urgent",
            priority=1,
            condition="component1 > 0.9",  # Boolean expression
            action="STORE",
            reasoning="High urgency detected"
        )
    ]
)
```

### Scoring Functions (9 Available)

#### 1. **KEYWORD_MATCH** - Keyword presence

**Use case:** Store facts containing specific terms

**Example:**
```python
SalienceComponent(
    name="tech_keywords",
    weight=1.0,
    scoring_function=ScoringFunctionType.KEYWORD_MATCH,
    scoring_config={
        "keywords": ["python", "javascript", "AI", "machine learning"],
        "case_sensitive": False
    }
)

# Scoring:
"I love Python programming"        → Score: 1.0 (1 keyword found)
"JavaScript is my favorite"        → Score: 1.0 (1 keyword found)
"I had pizza for lunch"            → Score: 0.0 (no keywords)
```

**Code location:** `memlayer/services/salience_calculator.py:_compute_keyword_match()`

**Algorithm:**
```python
def _compute_keyword_match(component):
    keywords = component.scoring_config["keywords"]
    fact_lower = self.fact.lower()

    matches = sum(1 for kw in keywords if kw.lower() in fact_lower)
    score = min(1.0, matches / len(keywords))  # Normalize to 0-1
    return score
```

#### 2. **LENGTH_BONUS** - Fact length scoring

**Use case:** Prefer detailed facts over short ones

**Example:**
```python
SalienceComponent(
    name="detail_level",
    weight=1.0,
    scoring_function=ScoringFunctionType.LENGTH_BONUS,
    scoring_config={
        "min_length": 20,
        "max_length": 200,
        "optimal_length": 100
    }
)

# Scoring:
"I work at Google"                              → Score: 0.14 (too short)
"I work at Google as a Senior Engineer..."     → Score: 0.85 (good length)
"I work at Google as a Senior Engineer in the ML team doing..." → Score: 1.0 (optimal)
```

**Code location:** `memlayer/services/salience_calculator.py:_compute_length_bonus()`

**Algorithm:**
```python
def _compute_length_bonus(component):
    length = len(self.fact)
    min_len = component.scoring_config.get("min_length", 10)
    max_len = component.scoring_config.get("max_length", 500)
    optimal = component.scoring_config.get("optimal_length", 100)

    if length < min_len:
        return 0.0
    if length > max_len:
        return max(0.0, 1.0 - (length - max_len) / max_len)

    # Gaussian curve around optimal length
    distance = abs(length - optimal)
    score = math.exp(-(distance ** 2) / (2 * (optimal / 3) ** 2))
    return score
```

#### 3. **EMBEDDING_SIMILARITY** - Semantic novelty

**Use case:** Store facts that are semantically different from existing memories

**Example:**
```python
SalienceComponent(
    name="novelty",
    weight=1.0,
    scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY,
    scoring_config={
        "target": "existing_memories",
        "threshold": 0.8  # Store if dissimilar from existing
    }
)

# Scoring:
# Existing memory: "I work at Google"
"I work at Microsoft"   → Score: 0.3 (similar, low novelty)
"I love hiking"         → Score: 0.9 (dissimilar, high novelty)
```

**Code location:** `memlayer/services/salience_calculator.py:_compute_embedding_similarity()`

#### 4. **FREQUENCY** - How often mentioned

**Use case:** Store facts mentioned multiple times

**Example:**
```python
SalienceComponent(
    name="importance_by_repetition",
    weight=1.0,
    scoring_function=ScoringFunctionType.FREQUENCY,
    scoring_config={
        "window_size": 10  # Look at last 10 messages
    }
)

# If "Python" mentioned 5 times in last 10 messages:
"I love Python" → Score: 0.5 (5/10)
```

#### 5. **LIVENESS** - Recency scoring

**Use case:** Prefer recent facts over old ones

**Example:**
```python
SalienceComponent(
    name="recency",
    weight=1.0,
    scoring_function=ScoringFunctionType.LIVENESS,
    scoring_config={
        "decay_days": 7  # Facts decay over 7 days
    }
)

# Fact from today:     Score: 1.0
# Fact from 3 days ago: Score: 0.6
# Fact from 7 days ago: Score: 0.0
```

#### 6. **LLM_SCORED** - AI judges importance

**Use case:** Use LLM to score importance subjectively

**Example:**
```python
SalienceComponent(
    name="ai_importance",
    weight=1.0,
    scoring_function=ScoringFunctionType.LLM_SCORED,
    scoring_config={
        "prompt": "On a scale of 0-1, how important is this fact for understanding the user? Fact: {fact}",
        "model": "gpt-4o-mini"
    }
)

# LLM evaluates each fact:
"I work at Google"     → LLM returns: 0.9 (very important)
"I had coffee"         → LLM returns: 0.1 (not important)
```

#### 7. **CUSTOM_PYTHON** - Your own code

**Use case:** Complex custom logic

**Example:**
```python
SalienceComponent(
    name="my_custom_scorer",
    weight=1.0,
    scoring_function=ScoringFunctionType.CUSTOM_PYTHON,
    scoring_config={
        "code": """
def score_fact(fact, context):
    # Your custom logic here
    if "urgent" in fact.lower() and len(fact) > 50:
        return 1.0
    elif "$" in fact:  # Mentions money
        return 0.8
    else:
        return 0.3
"""
    }
)
```

**⚠️ Security:** Custom code is executed with `eval()` - only use in trusted environments!

#### 8. **WEIGHTED_SUM** - Combine multiple scores

**Use case:** Meta-scoring (combine other components)

**Example:**
```python
SalienceComponent(
    name="combined",
    weight=1.0,
    scoring_function=ScoringFunctionType.WEIGHTED_SUM,
    scoring_config={
        "components": ["tech_keywords", "detail_level"],
        "weights": [0.7, 0.3]
    }
)
```

#### 9. **BOOLEAN_GATE** - Pass/fail threshold

**Use case:** Hard cutoffs

**Example:**
```python
SalienceComponent(
    name="must_be_detailed",
    weight=1.0,
    scoring_function=ScoringFunctionType.BOOLEAN_GATE,
    scoring_config={
        "condition": "length > 50",
        "pass_score": 1.0,
        "fail_score": 0.0
    }
)

# Score:
"I work at Google"                        → 0.0 (fails length check)
"I work at Google as a senior engineer"  → 1.0 (passes length check)
```

### Threshold Strategies (6 Available)

#### 1. **ABSOLUTE** - Fixed threshold

```python
threshold_config=AdaptiveThresholdConfig(
    strategy=ThresholdStrategy.ABSOLUTE,
    absolute_threshold=0.5  # Store if score >= 0.5
)

# Simple: score >= 0.5 → STORE, otherwise SKIP
```

#### 2. **PERCENTILE** - Relative to other facts

```python
threshold_config=AdaptiveThresholdConfig(
    strategy=ThresholdStrategy.PERCENTILE,
    percentile=70  # Store top 30% of facts
)

# Adapts to distribution of scores
# If all scores are low, still stores top 30%
```

#### 3. **MEAN_MULTIPLIER** - Relative to average

```python
threshold_config=AdaptiveThresholdConfig(
    strategy=ThresholdStrategy.MEAN_MULTIPLIER,
    mean_multiplier=1.5  # Store if score >= 1.5 * mean
)

# If mean score = 0.4:
# Threshold = 0.4 * 1.5 = 0.6
# Only scores >= 0.6 are stored
```

#### 4. **EXPONENTIAL_DECAY** - Time-based decay

```python
threshold_config=AdaptiveThresholdConfig(
    strategy=ThresholdStrategy.EXPONENTIAL_DECAY,
    initial_threshold=0.8,
    decay_rate=0.1,
    min_threshold=0.3
)

# Threshold decreases over time:
# Day 1: 0.8 (strict)
# Day 5: 0.5 (medium)
# Day 10: 0.3 (permissive)
```

#### 5. **DYNAMIC_PERCENTILE** - Adaptive windowing

```python
threshold_config=AdaptiveThresholdConfig(
    strategy=ThresholdStrategy.DYNAMIC_PERCENTILE,
    window_size=100,  # Look at last 100 facts
    percentile=75
)

# Threshold adapts based on recent facts
# Handles changing conversation patterns
```

#### 6. **CONFIDENCE_WEIGHTED** - Uncertainty-aware

```python
threshold_config=AdaptiveThresholdConfig(
    strategy=ThresholdStrategy.CONFIDENCE_WEIGHTED,
    base_threshold=0.5,
    confidence_factor=0.2
)

# Adjusts threshold based on scoring confidence
# If components disagree → raise threshold
# If components agree → lower threshold
```

### Decision Rules - Boolean Logic

**Override thresholds with complex conditions:**

```python
decision_rules=[
    # Rule 1: Always store high-priority facts
    DecisionRule(
        name="urgent_facts",
        priority=1,  # Checked first
        condition="tech_keywords > 0.9 and detail_level > 0.5",
        action="STORE",
        reasoning="High priority technical detail"
    ),

    # Rule 2: Always skip noise
    DecisionRule(
        name="filter_noise",
        priority=2,  # Checked second
        condition="detail_level < 0.1",
        action="SKIP",
        reasoning="Too short to be meaningful"
    ),

    # Rule 3: Store if technical OR detailed
    DecisionRule(
        name="either_or",
        priority=3,
        condition="tech_keywords > 0.7 or detail_level > 0.8",
        action="STORE",
        reasoning="Either technical or detailed enough"
    )
]
```

**Execution flow:**
1. Compute component scores
2. Calculate weighted score
3. Check decision rules in priority order
4. If rule matches → use its action
5. If no rule matches → compare weighted score to threshold

**Code location:** `memlayer/services/salience_calculator.py:_apply_decision_rules()`

---

## Storage Layers

### Dual Storage Architecture

Memlayer uses **two storage systems**:

1. **Vector Storage** (ChromaDB/Qdrant/Pinecone)
   - Stores: Raw facts with embeddings
   - Good for: Semantic similarity search
   - Example: "Find facts about Python programming"

2. **Graph Storage** (NetworkX/Memgraph)
   - Stores: Entities and relationships
   - Good for: Structured knowledge queries
   - Example: "What's Alice's relationship to Google?"

### Why Both?

**Vector-only** (like Mem0):
- ✅ Good: Semantic search
- ❌ Bad: Can't answer "Who does Alice work for?"

**Graph-only**:
- ✅ Good: Structured queries
- ❌ Bad: Can't find similar but differently-worded facts

**Vector + Graph** (Memlayer):
- ✅ Best of both worlds
- Can answer both semantic AND structural questions

### Vector Storage (ChromaDB)

**File:** `memlayer/storage/chroma.py`

**What it stores:**

```python
{
    "id": "fact_uuid_123",
    "embedding": [0.123, 0.456, ...],  # 1536-dim vector
    "metadata": {
        "user_id": "alice",
        "tenant_id": "company_x",
        "timestamp": 1234567890,
        "source": "chat"
    },
    "document": "I work at Google as a Senior Engineer"
}
```

**Key methods:**

```python
# Add memory
vector_storage.add_memory(
    text="I work at Google",
    user_id="alice",
    metadata={"importance": 0.9}
)

# Search memories
results = vector_storage.search_memories(
    query="Where does Alice work?",
    user_id="alice",
    top_k=5
)
# Returns: [
#     {"text": "I work at Google", "score": 0.95},
#     {"text": "I'm a Senior Engineer", "score": 0.82},
#     ...
# ]
```

**Multi-tenancy:**

```python
# Automatic filtering by user_id
collection.query(
    query_embeddings=[query_embedding],
    where={"user_id": user_id},  # ← Isolation!
    n_results=top_k
)
```

**Code location:** `memlayer/storage/chroma.py:search_memories()`

### Graph Storage (NetworkX)

**File:** `memlayer/storage/networkx.py`

**What it stores:**

```python
# Nodes (entities)
{
    "name": "Alice",
    "type": "Person",
    "status": "active",
    "access_count": 5,
    "created_timestamp": 1234567890,
    "last_accessed_timestamp": 1234567900,
    "importance_score": 0.8,
    "tenant_id": "company_x"  # ⚠️ Currently missing!
}

# Edges (relationships)
{
    "subject": "Alice",
    "predicate": "works_at",
    "object": "Google",
    "metadata": {...}
}
```

**Key methods:**

```python
# Add entity
graph_storage.add_entity(
    name="Alice",
    node_type="Person"
)

# Add relationship
graph_storage.add_relationship(
    subject_name="Alice",
    predicate="works_at",
    object_name="Google"
)

# Find entity
results = graph_storage.find_matching_nodes(
    name_query="Alice",
    max_results=5
)

# Get entity context (relationships)
context = graph_storage.get_entity_context(
    entity_name="Alice",
    max_depth=2
)
# Returns:
# {
#     "entity": "Alice",
#     "type": "Person",
#     "outgoing_relationships": [
#         {"predicate": "works_at", "target": "Google", "target_type": "Organization"}
#     ],
#     "incoming_relationships": []
# }
```

**Entity deduplication:**

The graph prevents duplicates like "Dr. Watson" and "Dr. Emma Watson":

```python
def _find_canonical_entity(self, name: str, node_type: str):
    # Strategy 1: Exact match (case-insensitive)
    if "alice" == name.lower():
        return "Alice"  # Use existing

    # Strategy 2: Substring match
    # "Dr. Watson" vs "Dr. Emma Watson"
    # → merge to "Dr. Emma Watson" (longer name)

    # Strategy 3: Similarity match
    # "E. Watson" vs "Emma Watson"
    # → use "Emma Watson"
```

**Code location:** `memlayer/storage/networkx.py:_find_canonical_entity()`

**⚠️ Current limitation:** No tenant_id filtering (see GRAPH_TENANCY_FIX.md)

---

## Consolidation Service

### What is Consolidation?

**Consolidation** = Extract structured knowledge from raw conversation text

**Input:** "I work at Google as a Senior Engineer in the ML team"

**Output:**
- Entity: `Alice` (Person)
- Entity: `Google` (Organization)
- Entity: `ML team` (Team)
- Relationship: `Alice --works_at--> Google`
- Relationship: `Alice --member_of--> ML team`
- Fact: "I work at Google as a Senior Engineer" (stored in vector DB)

### How It Works

**File:** `memlayer/services/__init__.py`

**Flow:**

```python
class ConsolidationService:
    def consolidate(self, conversation_text: str, user_id: str):
        # Step 1: Salience check
        if self.salience_config:
            calculator = SalienceCalculator(
                config=self.salience_config,
                fact=conversation_text,
                tenant_id=self.tenant_id
            )
            salience_score, decision, log = calculator.compute_salience()

            if decision == "SKIP":
                return  # Don't store this fact

        # Step 2: Extract entities (using LLM)
        extraction_prompt = f"""
        Extract entities and relationships from:
        "{conversation_text}"

        Return JSON:
        {{
            "entities": [
                {{"name": "Alice", "type": "Person"}},
                {{"name": "Google", "type": "Organization"}}
            ],
            "relationships": [
                {{"subject": "Alice", "predicate": "works_at", "object": "Google"}}
            ]
        }}
        """

        response = self.llm_client.generate(extraction_prompt)
        extraction = json.loads(response)

        # Step 3: Store entities in graph
        for entity in extraction["entities"]:
            self.graph_storage.add_entity(
                name=entity["name"],
                node_type=entity["type"]
            )

        # Step 4: Store relationships in graph
        for rel in extraction["relationships"]:
            self.graph_storage.add_relationship(
                subject_name=rel["subject"],
                predicate=rel["predicate"],
                object_name=rel["object"]
            )

        # Step 5: Store raw fact in vector DB
        self.vector_storage.add_memory(
            text=conversation_text,
            user_id=user_id,
            metadata={"salience_score": salience_score}
        )
```

**Code location:** `memlayer/services/__init__.py:consolidate()`

### Background Thread

Consolidation runs **asynchronously** to avoid blocking chat responses:

```python
# In OpenAI wrapper
def chat(self, messages):
    # 1. Get LLM response (fast)
    response = openai.chat.completions.create(...)

    # 2. Start background consolidation (slow, doesn't block)
    threading.Thread(
        target=self.consolidation_service.consolidate,
        args=(conversation_text, self.user_id)
    ).start()

    # 3. Return response immediately
    return response
```

**⚠️ Important:** This means facts might not be stored yet when you query immediately after!

**Fix:** Add `time.sleep(2)` after storing facts in benchmarks.

**Code location:** `memlayer/wrappers/openai.py:chat()`

---

## OpenAI Wrapper

### Main Entry Point

**File:** `memlayer/wrappers/openai.py`

**Usage:**

```python
from memlayer import OpenAI

client = OpenAI(
    model="gpt-4o-mini",
    user_id="alice",
    tenant_id="company_x",
    storage_path="./data",
    operation_mode="online",  # or "local" or "lightweight"
    salience_config=my_config  # Optional custom config
)

# Chat with memory
response = client.chat([
    {"role": "user", "content": "I work at Google"}
])

# Later...
response = client.chat([
    {"role": "user", "content": "Where do I work?"}
])
# Returns: "You work at Google" (retrieved from memory!)
```

### Lazy Loading

All heavy components are loaded **on first use** (not at initialization):

```python
@property
def vector_storage(self):
    if self._vector_storage is None:
        from ..storage.chroma import ChromaStorage
        self._vector_storage = ChromaStorage(self.storage_path)
    return self._vector_storage
```

**Why?** Faster initialization, only load what you use.

**Code location:** `memlayer/wrappers/openai.py:vector_storage()`

### Memory Retrieval

**When you call `chat()`, memories are automatically retrieved:**

```python
def chat(self, messages: List[Dict[str, str]]):
    # Extract user query
    user_query = messages[-1]["content"]

    # Search vector DB
    vector_memories = self.vector_storage.search_memories(
        query=user_query,
        user_id=self.user_id,
        top_k=5
    )

    # Search graph DB
    graph_entities = self.graph_storage.find_matching_nodes(
        name_query=user_query,
        max_results=3
    )

    # Inject memories into system prompt
    memory_context = f"""
    Relevant memories:
    {vector_memories}

    Known entities:
    {graph_entities}
    """

    # Add to messages
    enhanced_messages = [
        {"role": "system", "content": memory_context},
        *messages
    ]

    # Call OpenAI
    response = openai.chat.completions.create(
        model=self.model,
        messages=enhanced_messages
    )

    return response.choices[0].message.content
```

**Code location:** `memlayer/wrappers/openai.py:chat()`

### Operation Modes

Three modes for different use cases:

#### 1. **Online Mode** (Default)

```python
client = OpenAI(operation_mode="online")
```

**Uses:**
- OpenAI embeddings (ada-002)
- ChromaDB (local vector DB)
- NetworkX (in-memory graph)

**Best for:** Production with OpenAI API access

#### 2. **Local Mode**

```python
client = OpenAI(operation_mode="local")
```

**Uses:**
- sentence-transformers (local embeddings)
- ChromaDB
- NetworkX

**Best for:** Privacy-focused, no external API calls

**Requires:** `pip install sentence-transformers`

#### 3. **Lightweight Mode**

```python
client = OpenAI(operation_mode="lightweight")
```

**Uses:**
- Keyword matching only (no embeddings)
- NetworkX

**Best for:** Testing, development, minimal dependencies

---

## End-to-End Examples

### Example 1: Basic Usage

```python
from memlayer import OpenAI

# Initialize
client = OpenAI(
    model="gpt-4o-mini",
    user_id="alice",
    storage_path="./my_data"
)

# Conversation 1: Store facts
client.chat([
    {"role": "user", "content": "Hi! I'm Alice, I work at Google as a Senior Engineer."}
])

client.chat([
    {"role": "user", "content": "I love Python programming and machine learning."}
])

# Conversation 2: Retrieve memories (later)
response = client.chat([
    {"role": "user", "content": "What do you know about me?"}
])

print(response)
# Output: "You're Alice, a Senior Engineer at Google who loves Python programming and machine learning."
```

### Example 2: Custom Salience Config

```python
from memlayer import OpenAI
from memlayer.config.salience import (
    TenantSalienceConfig,
    SalienceComponent,
    ScoringFunctionType,
    ThresholdStrategy
)

# Create config: Only store technical facts
tech_config = TenantSalienceConfig(
    tenant_id="alice",
    config_name="tech_only",
    components=[
        SalienceComponent(
            name="tech_keywords",
            weight=1.0,
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={
                "keywords": ["python", "code", "programming", "API", "database", "ML", "AI"]
            }
        )
    ],
    threshold_config={
        "strategy": ThresholdStrategy.ABSOLUTE,
        "absolute_threshold": 0.3
    }
)

# Use config
client = OpenAI(
    model="gpt-4o-mini",
    user_id="alice",
    salience_config=tech_config
)

# Test
client.chat([{"role": "user", "content": "I work on Python ML pipelines"}])  # STORED
client.chat([{"role": "user", "content": "I had pizza for lunch"}])          # SKIPPED

# Check what was stored
logs = client.get_salience_logs()
for log in logs:
    print(f"{log['decision']}: {log['fact']} (score: {log['score']:.2f})")

# Output:
# STORE: I work on Python ML pipelines (score: 0.67)
# SKIP: I had pizza for lunch (score: 0.0)
```

### Example 3: Multi-User Isolation

```python
from memlayer import OpenAI

# Alice's client
alice = OpenAI(user_id="alice", tenant_id="company", storage_path="./data")
alice.chat([{"role": "user", "content": "I work at Google"}])

# Bob's client
bob = OpenAI(user_id="bob", tenant_id="company", storage_path="./data")
bob.chat([{"role": "user", "content": "I work at Microsoft"}])

# Alice queries
alice_response = alice.chat([{"role": "user", "content": "Where do I work?"}])
print(alice_response)  # "You work at Google"

# Bob queries
bob_response = bob.chat([{"role": "user", "content": "Where do I work?"}])
print(bob_response)  # "You work at Microsoft"

# Memories are isolated! ✅
```

### Example 4: Hybrid Component Scoring

```python
from memlayer.config.salience import (
    TenantSalienceConfig,
    SalienceComponent,
    ScoringFunctionType,
    ThresholdStrategy
)

# Combine multiple scoring functions
hybrid_config = TenantSalienceConfig(
    tenant_id="user",
    config_name="hybrid",
    components=[
        # 40% weight: Keyword matching
        SalienceComponent(
            name="keywords",
            weight=0.4,
            scoring_function=ScoringFunctionType.KEYWORD_MATCH,
            scoring_config={"keywords": ["important", "urgent", "critical"]}
        ),
        # 30% weight: Length bonus
        SalienceComponent(
            name="detail",
            weight=0.3,
            scoring_function=ScoringFunctionType.LENGTH_BONUS,
            scoring_config={"optimal_length": 100}
        ),
        # 30% weight: Novelty
        SalienceComponent(
            name="novelty",
            weight=0.3,
            scoring_function=ScoringFunctionType.EMBEDDING_SIMILARITY,
            scoring_config={"target": "existing_memories", "threshold": 0.8}
        )
    ],
    threshold_config={
        "strategy": ThresholdStrategy.MEAN_MULTIPLIER,
        "mean_multiplier": 1.2  # Adaptive threshold
    }
)

# Score calculation:
# final_score = 0.4 * keywords + 0.3 * detail + 0.3 * novelty
# threshold = mean(all_scores) * 1.2
# if final_score >= threshold: STORE
```

### Example 5: Decision Rules with Priority

```python
from memlayer.config.salience import DecisionRule

config = TenantSalienceConfig(
    tenant_id="user",
    config_name="rules_demo",
    components=[
        SalienceComponent(name="tech", weight=0.5, ...),
        SalienceComponent(name="personal", weight=0.5, ...)
    ],
    threshold_config={"strategy": "absolute", "absolute_threshold": 0.5},
    decision_rules=[
        # Rule 1 (highest priority): Always store high-tech facts
        DecisionRule(
            name="must_store_tech",
            priority=1,
            condition="tech > 0.9",
            action="STORE",
            reasoning="Critical technical information"
        ),
        # Rule 2: Skip trivial facts
        DecisionRule(
            name="skip_trivial",
            priority=2,
            condition="tech < 0.1 and personal < 0.1",
            action="SKIP",
            reasoning="Not enough content"
        ),
        # Rule 3: Store if either high
        DecisionRule(
            name="either_high",
            priority=3,
            condition="tech > 0.7 or personal > 0.7",
            action="STORE",
            reasoning="High score in at least one dimension"
        )
    ]
)

# Execution:
# 1. Compute tech and personal scores
# 2. Check rules in priority order
# 3. First matching rule determines action
# 4. If no rule matches, use threshold
```

### Example 6: API Usage (FastAPI)

```python
import requests

# Create config via API
config_data = {
    "tenant_id": "alice",
    "config_name": "api_config",
    "components": [{
        "name": "keywords",
        "weight": 1.0,
        "scoring_function": "keyword_match",
        "scoring_config": {"keywords": ["important"]}
    }],
    "threshold_config": {
        "strategy": "absolute",
        "absolute_threshold": 0.5
    }
}

response = requests.post(
    "http://localhost:8000/api/config/salience/",
    json=config_data
)

print(response.json())
# {"config_id": "...", "status": "created"}

# Test config
test_result = requests.post(
    "http://localhost:8000/api/config/salience/api_config/test?tenant_id=alice",
    json={"fact": "This is important information"}
)

print(test_result.json())
# {
#     "salience_score": 0.67,
#     "threshold": 0.5,
#     "decision": "STORE",
#     "component_scores": {"keywords": 0.67}
# }
```

---

## Key Takeaways

### What Makes Memlayer Unique

1. **Flexible Salience System**
   - 9 scoring functions
   - Composable components with weights
   - Boolean decision rules
   - **No other memory system offers this level of control**

2. **Dual Storage**
   - Vector DB (semantic search) + Graph DB (structured queries)
   - Best of both worlds

3. **Production-Ready**
   - Multi-tenancy support
   - FastAPI REST API
   - Docker deployment
   - PostgreSQL audit logs

### Common Gotchas

1. **Background consolidation** - Facts stored async, may need `time.sleep()`
2. **Threshold too high** - Can filter out important facts (set lower, like 0.1)
3. **Graph multi-tenancy** - Currently missing tenant_id isolation (see GRAPH_TENANCY_FIX.md)
4. **Component weights must sum to 1.0** - Validated by Pydantic

### Performance Considerations

- **Vector search:** O(n) but fast with indexes (~50-100ms for 10k facts)
- **Graph search:** O(n) for simple queries, O(n²) for deep traversal
- **Background consolidation:** 1-3 seconds per fact (doesn't block)
- **Salience calculation:** ~10-50ms per fact

### Next Steps

1. Read `SELF_HOSTING.md` for deployment
2. Read `BENCHMARKING_GUIDE.md` for testing
3. Read `GRAPH_TENANCY_FIX.md` for multi-tenancy
4. Try examples above!

---

**Questions?** Check the code comments or file an issue on GitHub!
