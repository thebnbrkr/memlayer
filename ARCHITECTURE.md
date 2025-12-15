# Memlayer Architecture - Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [Component Breakdown](#component-breakdown)
4. [Data Flow](#data-flow)
5. [Memory Lifecycle](#memory-lifecycle)
6. [Operation Modes](#operation-modes)
7. [Search Tiers](#search-tiers)
8. [Storage Systems](#storage-systems)
9. [LLM Provider Integration](#llm-provider-integration)
10. [Advanced Features](#advanced-features)
11. [Configuration Guide](#configuration-guide)

---

## Overview

**Memlayer** is a plug-and-play memory layer that adds persistent, intelligent memory to any Large Language Model (LLM). It acts as a wrapper around popular LLM providers (OpenAI, Claude, Gemini, Ollama, LMStudio) and automatically:

1. **Filters** salient (important) information from conversations
2. **Extracts** structured knowledge (facts, entities, relationships)
3. **Stores** memories in hybrid vector + graph databases
4. **Retrieves** relevant context when needed
5. **Injects** memories seamlessly into LLM prompts

### Key Value Proposition
- **3 lines of code** to add memory to any LLM
- **<100ms** fast search capabilities
- **Zero configuration** required for basic usage
- **100% local** option available (no API dependencies)
- **Production-ready** with built-in observability

---

## Core Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER APPLICATION                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MEMLAYER WRAPPER (OpenAI/Claude/Gemini/etc)  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  1. Memory Search Tool (search_memory)                   │   │
│  │  2. Task Scheduling Tool (schedule_task)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────┬──────────────────────────────┬─────────────────────────┘
          │                              │
          ▼                              ▼
┌───────────────────┐         ┌──────────────────────┐
│  SEARCH SERVICE   │         │ CONSOLIDATION SERVICE│
│  - Vector Search  │         │ - Knowledge Extract  │
│  - Graph Search   │         │ - Salience Filter    │
│  - Hybrid Fusion  │         │ - Background Thread  │
└─────────┬─────────┘         └──────────┬───────────┘
          │                              │
          ▼                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    STORAGE LAYER                              │
│  ┌─────────────────────┐      ┌────────────────────────┐     │
│  │  VECTOR STORAGE     │      │   GRAPH STORAGE        │     │
│  │  (ChromaDB/Qdrant)  │      │   (NetworkX)           │     │
│  │  - Semantic Search  │      │   - Entity Relations   │     │
│  │  - Embeddings       │      │   - Knowledge Graph    │     │
│  └─────────────────────┘      └────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌──────────────────────────────────────────────────────────────┐
│                  BACKGROUND SERVICES                          │
│  - SchedulerService (task reminders)                         │
│  - CurationService (memory decay & expiration)               │
│  - SalienceGate (ML-based filtering)                         │
└──────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Layer |
|-----------|---------------|-------|
| **LLM Wrappers** | Interface with specific LLM providers | Application |
| **SearchService** | Retrieve relevant memories | Business Logic |
| **ConsolidationService** | Extract & store knowledge | Business Logic |
| **Vector Storage** | Semantic similarity search | Data |
| **Graph Storage** | Entity relationship queries | Data |
| **SalienceGate** | Filter important information | ML |
| **Background Services** | Task scheduling & curation | System |

---

## Component Breakdown

### 1. LLM Wrappers (`memlayer/wrappers/`)

#### Purpose
Provide a unified interface for different LLM providers while adding memory capabilities.

#### Available Wrappers
- **OpenAI** (`openai.py`) - GPT-4, GPT-4.1, GPT-5, etc.
- **Claude** (`claude.py`) - Claude 3/4 Sonnet, Opus
- **Gemini** (`gemini.py`) - Gemini 2.0/2.5 Flash, Pro
- **Ollama** (`ollama.py`) - Local models (Llama, Mistral, Qwen)
- **LMStudio** (`lmstudio.py`) - LMStudio local server

#### Key Methods
```python
class BaseLLMWrapper:
    def chat(messages: list, **kwargs) -> str:
        """Main chat interface with automatic memory injection"""
        
    def analyze_and_extract_knowledge(text: str) -> Dict:
        """Extract facts, entities, relationships from text"""
        
    def extract_query_entities(query: str) -> List[str]:
        """Extract key entities for graph traversal"""
```

#### Memory Tool Schema
Each wrapper automatically provides two tools to the LLM:

1. **search_memory** - Search long-term memory
   - Parameters: `query` (str), `search_tier` ("fast"|"balanced"|"deep")
   - Automatically called when user asks about past information

2. **schedule_task** - Schedule future reminders
   - Parameters: `task_description` (str), `due_date` (ISO 8601 string)
   - Creates persistent task reminders

### 2. Storage Layer (`memlayer/storage/`)

#### Vector Storage - Semantic Search

**Supported Backends:**
- **ChromaDB** (`chroma.py`) - Default, local-first
- **Qdrant** (`qdrant_backend.py`) - High-performance cloud option
- **Zilliz** (`zilliz_backend.py`) - Managed vector database

**What it stores:**
- **Facts** - Extracted declarative statements
- **Embeddings** - Vector representations (1536-dim for OpenAI)
- **Metadata** - User ID, timestamp, importance score, expiration date

**Key operations:**
```python
# Add memory with embedding
vector_storage.add_memory(
    content="Alice works at TechCorp",
    embedding=[0.123, -0.456, ...],  # 1536-dim vector
    user_id="user_123",
    metadata={"importance_score": 0.8}
)

# Search by semantic similarity
results = vector_storage.search_memories(
    query_embedding=[...],
    user_id="user_123",
    top_k=5
)
```

#### Graph Storage - Relationship Queries

**Backend:** NetworkX (`networkx.py`) - In-memory directed graph

**What it stores:**
- **Entities** - People, organizations, concepts
- **Relationships** - Connections between entities (edges)
- **Tasks** - Scheduled reminders with due dates

**Node Schema:**
```python
{
    "name": "Alice",
    "type": "Person",  # Person, Organization, Concept, Task
    "created_timestamp": 1234567890,
    "access_count": 5,
    "last_accessed_timestamp": 1234567890
}
```

**Edge Schema:**
```python
{
    "subject": "Alice",
    "predicate": "works_at",
    "object": "TechCorp"
}
```

**Key operations:**
```python
# Add entity
graph_storage.add_entity(name="Alice", node_type="Person")

# Add relationship
graph_storage.add_relationship(
    subject_name="Alice",
    predicate="works_at",
    object_name="TechCorp"
)

# Graph traversal (2-hop neighbors)
subgraph = graph_storage.get_subgraph_context("Alice", depth=2)
# Returns: ["(Person) Alice -- works_at --> (Organization) TechCorp"]
```

### 3. Search Service (`memlayer/services/__init__.py`)

#### Hybrid Search Strategy

**Three-tier search system:**

| Tier | Vector Results | Graph Traversal | Latency | Use Case |
|------|---------------|-----------------|---------|----------|
| **Fast** | 2 | No | <100ms | Quick facts, real-time chat |
| **Balanced** | 5 | No | <500ms | General queries (default) |
| **Deep** | 10 | Yes (2-hop) | <2s | Complex reasoning, "tell me everything" |

**Search flow:**
```python
def search(query, user_id, search_tier="balanced"):
    # 1. Generate query embedding (with LRU cache)
    query_embedding = embedding_model.get_embeddings([query])[0]
    
    # 2. Vector similarity search
    vector_results = vector_storage.search_memories(
        query_embedding, user_id, top_k=5
    )
    
    # 3. Graph traversal (only for "deep" tier)
    if search_tier == "deep":
        entities = llm_client.extract_query_entities(query)
        graph_facts = []
        for entity in entities:
            # Find entity in graph
            matching_nodes = graph_storage.find_matching_nodes(entity)
            # Get 2-hop neighborhood
            subgraph = graph_storage.get_subgraph_context(
                matching_nodes[0], depth=2
            )
            graph_facts.extend(subgraph)
    
    # 4. Combine & format results
    return f"{vector_context}\n{graph_context}"
```

### 4. Consolidation Service (`memlayer/services/__init__.py`)

#### Knowledge Extraction Pipeline

**Purpose:** Extract structured knowledge from conversations and store it in both vector and graph databases.

**Process (runs in background thread):**

```python
def consolidate(conversation_text, user_id):
    # 1. Salience Check (filter trivial content)
    if not salience_gate.is_worth_saving(conversation_text):
        return  # Skip consolidation
    
    # 2. Knowledge Extraction (via LLM)
    knowledge = llm_client.analyze_and_extract_knowledge(conversation_text)
    # Returns:
    # {
    #   "facts": [{"fact": "...", "importance_score": 0.8}],
    #   "entities": [{"name": "Alice", "type": "Person"}],
    #   "relationships": [{"subject": "Alice", "predicate": "works_at", "object": "TechCorp"}]
    # }
    
    # 3. Store facts in vector database
    for fact in knowledge["facts"]:
        embedding = embedding_model.get_embeddings([fact["fact"]])[0]
        vector_storage.add_memory(
            content=fact["fact"],
            embedding=embedding,
            user_id=user_id,
            metadata={"importance_score": fact.get("importance_score", 0.5)}
        )
    
    # 4. Store entities & relationships in graph
    for entity in knowledge["entities"]:
        graph_storage.add_entity(entity["name"], entity["type"])
    
    for rel in knowledge["relationships"]:
        graph_storage.add_relationship(
            rel["subject"], rel["predicate"], rel["object"]
        )
```

**Performance:** Runs asynchronously (1-3s) - doesn't block user response.

### 5. Salience Gate (`memlayer/ml_gate.py`)

#### Purpose
Filter out trivial conversation content (greetings, acknowledgments) before storing.

#### Three Modes

| Mode | Method | Startup Time | Accuracy | Cost |
|------|--------|--------------|----------|------|
| **LOCAL** | sentence-transformers | ~10s | High | Free |
| **ONLINE** | OpenAI embeddings API | ~2s | High | $0.0001/op |
| **LIGHTWEIGHT** | Keyword-based | <1s | Medium | Free |

#### How it works (LOCAL/ONLINE mode):

```python
# 1. Define prototypes
SALIENT_PROTOTYPES = [
    "My name is Sarah and I work in Marketing.",
    "The project deadline is next Friday.",
    "I prefer all reports in PDF format."
]

NON_SALIENT_PROTOTYPES = [
    "Hello, how are you?",
    "Thank you!",
    "Got it, thanks."
]

# 2. Compute prototype embeddings (done once at startup)
salient_embeddings = embedding_model.get_embeddings(SALIENT_PROTOTYPES)
non_salient_embeddings = embedding_model.get_embeddings(NON_SALIENT_PROTOTYPES)

# 3. Classify new text
def is_worth_saving(text):
    text_embedding = embedding_model.get_embeddings([text])[0]
    
    # Compute similarities to prototypes
    salient_sim = max(cosine_similarity(text_embedding, e) 
                     for e in salient_embeddings)
    non_salient_sim = max(cosine_similarity(text_embedding, e) 
                         for e in non_salient_embeddings)
    
    # Decision: margin-based classification
    score = salient_sim - non_salient_sim
    return score > threshold  # Default threshold = 0.0
```

#### LIGHTWEIGHT mode (keyword-based):
```python
SALIENT_KEYWORDS = ["name", "prefer", "deadline", "important", "remember"]
NON_SALIENT_KEYWORDS = ["hello", "thanks", "okay", "bye"]

def is_worth_saving(text):
    text_lower = text.lower()
    salient_count = sum(1 for kw in SALIENT_KEYWORDS if kw in text_lower)
    non_salient_count = sum(1 for kw in NON_SALIENT_KEYWORDS if kw in text_lower)
    return salient_count > non_salient_count
```

### 6. Background Services

#### SchedulerService
**Purpose:** Check for due tasks and mark them as "triggered"

```python
# Check every 60 seconds (configurable)
while not stopped:
    pending_tasks = graph_storage.get_pending_tasks()
    for task in pending_tasks:
        if task["due_timestamp"] <= current_time:
            graph_storage.update_task_status(task["id"], "triggered")
    
    sleep(60)
```

#### CurationService
**Purpose:** Expire time-sensitive facts and archive low-relevance memories

```python
# Check every 3600 seconds (1 hour, configurable)
while not stopped:
    all_memories = get_all_memories()
    
    for memory in all_memories:
        # 1. Hard expiration (Librarian)
        if memory["expiration_timestamp"] < current_time:
            delete_memory(memory["id"])
        
        # 2. Soft decay (Gardener)
        relevance = calculate_relevance(memory)
        if relevance < 0.3:
            archive_memory(memory["id"])
    
    sleep(3600)

def calculate_relevance(memory):
    age_days = (now - memory["created_timestamp"]) / 86400
    recency_boost = max(0, 1 - (now - memory["last_accessed"]) / (86400 * 7))
    attention_score = log(1 + memory["access_count"])
    
    return (memory["importance_score"] + attention_score + recency_boost) / age_days
```

---

## Data Flow

### Complete Chat Flow (Non-Streaming)

```
1. USER: "What's my name?"
   │
   ▼
2. LLM Wrapper receives message
   │
   ▼
3. LLM makes tool call: search_memory(query="user's name", tier="fast")
   │
   ▼
4. SearchService.search()
   ├─► Generate query embedding (cached)
   ├─► Vector search → ["My name is Alice"] (similarity: 0.95)
   └─► Return: "Relevant memories: My name is Alice"
   │
   ▼
5. LLM receives search results in context
   │
   ▼
6. LLM generates response: "Your name is Alice."
   │
   ├─► Return response to user (IMMEDIATE)
   │
   └─► Background thread starts (NON-BLOCKING):
       ├─► Salience check: "Your name is Alice" → NOT SALIENT (skip)
       └─► Consolidation skipped
```

### Knowledge Storage Flow

```
1. USER: "My name is Alice and I work at TechCorp in London."
   │
   ▼
2. LLM responds: "Nice to meet you, Alice!"
   │
   ├─► Return to user (IMMEDIATE)
   │
   └─► Background consolidation thread:
       │
       ▼
   3. Salience check:
      Similarity to salient prototypes: 0.85
      Similarity to non-salient prototypes: 0.23
      Score: 0.85 - 0.23 = 0.62 > 0.0 threshold → SAVE
      │
      ▼
   4. Knowledge extraction (LLM call):
      {
        "facts": [
          {"fact": "User's name is Alice", "importance_score": 0.9},
          {"fact": "Alice works at TechCorp", "importance_score": 0.8},
          {"fact": "Alice is located in London", "importance_score": 0.7}
        ],
        "entities": [
          {"name": "Alice", "type": "Person"},
          {"name": "TechCorp", "type": "Organization"},
          {"name": "London", "type": "Location"}
        ],
        "relationships": [
          {"subject": "Alice", "predicate": "works_at", "object": "TechCorp"},
          {"subject": "Alice", "predicate": "located_in", "object": "London"}
        ]
      }
      │
      ▼
   5. Store facts in ChromaDB:
      - Generate embeddings for 3 facts
      - Insert with metadata (user_id, timestamp, importance)
      │
      ▼
   6. Store entities & relationships in NetworkX:
      - Add nodes: Alice, TechCorp, London
      - Add edges: Alice --works_at--> TechCorp
      - Add edges: Alice --located_in--> London
      │
      ▼
   7. Persist graph to disk (knowledge_graph.pkl)
```

### Deep Search Flow (Graph Traversal)

```
1. USER: "Tell me everything about Alice and her relationships"
   │
   ▼
2. LLM calls: search_memory(query="...", tier="deep")
   │
   ▼
3. SearchService.search() with tier="deep":
   │
   ├─► Vector search (top_k=10):
   │   ├─► "User's name is Alice" (0.92)
   │   ├─► "Alice works at TechCorp" (0.88)
   │   └─► "Alice located in London" (0.85)
   │
   └─► Graph traversal:
       ├─► Extract entities from query: ["Alice"]
       ├─► Find "Alice" node in graph
       └─► 2-hop traversal:
           ├─► Direct neighbors (1-hop):
           │   ├─► Alice --works_at--> TechCorp
           │   └─► Alice --located_in--> London
           └─► Secondary neighbors (2-hop):
               └─► TechCorp --located_in--> London
   │
   ▼
4. Combine results:
   "Relevant memories from vector search:
   - User's name is Alice (Similarity: 0.92)
   - Alice works at TechCorp (Similarity: 0.88)
   
   Related knowledge from graph:
   - (Person) Alice -- works_at --> (Organization) TechCorp
   - (Person) Alice -- located_in --> (Location) London
   - (Organization) TechCorp -- located_in --> (Location) London"
   │
   ▼
5. LLM receives enriched context and generates comprehensive answer
```

---

## Memory Lifecycle

### 1. Creation Phase
```
User Message → Salience Check → [PASS] → Knowledge Extraction → Storage
                               ↓ [FAIL]
                               Skip
```

**Salience Factors:**
- Semantic similarity to salient/non-salient prototypes
- Keyword presence (lightweight mode)
- Custom salience configuration (if provided)

### 2. Active Phase
- **Status:** "active"
- **Searchable:** Yes
- **Access tracking:** Increment `access_count`, update `last_accessed_timestamp`
- **Relevance:** High (boosted by recency and attention)

### 3. Decay Phase
- **Trigger:** Curation service runs periodically
- **Condition:** `relevance_score < 0.3`
- **Action:** Status → "archived"
- **Searchable:** No (filtered out from search results)

### 4. Expiration Phase
- **Trigger:** Curation service checks `expiration_timestamp`
- **Condition:** `expiration_timestamp < current_time`
- **Action:** Delete from both vector and graph storage
- **Example:** "Meeting tomorrow at 3pm" → expires after meeting time

---

## Operation Modes

### Mode Comparison Matrix

| Feature | LOCAL | ONLINE | LIGHTWEIGHT |
|---------|-------|--------|-------------|
| **Embedding Model** | sentence-transformers | OpenAI API | None |
| **Salience Method** | ML (local model) | ML (API) | Keywords |
| **Vector Storage** | ✅ ChromaDB | ✅ ChromaDB | ❌ None |
| **Graph Storage** | ✅ NetworkX | ✅ NetworkX | ✅ NetworkX |
| **Startup Time** | ~10s (model load) | ~2s | <1s |
| **Semantic Search** | ✅ Yes | ✅ Yes | ❌ No |
| **Graph Search** | ✅ Yes | ✅ Yes | ✅ Yes |
| **API Costs** | $0 | ~$0.0001/op | $0 |
| **Offline Support** | ✅ 100% | ❌ No | ✅ 100% |
| **Best For** | Production, privacy | Serverless, fast start | Prototyping, testing |

### Architecture by Mode

#### LOCAL Mode (Default for Ollama)
```
┌─────────────────────────────────────────┐
│  sentence-transformers/paraphrase-MiniLM│ (Loaded locally)
│  - Salience classification              │
│  - Embedding generation                 │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  ChromaDB (Vector) + NetworkX (Graph)   │
└─────────────────────────────────────────┘
```

**Pros:** No external dependencies, privacy-preserving, no API costs
**Cons:** Slower startup (~10s for model loading)

#### ONLINE Mode (Default for OpenAI/Claude/Gemini)
```
┌─────────────────────────────────────────┐
│  OpenAI Embeddings API (text-embedding-3)│ (Cloud API)
│  - Salience classification              │
│  - Embedding generation                 │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  ChromaDB (Vector) + NetworkX (Graph)   │
└─────────────────────────────────────────┘
```

**Pros:** Fast startup (~2s), serverless-friendly, high accuracy
**Cons:** Small API cost (~$0.0001 per operation), requires internet

#### LIGHTWEIGHT Mode
```
┌─────────────────────────────────────────┐
│  Keyword-based Salience                 │ (No embeddings)
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  NetworkX (Graph only, NO vector store) │
└─────────────────────────────────────────┘
```

**Pros:** Instant startup (<1s), zero API costs, minimal dependencies
**Cons:** No semantic search (graph-only), lower accuracy for salience

---

## Search Tiers

### Tier Selection Logic

**LLM automatically chooses tier based on query complexity:**

```python
# Fast tier (auto-selected for simple queries)
"What's my name?" → fast tier (2 vector results, 50-100ms)

# Balanced tier (default for most queries)
"What projects am I working on?" → balanced tier (5 vector results, 200-500ms)

# Deep tier (auto-selected or explicitly requested)
"Tell me everything about Alice and her team" → deep tier (10 vector + graph, 800-2000ms)
```

### Performance Characteristics

| Tier | Vector Top-K | Graph Depth | Avg Latency | Use Cases |
|------|--------------|-------------|-------------|-----------|
| **Fast** | 2 | 0 | 50-150ms | Simple fact recall, real-time chat UI |
| **Balanced** | 5 | 0 | 200-600ms | General Q&A, most conversational queries |
| **Deep** | 10 | 2-hop | 800-2500ms | Research, "tell me everything", multi-entity queries |

---

## Storage Systems

### Vector Storage (ChromaDB)

**Purpose:** Semantic similarity search over facts

**Data Model:**
```json
{
  "id": "mem_abc123",
  "content": "Alice works at TechCorp",
  "embedding": [0.123, -0.456, ...],  // 1536-dim for OpenAI
  "metadata": {
    "user_id": "user_123",
    "timestamp": 1234567890,
    "status": "active",  // or "archived"
    "access_count": 5,
    "last_accessed_timestamp": 1234567890,
    "importance_score": 0.8,
    "expiration_timestamp": 1234999999  // optional
  }
}
```

**Index:** HNSW (Hierarchical Navigable Small World) for fast approximate nearest neighbor search

**Distance Metric:** Cosine similarity

### Graph Storage (NetworkX)

**Purpose:** Entity relationship queries and graph traversal

**Persistence:** Pickled to `knowledge_graph.pkl` after every modification

**Node Types:**
- **Person** - e.g., "Alice", "Dr. Watson"
- **Organization** - e.g., "TechCorp", "Stanford University"
- **Concept** - e.g., "Project Phoenix", "Machine Learning"
- **Task** - Scheduled reminders with `due_timestamp`
- **Location** - e.g., "London", "San Francisco"

**Entity Deduplication:**
```python
# Canonical entity matching prevents duplicates
"Dr. Watson" → merged with → "Dr. Emma Watson" (fuzzy match, threshold=0.85)
"John" → merged with → "John Smith" (substring match)
```

**Graph Operations:**
```python
# Add entity
graph.add_node("Alice", type="Person", created_timestamp=now)

# Add relationship
graph.add_edge("Alice", "TechCorp", predicate="works_at")

# Subgraph traversal (2-hop)
neighbors = nx.ego_graph(graph, "Alice", radius=2)
# Returns: Alice, TechCorp, London, [all nodes within 2 edges]
```

---

## LLM Provider Integration

### Common Wrapper Interface

All providers implement the same base interface:

```python
class BaseLLMWrapper:
    def __init__(
        self,
        storage_path="./memlayer_data",
        user_id="default_user",
        operation_mode="online",  # or "local" or "lightweight"
        salience_threshold=0.0,
        **kwargs
    ):
        # Initialize storage
        self.vector_storage = ChromaStorage(...)
        self.graph_storage = NetworkXStorage(...)
        
        # Initialize services
        self.search_service = SearchService(...)
        self.consolidation_service = ConsolidationService(...)
        self.scheduler_service = SchedulerService(...)
        self.curation_service = CurationService(...)
    
    def chat(self, messages, stream=False, **kwargs):
        # 1. Check for due tasks
        task_context = self.search_service.get_triggered_tasks_context(user_id)
        if task_context:
            messages = inject_system_message(messages, task_context)
        
        # 2. Call LLM with memory tools
        response = self.client.chat(
            messages=messages,
            tools=[search_memory_tool, schedule_task_tool],
            **kwargs
        )
        
        # 3. Handle tool calls (memory search)
        while response has tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.name == "search_memory":
                    result = self.search_service.search(
                        query=tool_call.args["query"],
                        search_tier=tool_call.args["search_tier"],
                        user_id=user_id
                    )
                    messages.append(tool_result)
                    
                elif tool_call.name == "schedule_task":
                    self.graph_storage.create_task(
                        description=tool_call.args["task_description"],
                        due_date=tool_call.args["due_date"],
                        user_id=user_id
                    )
            
            response = self.client.chat(messages=messages, ...)
        
        # 4. Background consolidation (async)
        full_conversation = format_conversation(messages, response)
        self.consolidation_service.consolidate(full_conversation, user_id)
        
        return response
```

### Provider-Specific Adaptations

#### OpenAI
```python
from memlayer import OpenAI

client = OpenAI(
    api_key="sk-...",
    model="gpt-4.1-mini",
    storage_path="./memories",
    user_id="user_123",
    operation_mode="online"  # Uses OpenAI embeddings
)
```

#### Claude (Anthropic)
```python
from memlayer import Claude

client = Claude(
    api_key="sk-ant-...",
    model="claude-4-sonnet",
    storage_path="./memories",
    user_id="user_123",
    operation_mode="online"  # Uses OpenAI embeddings for salience
)
```

#### Ollama (Local)
```python
from memlayer import Ollama

client = Ollama(
    host="http://localhost:11434",
    model="qwen3:14b",
    storage_path="./memories",
    user_id="user_123",
    operation_mode="local"  # Uses local sentence-transformers
)
```

---

## Advanced Features

### 1. Proactive Task Reminders

**User Request:**
```
User: "Remind me to submit the report next Friday at 9am"
```

**System Action:**
```python
# LLM calls schedule_task tool
graph_storage.create_task(
    task_id="task_xyz",
    description="Submit the report",
    due_timestamp=parse_date("next Friday at 9am").timestamp(),
    user_id="user_123",
    status="pending"
)
```

**Reminder Trigger:**
```python
# SchedulerService checks every 60 seconds
if current_time >= task["due_timestamp"]:
    graph_storage.update_task_status("task_xyz", "triggered")
```

**Next Chat:**
```
User: "What should I do today?"
# SearchService detects triggered task
# Injects: "🚨 REMINDER: Submit the report (due today at 9am)"
Assistant: "Don't forget to submit the report - it's due today at 9am!"
```

### 2. Observability & Tracing

**Every search operation returns detailed telemetry:**

```python
response = client.chat(messages)

# Inspect last search trace
trace = client.last_trace
print(f"Total duration: {trace.total_duration_ms}ms")

for event in trace.events:
    print(f"{event.event_type}: {event.duration_ms}ms")
    if event.metadata:
        print(f"  Metadata: {event.metadata}")

# Example output:
# embedding_generation: 15ms
#   Metadata: {'cache_status': 'hit'}
# vector_search: 45ms
#   Metadata: {'tier': 'balanced', 'top_k': 5, 'results_found': 3}
# graph_search: 120ms
#   Metadata: {'extracted_entities': ['Alice'], 'relationships_found': 5}
# result_formatting: 2ms
# Total duration: 182ms
```

### 3. Custom Salience Configuration

**Per-Tenant Salience Rules:**

```python
from memlayer import TenantSalienceConfig, OpenAI

# Define custom salience rules
config = TenantSalienceConfig(
    tenant_id="enterprise_corp",
    default_threshold=0.6,  # Stricter filtering
    custom_rules=[
        {
            "pattern": r"budget|financial|revenue",
            "threshold": 0.3,  # Always save financial info
            "reason": "Financial information is critical"
        }
    ],
    component_weights={
        "semantic_similarity": 0.5,
        "keyword_match": 0.3,
        "importance_score": 0.2
    }
)

client = OpenAI(
    salience_config=config,
    tenant_id="enterprise_corp"
)
```

### 4. Direct Knowledge Ingestion

**Bulk import from documents:**

```python
# Import knowledge from text
client.update_from_text("""
Project Phoenix is led by Alice.
The project uses Python and FastAPI.
The deadline is December 1st, 2024.
""")

# Immediately searchable
response = client.chat([
    {"role": "user", "content": "Who leads Project Phoenix?"}
])
# "Alice leads Project Phoenix."
```

### 5. Memory Browser (UI)

**Web interface for inspecting memories:**

```python
from memlayer import MemoryBrowser

browser = MemoryBrowser(storage_path="./memories")
browser.start(port=8080)
# Open http://localhost:8080 to browse memories
```

**Features:**
- View all stored memories
- Search by query
- Visualize knowledge graph
- Inspect entity relationships
- Track memory access patterns

---

## Configuration Guide

### Essential Parameters

```python
from memlayer import OpenAI

client = OpenAI(
    # ===== LLM Settings =====
    api_key="sk-...",              # Optional, defaults to OPENAI_API_KEY env var
    model="gpt-4.1-mini",          # Model name
    temperature=0.7,               # Sampling temperature
    
    # ===== Storage Settings =====
    storage_path="./memlayer_data",  # Where to store memories
    user_id="user_123",              # Unique user identifier
    
    # ===== Memory Behavior =====
    operation_mode="online",         # "local" | "online" | "lightweight"
    salience_threshold=0.0,          # -0.1 to 0.2 (-0.1=permissive, 0.2=strict)
    
    # ===== Performance Tuning =====
    scheduler_interval_seconds=60,   # Task reminder check frequency
    curation_interval_seconds=3600,  # Memory curation frequency (1 hour)
    
    # ===== Advanced =====
    embedding_model=None,            # Custom embedding model (optional)
    salience_config=None,            # Custom salience rules (optional)
    tenant_id=None                   # Multi-tenancy support (optional)
)
```

### Salience Threshold Guide

| Threshold | Behavior | Storage Rate | Best For |
|-----------|----------|--------------|----------|
| **-0.1** | Very permissive | ~80% saved | Comprehensive memory, storage not a concern |
| **0.0** | Balanced (default) | ~50% saved | General use, good accuracy/storage tradeoff |
| **0.1** | Moderately strict | ~30% saved | Cost-conscious, quality over quantity |
| **0.2** | Very strict | ~10% saved | Minimal storage, only critical information |

### Performance Tuning

**For low-latency applications:**
```python
client = OpenAI(
    operation_mode="online",  # Fast startup
    salience_threshold=0.1,   # Less frequent consolidation
    curation_interval_seconds=7200  # Less frequent curation
)
```

**For privacy-sensitive applications:**
```python
client = Ollama(
    operation_mode="local",   # 100% offline
    model="qwen3:14b",
    salience_threshold=0.0
)
```

**For cost-optimization:**
```python
client = OpenAI(
    operation_mode="lightweight",  # No embeddings
    salience_threshold=0.15       # Strict filtering
)
# Note: No semantic search, graph-only retrieval
```

---

## Summary

**Memlayer** is a sophisticated yet easy-to-use memory layer that:

1. **Wraps any LLM** with automatic memory capabilities
2. **Filters intelligently** using ML-based salience detection
3. **Stores hybrid** in both vector (semantic) and graph (relational) databases
4. **Searches efficiently** with 3-tier latency options
5. **Runs asynchronously** - consolidation doesn't block responses
6. **Manages lifecycle** - automatic expiration and decay
7. **Provides observability** - detailed tracing for every operation

**Architecture Highlights:**
- **Modular design** - swap storage backends, embedding models, LLM providers
- **Production-ready** - background services, error handling, persistence
- **Developer-friendly** - 3 lines of code to add memory
- **Flexible** - 3 operation modes for different use cases

This makes Memlayer suitable for:
- ✅ Conversational AI with context retention
- ✅ Personal assistant applications
- ✅ Customer support bots with memory
- ✅ Knowledge management systems
- ✅ Research assistants with persistent knowledge graphs

---

**For more details, see:**
- [README.md](README.md) - Getting started
- [docs/basics/overview.md](docs/basics/overview.md) - Basic concepts
- [docs/tuning/operation_mode.md](docs/tuning/operation_mode.md) - Mode selection guide
- [examples/](examples/) - Code examples
