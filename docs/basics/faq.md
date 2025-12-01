# Frequently Asked Questions

This document answers common questions about Memlayer's architecture, storage mechanisms, deployment options, and customization capabilities.

## Table of Contents

- [How Does MemoryBrowser Work?](#how-does-memorybrowser-work)
- [Storage Architecture](#storage-architecture)
- [Operation Modes and Storage Allocation](#operation-modes-and-storage-allocation)
- [Deployment Options](#deployment-options)
- [Custom Salience Configuration](#custom-salience-configuration)

---

## How Does MemoryBrowser Work?

### Overview

The `MemoryBrowser` class provides a built-in tool for viewing, searching, and managing memories stored in Memlayer. It works with the **vector storage backend** (ChromaDB, Qdrant, or Zilliz), not just graphs.

### What Does MemoryBrowser Access?

```
MemoryBrowser
     │
     └──► Vector Storage (ChromaDB/Qdrant/Zilliz)
              │
              └── Contains: Facts, content, embeddings, metadata
                   - user_id
                   - timestamp
                   - status (active/archived)
                   - access_count
                   - importance_score
                   - expiration_timestamp
```

### Available Methods

```python
from memlayer import MemoryBrowser

# Initialize with vector storage
browser = MemoryBrowser(client.storage)

# List all memories
browser.show_all()                         # Display all memories with metadata
browser.show_all(user_id="alice")          # Filter by user
browser.show_all(max_items=50)             # Limit results

# Search memories
browser.search("project")                  # Keyword search in content
browser.search("alice", user_id="user123") # Search with user filter

# Statistics
browser.print_stats()                      # Memory statistics
browser.print_stats(user_id="alice")       # User-specific stats

# Filter by importance
high_importance = browser.filter_by_importance(min_score=0.8)

# Filter by status
active_memories = browser.filter_by_status(status="active")

# Export data
browser.export_json("backup.json")         # Export to JSON file

# Manage memories
browser.delete_by_id("mem_abc123")         # Permanently delete
browser.archive_by_id("mem_abc123")        # Change status to archived
```

### Note About Graph Storage

The `MemoryBrowser` primarily works with **vector storage** (facts/memories with embeddings). The **graph storage** (NetworkX) stores entities and relationships separately and is accessed through the graph storage interface directly:

```python
# Access graph storage for entities and relationships
graph = client.graph_storage
relationships = graph.get_related_concepts("Alice")
subgraph = graph.get_subgraph_context("Project Phoenix", depth=2)
```

---

## Storage Architecture

### Is It Vector DB or Graphs?

**Answer: It's BOTH** (in LOCAL and ONLINE modes).

Memlayer uses a **hybrid storage architecture**:

```
┌──────────────────────────────────────────────────────────────┐
│                    Memlayer Storage                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────┐   ┌─────────────────────────┐  │
│  │   Vector Storage        │   │   Graph Storage         │  │
│  │   (ChromaDB)            │   │   (NetworkX)            │  │
│  │                         │   │                         │  │
│  │   Stores:               │   │   Stores:               │  │
│  │   • Facts (text)        │   │   • Entities (nodes)    │  │
│  │   • Embeddings          │   │   • Relationships       │  │
│  │   • Metadata            │   │   • Entity types        │  │
│  │   • Importance scores   │   │   • Task reminders      │  │
│  │                         │   │                         │  │
│  │   Used for:             │   │   Used for:             │  │
│  │   • Semantic search     │   │   • Graph traversal     │  │
│  │   • Fast/Balanced tiers │   │   • Deep search tier    │  │
│  │   • Similarity matching │   │   • Relationship queries│  │
│  └─────────────────────────┘   └─────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### How Data Gets Stored

When you have a conversation:

```python
client.chat([{"role": "user", "content": "Alice works at TechCorp on Project Phoenix"}])
```

Memlayer extracts and stores:

**In Vector Storage (ChromaDB):**
```json
{
  "content": "Alice works at TechCorp",
  "embedding": [0.123, 0.456, ...],  // 384-1536 dimensions
  "metadata": {
    "user_id": "user123",
    "importance_score": 0.7,
    "timestamp": 1701234567
  }
}
```

**In Graph Storage (NetworkX):**
```
Nodes:
  - "Alice" (type: Person)
  - "TechCorp" (type: Organization)
  - "Project Phoenix" (type: Project)

Edges:
  - Alice --[works_at]--> TechCorp
  - Alice --[works_on]--> Project Phoenix
```

### Search Flow

| Search Tier | Vector Search | Graph Traversal | Best For |
|-------------|---------------|-----------------|----------|
| **fast** | ✅ 2 results | ❌ | Simple lookups, "What's my name?" |
| **balanced** | ✅ 5 results | ❌ | General queries |
| **deep** | ✅ 10 results | ✅ 2-hop traversal | Complex queries, "Tell me everything about..." |

---

## Operation Modes and Storage Allocation

### How Modes Affect Storage

The `salience_mode` (or `operation_mode`) parameter fundamentally changes what storage backends are used:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Storage by Operation Mode                            │
├─────────────────┬─────────────────────┬─────────────────────────────────────┤
│ Mode            │ Vector Storage      │ Graph Storage                       │
├─────────────────┼─────────────────────┼─────────────────────────────────────┤
│ LOCAL           │ ✅ ChromaDB         │ ✅ NetworkX                         │
│                 │ (local embeddings)  │ (entities + relationships)          │
├─────────────────┼─────────────────────┼─────────────────────────────────────┤
│ ONLINE          │ ✅ ChromaDB         │ ✅ NetworkX                         │
│                 │ (API embeddings)    │ (entities + relationships)          │
├─────────────────┼─────────────────────┼─────────────────────────────────────┤
│ LIGHTWEIGHT     │ ❌ DISABLED         │ ✅ NetworkX only                    │
│                 │ (no vector search)  │ (entities + relationships)          │
└─────────────────┴─────────────────────┴─────────────────────────────────────┘
```

### Code Example: Mode Selection

```python
# LOCAL mode: Full hybrid storage (vector + graph)
# - Embedding model loaded locally (~500MB RAM)
# - No API costs for embeddings
# - Startup: ~10 seconds (model loading)
client = OpenAI(
    model="gpt-4.1-mini",
    salience_mode="local",
    storage_path="./memories"
)

# ONLINE mode: Full hybrid storage (vector + graph)
# - Embeddings via OpenAI API
# - Cost: ~$0.0001 per operation
# - Startup: ~2 seconds (fast!)
client = OpenAI(
    model="gpt-4.1-mini",
    salience_mode="online",
    storage_path="./memories"
)

# LIGHTWEIGHT mode: Graph-only storage
# - NO vector embeddings
# - NO semantic search (keyword-based only)
# - Startup: <1 second (instant)
client = OpenAI(
    model="gpt-4.1-mini",
    salience_mode="lightweight",
    storage_path="./memories"
)
```

### Which Mode Should I Use?

| Use Case | Recommended Mode | Why |
|----------|------------------|-----|
| **Production API** | ONLINE | Fast startup, reliable semantic search |
| **Privacy-sensitive** | LOCAL | No data sent to embedding APIs |
| **Offline/air-gapped** | LOCAL | Works without internet |
| **Prototyping/demos** | LIGHTWEIGHT | Instant startup, zero dependencies |
| **Cost-conscious (high volume)** | LOCAL | Zero API costs after model load |
| **Serverless (Lambda)** | ONLINE | Fast cold starts |

---

## Deployment Options

### Can I Host Memlayer on EC2 or Lambda?

**Yes!** Memlayer is designed for production deployments including AWS EC2, Lambda, and other cloud platforms.

### EC2 Deployment

EC2 is ideal for long-running servers where you can absorb the startup cost once.

```python
# Recommended: LOCAL mode on EC2 (no ongoing API costs)
from memlayer import OpenAI

client = OpenAI(
    model="gpt-4.1-mini",
    salience_mode="local",         # Load model once, use forever
    storage_path="/data/memories",  # Persistent EBS volume
    user_id="service_user"
)

# For FastAPI/Flask server:
# Initialize client once at startup, reuse for all requests
```

**EC2 Best Practices:**
- Use an instance with 4GB+ RAM for LOCAL mode
- Store data on EBS for persistence
- Consider GPU instances (g4dn.xlarge) for faster local inference
- LOCAL mode recommended for cost efficiency

### Lambda Deployment

Lambda requires fast cold starts. Use ONLINE or LIGHTWEIGHT mode.

```python
# Recommended: ONLINE mode for Lambda (fast cold starts)
from memlayer import OpenAI

def handler(event, context):
    # Client initialization is fast (~200ms with ONLINE mode)
    client = OpenAI(
        model="gpt-4.1-mini",
        salience_mode="online",     # Fast startup!
        storage_path="/tmp/memories",
        user_id=event["user_id"]
    )
    
    response = client.chat([
        {"role": "user", "content": event["message"]}
    ])
    
    return {"response": response}
```

**Lambda Considerations:**
- Use ONLINE mode for cold start performance
- Lambda has 512MB /tmp storage - enough for most use cases
- For persistent storage, use external ChromaDB or Qdrant Cloud
- LIGHTWEIGHT mode works but lacks semantic search

### Can I Offer "Memlayer as a Service"?

**Yes!** You can build a multi-tenant API service using Memlayer. Here's a basic architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                   Memlayer as a Service                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐         ┌──────────────────────────────┐    │
│   │   API Layer  │         │     Storage Layer            │    │
│   │   (FastAPI)  │         │                              │    │
│   │              │────────►│  ChromaDB / Qdrant Cloud     │    │
│   │  /chat       │         │  (vector storage)            │    │
│   │  /memories   │         │                              │    │
│   │  /search     │────────►│  NetworkX / Memgraph         │    │
│   │              │         │  (graph storage)             │    │
│   └──────────────┘         └──────────────────────────────┘    │
│          │                                                      │
│          │                                                      │
│   ┌──────▼──────────────────────────────────────────────┐      │
│   │              Multi-Tenant User Management            │      │
│   │                                                      │      │
│   │   user_id="tenant_001"  ──►  Isolated memories      │      │
│   │   user_id="tenant_002"  ──►  Isolated memories      │      │
│   │   user_id="tenant_003"  ──►  Isolated memories      │      │
│   └──────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Example FastAPI Service:**

```python
from fastapi import FastAPI, HTTPException
from memlayer import OpenAI

app = FastAPI()

# Singleton client (or use dependency injection)
clients = {}

def get_client(user_id: str):
    if user_id not in clients:
        clients[user_id] = OpenAI(
            model="gpt-4.1-mini",
            salience_mode="online",
            storage_path=f"./data/{user_id}",
            user_id=user_id
        )
    return clients[user_id]

@app.post("/chat")
async def chat(user_id: str, message: str):
    client = get_client(user_id)
    response = client.chat([{"role": "user", "content": message}])
    return {"response": response}

@app.get("/memories")
async def get_memories(user_id: str):
    client = get_client(user_id)
    browser = MemoryBrowser(client.storage)
    return browser.get_all_memories(user_id=user_id)
```

---

## Custom Salience Configuration

### What is Salience?

Salience determines **what information is worth saving**. Not everything said in a conversation is valuable:

- ✅ **Salient**: "My name is Alice and I work at TechCorp"
- ❌ **Non-salient**: "Hi, how are you?", "Thanks!", "Okay"

### Can I Customize Salience?

**Yes!** There are several ways to customize salience behavior:

### 1. Adjust the Salience Threshold

```python
# More permissive: saves more content (lower threshold)
client = OpenAI(
    salience_threshold=-0.1,  # Saves most content except obvious filler
    salience_mode="online"
)

# Balanced: default behavior
client = OpenAI(
    salience_threshold=0.0,   # Default
    salience_mode="online"
)

# Strict: only saves clear facts and preferences
client = OpenAI(
    salience_threshold=0.1,   # Stricter filtering
    salience_mode="online"
)
```

**Threshold Guidelines:**

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| `-0.1` | Very permissive | Personal assistants, maximum recall |
| `0.0` | Balanced (default) | General purpose |
| `0.1` | Strict | Privacy-sensitive, minimize storage |
| `0.2+` | Very strict | Only critical facts |

### 2. Use Different Salience Modes

Each mode uses a different salience detection method:

```python
# LOCAL mode: Uses sentence-transformers for semantic similarity
# - Most accurate, but slow startup
# - Compares input against semantic prototypes
client = OpenAI(salience_mode="local")

# ONLINE mode: Uses OpenAI embeddings API
# - Fast startup, accurate
# - Small API cost per check (~$0.0001)
client = OpenAI(salience_mode="online")

# LIGHTWEIGHT mode: Uses keyword matching
# - Instant, no dependencies
# - Less accurate (rule-based)
client = OpenAI(salience_mode="lightweight")
```

### 3. Extend Salience Prototypes (Advanced)

The salience gate compares text against prototype examples. You can extend these:

```python
from memlayer.ml_gate import (
    SALIENT_PROTOTYPES, 
    NON_SALIENT_PROTOTYPES,
    SALIENT_KEYWORDS,
    NON_SALIENT_KEYWORDS
)

# Add your own salient examples
SALIENT_PROTOTYPES.extend([
    "The quarterly revenue target is $5 million",
    "Customer complaint about delivery delays",
    "Product SKU ABC-123 is out of stock"
])

# Add your own non-salient examples
NON_SALIENT_PROTOTYPES.extend([
    "Let me check on that",
    "I'll get back to you",
    "One moment please"
])

# For LIGHTWEIGHT mode: Add keywords
SALIENT_KEYWORDS.extend(['sku', 'revenue', 'complaint', 'delivery'])
NON_SALIENT_KEYWORDS.extend(['moment', 'check', 'back'])
```

### 4. Create Custom Salience Gate (Advanced)

For full control, implement a custom salience gate:

```python
from memlayer.ml_gate import SalienceGate, SalienceMode

class CustomSalienceGate(SalienceGate):
    """Custom salience gate with domain-specific logic."""
    
    def is_worth_saving(self, text: str, verbose: bool = False) -> bool:
        # Custom rules first
        if self._is_financial_data(text):
            return True  # Always save financial info
        
        if self._is_customer_name(text):
            return True  # Always save customer names
        
        # Fall back to default behavior
        return super().is_worth_saving(text, verbose)
    
    def _is_financial_data(self, text: str) -> bool:
        import re
        return bool(re.search(r'\$[\d,]+|\d+%|revenue|profit|loss', text, re.I))
    
    def _is_customer_name(self, text: str) -> bool:
        return 'customer' in text.lower() and any(
            word[0].isupper() for word in text.split()
        )
```

### Salience Gate Flow

```
Input Text: "My name is Alice"
          │
          ▼
   ┌─────────────────────┐
   │ Quick Heuristic     │
   │ (Regex patterns)    │
   └─────────────────────┘
          │
   Match? ├── Yes (Salient pattern) ──► SAVE ✅
          │
          ├── Yes (Non-salient) ──► SKIP ❌
          │
          └── No (Uncertain) ──►
                    │
                    ▼
          ┌─────────────────────────┐
          │ Semantic Check          │
          │ (Mode-dependent)        │
          │                         │
          │ LOCAL: sentence-trans   │
          │ ONLINE: OpenAI API      │
          │ LIGHTWEIGHT: keywords   │
          └─────────────────────────┘
                    │
                    ▼
   salient_score > (non_salient_score + threshold)?
          │
          ├── Yes ──► SAVE ✅
          │
          └── No  ──► SKIP ❌
```

---

## Summary

| Topic | Key Points |
|-------|------------|
| **MemoryBrowser** | Works with vector storage (ChromaDB); use graph storage directly for entities/relationships |
| **Storage** | Hybrid: Vector DB (facts/embeddings) + Graph (entities/relationships) |
| **Modes** | LOCAL/ONLINE = full storage; LIGHTWEIGHT = graph-only |
| **Deployment** | EC2 (LOCAL mode) or Lambda (ONLINE mode) both work |
| **Salience** | Customize via threshold, mode, prototypes, or custom gate class |

---

## Related Documentation

- [Operation Modes](../tuning/operation_mode.md) - Deep dive on modes
- [Salience Threshold](../tuning/salience_threshold.md) - Threshold tuning
- [ChromaDB Storage](../storage/chroma.md) - Vector storage details
- [NetworkX Storage](../storage/networkx.md) - Graph storage details
- [API Reference](../API_REFERENCE.md) - Complete API documentation
