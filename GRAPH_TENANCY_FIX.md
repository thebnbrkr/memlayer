# Graph Multi-Tenancy Fix Guide

This guide explains how to add proper tenant isolation to the graph storage.

## Problem Summary

Currently, graph entities are global - all tenants share the same entity namespace.

**Example:**
- Alice (tenant1): "I work at Google"
- Bob (tenant2): "I work at Google"
- Result: Both use the same "Google" node ❌

**Solution:** Add `tenant_id` to all graph operations.

---

## Fix Implementation

### **Step 1: Update NetworkXStorage.__init__**

Add tenant_id parameter:

```python
# File: memlayer/storage/networkx.py

class NetworkXStorage(BaseGraphStorage):
    def __init__(self, storage_path: str, tenant_id: str = None):
        self.tenant_id = tenant_id  # NEW
        self.graph_path = Path(storage_path) / "knowledge_graph.pkl"
        self._lock = threading.Lock()
        self.graph: nx.DiGraph = self._load_graph()
```

### **Step 2: Update add_entity to Store tenant_id**

```python
# File: memlayer/storage/networkx.py, around line 208

def add_entity(self, name: str, node_type: str = "Concept", metadata: Dict = None) -> str:
    if not name or not name.strip():
        return name

    # NEW: Namespace entity by tenant
    canonical_name = self._find_canonical_entity(name, node_type)

    if not self.graph.has_node(canonical_name):
        base_attrs = {
            "type": node_type,
            "tenant_id": self.tenant_id,  # NEW: Add tenant info
            "status": "active",
            "access_count": 0,
            "created_timestamp": time.time(),
            "last_accessed_timestamp": time.time(),
            "importance_score": 0.5,
            "expiration_timestamp": None
        }
        if metadata:
            base_attrs.update(metadata)

        self.graph.add_node(canonical_name, **base_attrs)
        self._save_graph()

    return canonical_name
```

### **Step 3: Update _find_canonical_entity to Filter by tenant_id**

```python
# File: memlayer/storage/networkx.py, around line 74

def _find_canonical_entity(self, name: str, node_type: str = "Concept", similarity_threshold: float = 0.85) -> str:
    if not name or not name.strip():
        return name

    name_lower = name.lower().strip()

    # Get all existing nodes of the same type AND same tenant
    same_type_nodes = [
        n for n, data in self.graph.nodes(data=True)
        if data.get('type') == node_type
        and data.get('tenant_id') == self.tenant_id  # NEW: Filter by tenant!
    ]

    if not same_type_nodes:
        return name  # No existing entities of this type for this tenant

    # Rest of the matching logic stays the same...
    # (exact match, substring match, similarity match)
```

### **Step 4: Update find_matching_nodes to Filter by tenant_id**

```python
# File: memlayer/storage/networkx.py

def find_matching_nodes(self, name_query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    name_lower = name_query.lower()

    matches = []
    for node, data in self.graph.nodes(data=True):
        # NEW: Only search within current tenant's nodes
        if data.get('tenant_id') != self.tenant_id:
            continue

        if name_lower in node.lower():
            matches.append({
                "name": node,
                "type": data.get("type", "unknown"),
                "access_count": data.get("access_count", 0),
                "importance_score": data.get("importance_score", 0.0)
            })

    # Sort by importance and return top results
    matches.sort(key=lambda x: x["importance_score"], reverse=True)
    return matches[:max_results]
```

### **Step 5: Update get_entity_context to Filter Relationships**

```python
# File: memlayer/storage/networkx.py

def get_entity_context(self, entity_name: str, max_depth: int = 2) -> Dict[str, Any]:
    if not self.graph.has_node(entity_name):
        return {"entity": entity_name, "exists": False}

    entity_data = self.graph.nodes[entity_name]

    # NEW: Only return context if entity belongs to current tenant
    if entity_data.get('tenant_id') != self.tenant_id:
        return {"entity": entity_name, "exists": False}

    # Get relationships (only with other tenant entities)
    outgoing = []
    for _, target, edge_data in self.graph.out_edges(entity_name, data=True):
        target_data = self.graph.nodes[target]
        # NEW: Only include relationships to entities in same tenant
        if target_data.get('tenant_id') == self.tenant_id:
            outgoing.append({
                "predicate": edge_data.get("predicate", "relates_to"),
                "target": target,
                "target_type": target_data.get("type", "unknown")
            })

    incoming = []
    for source, _, edge_data in self.graph.in_edges(entity_name, data=True):
        source_data = self.graph.nodes[source]
        # NEW: Only include relationships from entities in same tenant
        if source_data.get('tenant_id') == self.tenant_id:
            incoming.append({
                "predicate": edge_data.get("predicate", "relates_to"),
                "source": source,
                "source_type": source_data.get("type", "unknown")
            })

    return {
        "entity": entity_name,
        "exists": True,
        "type": entity_data.get("type"),
        "tenant_id": entity_data.get("tenant_id"),  # NEW
        "outgoing_relationships": outgoing,
        "incoming_relationships": incoming,
        "access_count": entity_data.get("access_count", 0),
        "importance_score": entity_data.get("importance_score", 0.0)
    }
```

### **Step 6: Update ConsolidationService to Pass tenant_id**

```python
# File: memlayer/services/__init__.py

class ConsolidationService:
    def __init__(
        self,
        vector_storage: Optional[ChromaStorage],
        graph_storage: MemgraphStorage,
        embedding_model: Optional[BaseEmbeddingModel],
        salience_gate: SalienceGate,
        llm_client: BaseLLMWrapper,
        salience_config: Optional["TenantSalienceConfig"] = None,
        tenant_id: Optional[str] = None
    ):
        self.vector_storage = vector_storage
        self.graph_storage = graph_storage
        self.tenant_id = tenant_id

        # NEW: Set tenant_id on graph storage
        if hasattr(self.graph_storage, 'tenant_id'):
            self.graph_storage.tenant_id = tenant_id

        # ... rest of init ...
```

### **Step 7: Update OpenAI Wrapper to Pass tenant_id**

```python
# File: memlayer/wrappers/openai.py

def __init__(
    self,
    # ... existing parameters ...
    tenant_id: Optional[str] = None,
    **kwargs
):
    self.tenant_id = tenant_id or user_id

    # When creating graph storage
    from ..storage.networkx import NetworkXStorage
    self._graph_storage = NetworkXStorage(
        storage_path=storage_path,
        tenant_id=self.tenant_id  # NEW: Pass tenant_id
    )
```

---

## Testing the Fix

### **Test 1: Verify Isolation**

```python
# Alice's client
alice = OpenAI(user_id="alice", tenant_id="tenant1", ...)
alice.chat([{"role": "user", "content": "I work at Google"}])

# Bob's client
bob = OpenAI(user_id="bob", tenant_id="tenant2", ...)
bob.chat([{"role": "user", "content": "I work at Google"}])

# Check Alice's graph
alice_context = alice.graph_storage.get_entity_context("Google")
print(alice_context)  # Should show tenant_id: tenant1

# Check Bob's graph
bob_context = bob.graph_storage.get_entity_context("Google")
print(bob_context)  # Should show tenant_id: tenant2

# Bob shouldn't see Alice's entities
alice_graph = alice.graph_storage.graph
bob_graph = bob.graph_storage.graph

# Both use same physical graph file, but filtered by tenant_id
alice_entities = [n for n, d in alice_graph.nodes(data=True) if d.get('tenant_id') == 'tenant1']
bob_entities = [n for n, d in bob_graph.nodes(data=True) if d.get('tenant_id') == 'tenant2']

assert set(alice_entities).isdisjoint(set(bob_entities))  # No overlap!
```

### **Test 2: Cross-Tenant Query Protection**

```python
# Alice creates entity
alice.chat([{"role": "user", "content": "Alice is CEO of TechCorp"}])

# Bob tries to query Alice's entity (should fail)
bob_search = bob.graph_storage.find_matching_nodes("Alice")
assert len(bob_search) == 0  # Bob can't see Alice's entities
```

---

## Alternative Approaches

### **Option A: Namespace in Entity Names** (Quick Fix)

Instead of filtering, prefix entity names with tenant_id:

```python
def add_entity(self, name: str, node_type: str = "Concept", ...):
    # Namespace by tenant
    namespaced_name = f"{self.tenant_id}:{name}"

    if not self.graph.has_node(namespaced_name):
        self.graph.add_node(namespaced_name, type=node_type, ...)
```

**Pros:**
- Simple to implement (5 minutes)
- No filtering logic needed
- Works with existing code

**Cons:**
- Ugly entity names in graph ("tenant1:Google")
- Can't share knowledge across tenants
- Hard to migrate later

### **Option B: Separate Graph Files Per Tenant** (Cleanest)

Store each tenant's graph in a separate file:

```python
def __init__(self, storage_path: str, tenant_id: str = None):
    # Each tenant gets their own graph file
    filename = f"knowledge_graph_{tenant_id}.pkl"
    self.graph_path = Path(storage_path) / filename
```

**Pros:**
- Perfect isolation
- Simple to understand
- Easy to backup/restore per tenant
- Can delete tenant data easily

**Cons:**
- Can't share entities across tenants
- More disk space (one file per tenant)
- Migration more complex

### **Option C: Hybrid Approach** (Most Flexible)

Use Option B (separate files) but allow "shared" graph for common entities:

```python
def __init__(self, storage_path: str, tenant_id: str = None, shared_mode: bool = False):
    if shared_mode:
        # Common entities (countries, cities, etc.)
        self.graph_path = Path(storage_path) / "shared_graph.pkl"
    else:
        # Tenant-specific entities
        self.graph_path = Path(storage_path) / f"graph_{tenant_id}.pkl"
```

---

## Recommendation

For **self-hosting with friends** (your use case):
- Use **Option B** (separate files) - simplest and safest

For **production SaaS**:
- Use **main solution** (tenant_id filtering) - most flexible
- Or use **Option C** (hybrid) if you want shared knowledge base

---

## Estimated Time to Implement

- **Option A** (namespacing): 15 minutes
- **Option B** (separate files): 30 minutes
- **Main solution** (filtering): 2-3 hours
- **Option C** (hybrid): 4-5 hours

---

## Migration Strategy

If you already have data in the graph:

```python
# Script to migrate existing graph to multi-tenant
import pickle
import networkx as nx

# Load old graph
with open('knowledge_graph.pkl', 'rb') as f:
    old_graph = pickle.load(f)

# Split by tenant (if you have user_id in facts)
tenant_graphs = {}
for node, data in old_graph.nodes(data=True):
    # Infer tenant from connected facts or default to "unknown"
    tenant_id = data.get('created_by', 'unknown')

    if tenant_id not in tenant_graphs:
        tenant_graphs[tenant_id] = nx.DiGraph()

    # Copy node to tenant graph
    tenant_graphs[tenant_id].add_node(node, **data, tenant_id=tenant_id)

# Copy edges
for tenant_id, graph in tenant_graphs.items():
    for u, v, data in old_graph.edges(data=True):
        if graph.has_node(u) and graph.has_node(v):
            graph.add_edge(u, v, **data)

# Save tenant graphs
for tenant_id, graph in tenant_graphs.items():
    with open(f'knowledge_graph_{tenant_id}.pkl', 'wb') as f:
        pickle.dump(graph, f)
```

---

## Summary

1. **Problem**: All tenants share same entity namespace
2. **Solution**: Add tenant_id filtering to all graph operations
3. **Quick Fix**: Use separate graph files per tenant (30 mins)
4. **Proper Fix**: Add tenant_id to node metadata (2-3 hours)
5. **Testing**: Verify isolation with multi-tenant tests

Choose Option B (separate files) for quick deployment, or implement the main solution for production SaaS.
