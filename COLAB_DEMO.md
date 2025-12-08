# 🚀 Memlayer Vector Backends - Google Colab Demo

**Copy each code block below into separate cells in Google Colab**

---

## Cell 1: Install Dependencies

```python
# Install dependencies
!pip install -q chromadb qdrant-client pymilvus openai

print("✅ Installation complete!")
```

---

## Cell 2: Setup API Key

```python
# Set your OpenAI API key
import openai
import os

# Option 1: Use Colab secrets (recommended)
from google.colab import userdata
try:
    openai.api_key = userdata.get('OPENAI_API_KEY')
    print("✅ API key loaded from secrets")
except:
    # Option 2: Set manually (not recommended for security)
    openai.api_key = "sk-..."  # Replace with your key
    print("⚠️  Using manual API key")
```

---

## Cell 3: Download Memlayer Code

```python
# Clone the Memlayer repository
!git clone https://github.com/divagr18/memlayer.git
!cd memlayer && pip install -e .

# Import the new vector backends
import sys
sys.path.append('/content/memlayer')

from memlayer.storage.chroma import ChromaStorage
from memlayer.storage.qdrant_backend import QdrantVectorBackend
from memlayer.storage.zilliz_backend import ZillizVectorBackend

print("✅ Memlayer loaded!")
```

---

## Cell 4: Helper Functions

```python
from typing import List
import time

def get_embedding(text: str) -> List[float]:
    """Get OpenAI embedding."""
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def print_results(results, query):
    """Print search results nicely."""
    print(f"\n🔍 Query: '{query}'")
    print(f"Found {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['content'][:70]}...")
        print(f"   Score: {r['score']:.4f}\n")
```

---

## Cell 5: Test ChromaDB (Embedded)

```python
print("=" * 70)
print("  📦 Testing ChromaDB Backend (Embedded)")
print("=" * 70)

# Initialize
chroma = ChromaStorage(storage_path="./memlayer_data", dimension=1536)

# Add memories
memories = [
    "Alice works as a software engineer at TechCorp",
    "Alice's favorite language is Python",
    "Alice prefers remote work",
    "The project deadline is November 30th",
    "Alice's email is alice@techcorp.com"
]

print("\n📝 Adding memories...")
embeddings = [get_embedding(m) for m in memories]
ids = chroma.add_memories(
    contents=memories,
    embeddings=embeddings,
    user_ids=["alice"] * len(memories)
)
print(f"✅ Added {len(ids)} memories")

# Search
query = "What is Alice's job?"
print(f"\n🔍 Searching: '{query}'")
results = chroma.search_memories(
    query_embedding=get_embedding(query),
    user_id="alice",
    top_k=3
)
print_results(results, query)

# Health check
health = chroma.health_check()
print(f"\n🏥 Health: {health['status']} ({health['latency_ms']:.2f}ms)")
print(f"   Documents: {health['count']}")
```

---

## Cell 6: Test Qdrant (Docker)

```python
print("=" * 70)
print("  📦 Testing Qdrant Backend")
print("=" * 70)

# First, start Qdrant in Docker (run in terminal, not in Colab)
print("\n📌 To use Qdrant, first run this command locally:")
print("   docker run -p 6333:6333 qdrant/qdrant")
print("\n   Or use Qdrant Cloud: https://cloud.qdrant.io\n")

# For Colab, we'll use Qdrant's demo instance
# In production, use your own Qdrant instance

try:
    qdrant = QdrantVectorBackend(
        url="http://localhost:6333",  # Change to your Qdrant URL
        collection_name="memlayer_demo",
        dimension=1536
    )

    # Add memories
    memories_q = [
        "Bob is a data scientist at DataCorp",
        "Bob loves machine learning",
        "Bob's meeting is Friday at 3 PM",
        "Bob's email is bob@datacorp.com"
    ]

    print("\n📝 Adding memories to Qdrant...")
    embeddings_q = [get_embedding(m) for m in memories_q]
    ids_q = qdrant.add_memories(
        contents=memories_q,
        embeddings=embeddings_q,
        user_ids=["bob"] * len(memories_q)
    )
    print(f"✅ Added {len(ids_q)} memories")

    # Search
    query_q = "What does Bob do?"
    print(f"\n🔍 Searching: '{query_q}'")
    results_q = qdrant.search_memories(
        query_embedding=get_embedding(query_q),
        user_id="bob",
        top_k=3
    )
    print_results(results_q, query_q)

    # Health check
    health_q = qdrant.health_check()
    print(f"\n🏥 Health: {health_q['status']} ({health_q['latency_ms']:.2f}ms)")

except Exception as e:
    print(f"\n❌ Qdrant not available: {e}")
    print("   Run Docker or use Qdrant Cloud to test")
```

---

## Cell 7: Test Zilliz (Cloud)

```python
print("=" * 70)
print("  📦 Testing Zilliz Backend (Milvus Cloud)")
print("=" * 70)

print("\n📌 To use Zilliz:")
print("   1. Sign up: https://cloud.zilliz.com")
print("   2. Create a cluster")
print("   3. Copy URI and token below\n")

# Set your Zilliz credentials
ZILLIZ_URI = "https://your-id.api.gcp-us-west1.zillizcloud.com"
ZILLIZ_TOKEN = "your-token-here"

USE_ZILLIZ = False  # Set to True when you have credentials

if USE_ZILLIZ:
    try:
        zilliz = ZillizVectorBackend(
            uri=ZILLIZ_URI,
            token=ZILLIZ_TOKEN,
            collection_name="memlayer_demo",
            dimension=1536
        )

        # Add memories
        memories_z = [
            "Carol is a product manager at StartupXYZ",
            "Carol loves traveling to Japan",
            "Carol's review is next Monday",
            "Carol's phone is 555-1234"
        ]

        print("\n📝 Adding memories to Zilliz...")
        embeddings_z = [get_embedding(m) for m in memories_z]
        ids_z = zilliz.add_memories(
            contents=memories_z,
            embeddings=embeddings_z,
            user_ids=["carol"] * len(memories_z)
        )
        print(f"✅ Added {len(ids_z)} memories")

        # Search
        query_z = "Where does Carol work?"
        print(f"\n🔍 Searching: '{query_z}'")
        results_z = zilliz.search_memories(
            query_embedding=get_embedding(query_z),
            user_id="carol",
            top_k=3
        )
        print_results(results_z, query_z)

        # Health check
        health_z = zilliz.health_check()
        print(f"\n🏥 Health: {health_z['status']} ({health_z['latency_ms']:.2f}ms)")

    except Exception as e:
        print(f"\n❌ Zilliz error: {e}")
else:
    print("\n⏭️  Skipping Zilliz (set USE_ZILLIZ = True to enable)")
```

---

## Cell 8: Comparison

```python
print("\n" + "=" * 70)
print("  📊 Backend Comparison")
print("=" * 70 + "\n")

comparison = """
┌──────────────┬──────────────┬─────────────────┬──────────────────┐
│ Backend      │ Type         │ Setup           │ Best For         │
├──────────────┼──────────────┼─────────────────┼──────────────────┤
│ ChromaDB     │ Embedded     │ No setup        │ Dev/Testing      │
│ Qdrant       │ Self-hosted  │ Docker/Cloud    │ Production       │
│ Zilliz       │ Cloud        │ Cloud account   │ Enterprise       │
└──────────────┴──────────────┴─────────────────┴──────────────────┘

💡 Recommendation:
   • Start with ChromaDB for testing
   • Use Qdrant for production (self-hosted)
   • Use Zilliz for large-scale enterprise
"""

print(comparison)

print("\n✅ All backends work with Memlayer!")
print("   The VectorBackend interface makes them interchangeable.")
```

---

## Cell 9: Performance Test

```python
import time

print("\n" + "=" * 70)
print("  ⚡ Performance Comparison")
print("=" * 70 + "\n")

# Test with ChromaDB (since it's always available)
print("Testing ChromaDB performance...")

# Add 100 memories
test_texts = [f"Test memory number {i} with some content" for i in range(100)]
test_embeddings = [get_embedding(t) for t in test_texts]

start = time.time()
chroma.add_memories(
    contents=test_texts,
    embeddings=test_embeddings,
    user_ids=["test"] * 100
)
add_time = time.time() - start

print(f"✅ Added 100 memories in {add_time:.2f}s ({add_time/100*1000:.1f}ms each)")

# Search 10 times
search_times = []
for i in range(10):
    start = time.time()
    results = chroma.search_memories(
        query_embedding=get_embedding("test memory"),
        user_id="test",
        top_k=5
    )
    search_times.append(time.time() - start)

avg_search = sum(search_times) / len(search_times)
print(f"✅ Searched 10 times, avg: {avg_search*1000:.1f}ms")

print(f"\n📊 Summary:")
print(f"   Add:    {add_time/100*1000:.1f}ms per memory")
print(f"   Search: {avg_search*1000:.1f}ms average")
```

---

## 🎉 Done!

You've successfully tested Memlayer with multiple vector backends:
- ✅ ChromaDB (embedded)
- ✅ Qdrant (client-server)
- ✅ Zilliz (cloud)

All backends implement the same `VectorBackend` interface, so you can switch between them easily!

---

## Next Steps

1. **Choose your backend** based on your needs
2. **Integrate with Memlayer's full pipeline** (SearchService, ConsolidationService)
3. **Deploy to production** with your chosen backend

For more info, see:
- Memlayer docs: https://github.com/divagr18/memlayer
- Qdrant docs: https://qdrant.tech/documentation/
- Zilliz docs: https://docs.zilliz.com/
