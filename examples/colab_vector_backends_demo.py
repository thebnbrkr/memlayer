"""
# Memlayer Vector Backends Demo - Google Colab

This notebook demonstrates how to use Memlayer with different vector backends:
1. ChromaDB (embedded, default)
2. Qdrant (self-hosted or cloud)
3. Zilliz (Milvus cloud)

## Setup Instructions

Copy this entire file and paste it into a Google Colab notebook.
Run each cell sequentially.
"""

# =============================================================================
# CELL 1: Install Dependencies
# =============================================================================

print("📦 Installing dependencies...")

# Install Memlayer from local files (if testing locally)
# !pip install -e /path/to/memlayer

# Or install from GitHub (when published)
# !pip install git+https://github.com/yourusername/memlayer.git

# Install vector database clients
get_ipython().system('pip install chromadb>=0.4.0')
get_ipython().system('pip install qdrant-client>=1.7.0')
get_ipython().system('pip install pymilvus>=2.3.0')

# Install OpenAI for embeddings
get_ipython().system('pip install openai')

print("✅ Installation complete!")

# =============================================================================
# CELL 2: Import Libraries
# =============================================================================

import os
import time
import numpy as np
from typing import List
import openai

# Set your OpenAI API key
# You can also use os.environ['OPENAI_API_KEY'] = 'your-key-here'
from google.colab import userdata
try:
    openai.api_key = userdata.get('OPENAI_API_KEY')
    print("✅ OpenAI API key loaded from Colab secrets")
except:
    print("⚠️  Please set OPENAI_API_KEY in Colab secrets or manually")
    # openai.api_key = "sk-..."  # Uncomment and add your key

# =============================================================================
# CELL 3: Helper Functions
# =============================================================================

def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """Get OpenAI embedding for text."""
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


# =============================================================================
# CELL 4: Example 1 - ChromaDB Backend (Embedded)
# =============================================================================

print_section("Example 1: ChromaDB Backend (Embedded)")

from memlayer.storage.chroma import ChromaStorage

# Initialize ChromaDB backend
print("🔧 Initializing ChromaDB backend...")
chroma_backend = ChromaStorage(
    storage_path="./memlayer_data",
    dimension=1536  # text-embedding-3-small dimension
)

# Test 1: Add memories
print("\n📝 Adding memories to ChromaDB...")
test_memories = [
    "The user's name is Alice Johnson",
    "Alice works as a software engineer at TechCorp",
    "Alice's favorite programming language is Python",
    "The project deadline is November 30th, 2025",
    "Alice prefers working remotely from home"
]

embeddings = [get_embedding(text) for text in test_memories]
user_ids = ["alice"] * len(test_memories)
metadatas = [{"importance_score": 0.8} for _ in test_memories]

memory_ids = chroma_backend.add_memories(
    contents=test_memories,
    embeddings=embeddings,
    user_ids=user_ids,
    metadatas=metadatas
)

print(f"✅ Added {len(memory_ids)} memories")
print(f"   Memory IDs: {memory_ids[:2]}...")

# Test 2: Search memories
print("\n🔍 Searching memories...")
query = "What is Alice's job?"
query_embedding = get_embedding(query)

results = chroma_backend.search_memories(
    query_embedding=query_embedding,
    user_id="alice",
    top_k=3
)

print(f"Query: '{query}'")
print(f"Found {len(results)} results:\n")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['content']}")
    print(f"   Score: {result['score']:.4f}")
    print()

# Test 3: Health check
print("\n🏥 Health check:")
health = chroma_backend.health_check()
print(f"   Status: {health['status']}")
print(f"   Latency: {health['latency_ms']:.2f}ms")
print(f"   Collection: {health['collection']}")
print(f"   Count: {health['count']}")

print("\n✅ ChromaDB backend test complete!")

# =============================================================================
# CELL 5: Example 2 - Qdrant Backend (Docker or Cloud)
# =============================================================================

print_section("Example 2: Qdrant Backend")

print("📌 SETUP INSTRUCTIONS:")
print("   Option A: Run Qdrant locally with Docker:")
print("   docker run -p 6333:6333 qdrant/qdrant")
print("")
print("   Option B: Use Qdrant Cloud:")
print("   1. Sign up at https://cloud.qdrant.io")
print("   2. Create a cluster")
print("   3. Get your URL and API key")
print("")

# Choose your setup
USE_QDRANT_CLOUD = False  # Set to True if using Qdrant Cloud

if USE_QDRANT_CLOUD:
    # Qdrant Cloud setup
    QDRANT_URL = "https://your-cluster-id.cloud.qdrant.io"
    QDRANT_API_KEY = "your-api-key-here"
else:
    # Local Docker setup
    QDRANT_URL = "http://localhost:6333"
    QDRANT_API_KEY = None

from memlayer.storage.qdrant_backend import QdrantVectorBackend

try:
    print(f"\n🔧 Initializing Qdrant backend at {QDRANT_URL}...")
    qdrant_backend = QdrantVectorBackend(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name="memlayer_demo",
        dimension=1536
    )

    # Test 1: Add memories
    print("\n📝 Adding memories to Qdrant...")
    test_memories_qdrant = [
        "Bob loves playing basketball",
        "Bob's email is bob@example.com",
        "Bob is learning machine learning",
        "Bob's meeting is on Friday at 3 PM",
        "Bob works at DataCorp as a data scientist"
    ]

    embeddings_qdrant = [get_embedding(text) for text in test_memories_qdrant]
    user_ids_qdrant = ["bob"] * len(test_memories_qdrant)

    memory_ids_qdrant = qdrant_backend.add_memories(
        contents=test_memories_qdrant,
        embeddings=embeddings_qdrant,
        user_ids=user_ids_qdrant
    )

    print(f"✅ Added {len(memory_ids_qdrant)} memories")

    # Test 2: Search memories
    print("\n🔍 Searching memories in Qdrant...")
    query_qdrant = "What does Bob do?"
    query_embedding_qdrant = get_embedding(query_qdrant)

    results_qdrant = qdrant_backend.search_memories(
        query_embedding=query_embedding_qdrant,
        user_id="bob",
        top_k=3
    )

    print(f"Query: '{query_qdrant}'")
    print(f"Found {len(results_qdrant)} results:\n")
    for i, result in enumerate(results_qdrant, 1):
        print(f"{i}. {result['content']}")
        print(f"   Score: {result['score']:.4f}")
        print()

    # Test 3: Health check
    print("\n🏥 Health check:")
    health_qdrant = qdrant_backend.health_check()
    print(f"   Status: {health_qdrant['status']}")
    print(f"   Latency: {health_qdrant['latency_ms']:.2f}ms")
    print(f"   Collection: {health_qdrant['collection']}")
    print(f"   Points count: {health_qdrant.get('points_count', 'N/A')}")

    print("\n✅ Qdrant backend test complete!")

except Exception as e:
    print(f"\n❌ Qdrant connection failed: {e}")
    print("   Make sure Qdrant is running or update your credentials")

# =============================================================================
# CELL 6: Example 3 - Zilliz Backend (Milvus Cloud)
# =============================================================================

print_section("Example 3: Zilliz Backend (Milvus Cloud)")

print("📌 SETUP INSTRUCTIONS:")
print("   1. Sign up at https://cloud.zilliz.com")
print("   2. Create a cluster")
print("   3. Get your URI and API token")
print("   4. Update the credentials below")
print("")

# Zilliz credentials
ZILLIZ_URI = "https://your-cluster-id.api.gcp-us-west1.zillizcloud.com"
ZILLIZ_TOKEN = "your-api-token-here"

# Set to True when you have valid credentials
USE_ZILLIZ = False

if USE_ZILLIZ:
    from memlayer.storage.zilliz_backend import ZillizVectorBackend

    try:
        print(f"\n🔧 Initializing Zilliz backend...")
        zilliz_backend = ZillizVectorBackend(
            uri=ZILLIZ_URI,
            token=ZILLIZ_TOKEN,
            collection_name="memlayer_demo",
            dimension=1536
        )

        # Test 1: Add memories
        print("\n📝 Adding memories to Zilliz...")
        test_memories_zilliz = [
            "Carol is a product manager",
            "Carol's phone number is 555-1234",
            "Carol loves traveling to Japan",
            "Carol's quarterly review is next Monday",
            "Carol works at StartupXYZ"
        ]

        embeddings_zilliz = [get_embedding(text) for text in test_memories_zilliz]
        user_ids_zilliz = ["carol"] * len(test_memories_zilliz)

        memory_ids_zilliz = zilliz_backend.add_memories(
            contents=test_memories_zilliz,
            embeddings=embeddings_zilliz,
            user_ids=user_ids_zilliz
        )

        print(f"✅ Added {len(memory_ids_zilliz)} memories")

        # Test 2: Search memories
        print("\n🔍 Searching memories in Zilliz...")
        query_zilliz = "Where does Carol work?"
        query_embedding_zilliz = get_embedding(query_zilliz)

        results_zilliz = zilliz_backend.search_memories(
            query_embedding=query_embedding_zilliz,
            user_id="carol",
            top_k=3
        )

        print(f"Query: '{query_zilliz}'")
        print(f"Found {len(results_zilliz)} results:\n")
        for i, result in enumerate(results_zilliz, 1):
            print(f"{i}. {result['content']}")
            print(f"   Score: {result['score']:.4f}")
            print()

        # Test 3: Health check
        print("\n🏥 Health check:")
        health_zilliz = zilliz_backend.health_check()
        print(f"   Status: {health_zilliz['status']}")
        print(f"   Latency: {health_zilliz['latency_ms']:.2f}ms")
        print(f"   Collection: {health_zilliz['collection']}")
        print(f"   Entities: {health_zilliz.get('num_entities', 'N/A')}")

        print("\n✅ Zilliz backend test complete!")

    except Exception as e:
        print(f"\n❌ Zilliz connection failed: {e}")
        print("   Make sure your credentials are correct")
else:
    print("⏭️  Skipping Zilliz test (set USE_ZILLIZ = True to enable)")

# =============================================================================
# CELL 7: Comparison and Benchmarks
# =============================================================================

print_section("Backend Comparison")

print("🔍 Comparing backends...")
print("")

comparison = {
    "ChromaDB": {
        "Type": "Embedded",
        "Setup": "No external service needed",
        "Cost": "Free (local storage)",
        "Use Case": "Dev/testing, small deployments",
        "Pros": "Simple, no dependencies",
        "Cons": "Not suitable for production scale"
    },
    "Qdrant": {
        "Type": "Client-server",
        "Setup": "Docker or Qdrant Cloud",
        "Cost": "Free (self-host) or $25+/mo (cloud)",
        "Use Case": "Production, high performance",
        "Pros": "Fast, feature-rich, good docs",
        "Cons": "Requires server management"
    },
    "Zilliz": {
        "Type": "Managed cloud",
        "Setup": "Cloud account",
        "Cost": "$0.10/hour (starter) or custom",
        "Use Case": "Large-scale production",
        "Pros": "Fully managed, highly scalable",
        "Cons": "Cloud-only, costs can add up"
    }
}

for backend, details in comparison.items():
    print(f"📊 {backend}")
    for key, value in details.items():
        print(f"   {key:12s}: {value}")
    print()

print("✅ Demo complete! All backends are compatible with Memlayer.")
print("   Choose the one that fits your needs:")
print("   • Development → ChromaDB")
print("   • Production → Qdrant (self-hosted)")
print("   • Enterprise → Zilliz or Qdrant Cloud")

# =============================================================================
# CELL 8: (Optional) Using with Memlayer's OpenAI Wrapper
# =============================================================================

print_section("Bonus: Using Vector Backends with Memlayer OpenAI Wrapper")

print("This demonstrates how the vector backends integrate with Memlayer's")
print("full memory consolidation and search pipeline.")
print("")

# Note: This would require updating SearchService and ConsolidationService
# to accept VectorBackend instead of ChromaStorage directly.
# The code below is a preview of how it would work:

print("""
from memlayer import OpenAI
from memlayer.storage.qdrant_backend import QdrantVectorBackend

# Initialize with custom vector backend
vector_backend = QdrantVectorBackend(
    url="http://localhost:6333",
    collection_name="memlayer_memories"
)

# Create OpenAI wrapper with custom backend
client = OpenAI(
    model="gpt-4o-mini",
    vector_backend=vector_backend,  # Use Qdrant instead of Chroma
    operation_mode="online"
)

# Use normally
response = client.chat([
    {"role": "user", "content": "My name is Alice and I work at TechCorp"}
])
# Memory automatically stored in Qdrant!

# Search later
response = client.chat([
    {"role": "user", "content": "Where do I work?"}
])
# LLM uses search_memory tool → searches Qdrant → returns context
""")

print("\n📝 Note: Full integration requires updating SearchService and")
print("   ConsolidationService to accept VectorBackend parameter.")
print("   See the implementation guide for details.")
