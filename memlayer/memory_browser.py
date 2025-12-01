"""
Built-in memory browser for Memlayer.

Provides easy-to-use functions to:
- View all stored memories
- Search and filter memories
- Export memory data
- Analyze memory statistics

Works with any vector backend (Chroma, Qdrant, Zilliz).
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import json


class MemoryBrowser:
    """
    Built-in memory browser for Memlayer.

    Usage:
        from memlayer.memory_browser import MemoryBrowser

        browser = MemoryBrowser(client.storage)
        browser.show_all()  # List all memories
        browser.search("project")  # Search memories
        browser.print_stats()  # Show statistics
        browser.show_embeddings()  # View embeddings info
        browser.filter_by_importance()  # Filter by importance score
        browser.export_json("memories.json")  # Export data
    """

    def __init__(self, vector_storage):
        """
        Initialize browser with a vector storage backend.

        Args:
            vector_storage: ChromaStorage, QdrantVectorBackend, or ZillizVectorBackend
        """
        self.storage = vector_storage

    def get_all_memories(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all memories from storage.

        Args:
            user_id: Optional filter by user_id

        Returns:
            List of memory dicts
        """
        try:
            memories = self.storage.get_all_memories_for_curation(user_id=user_id)
            return memories
        except Exception as e:
            print(f"❌ Error fetching memories: {e}")
            return []

    def show_all(
        self,
        user_id: Optional[str] = None,
        max_items: int = 20,
        show_metadata: bool = True
    ):
        """
        Display all memories in a formatted list.

        Args:
            user_id: Optional filter by user_id
            max_items: Maximum number of memories to show
            show_metadata: Whether to show metadata
        """
        memories = self.get_all_memories(user_id=user_id)

        if not memories:
            print("📭 No memories found.")
            return

        print("\n" + "=" * 70)
        print(f"  📚 All Memories ({len(memories)} total)")
        if user_id:
            print(f"  Filtered by user_id: {user_id}")
        print("=" * 70 + "\n")

        for i, memory in enumerate(memories[:max_items], 1):
            content = memory.get('content', '')
            mem_id = memory.get('id', 'unknown')
            metadata = memory.get('metadata', {})

            # Format content
            display_content = content if len(content) <= 100 else content[:100] + "..."

            print(f"{i}. {display_content}")
            print(f"   ID: {mem_id}")

            if show_metadata:
                # Show key metadata
                user = metadata.get('user_id', 'unknown')
                status = metadata.get('status', 'unknown')
                importance = metadata.get('importance_score', 0.0)
                access_count = metadata.get('access_count', 0)
                timestamp = metadata.get('timestamp', 0)

                # Format timestamp
                if timestamp:
                    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    time_str = 'unknown'

                print(f"   User: {user} | Status: {status} | Importance: {importance:.2f}")
                print(f"   Accessed: {access_count} times | Created: {time_str}")

            print()

        if len(memories) > max_items:
            print(f"... and {len(memories) - max_items} more memories")
            print(f"    (showing first {max_items})")

    def search(
        self,
        keyword: str,
        user_id: Optional[str] = None,
        max_items: int = 10
    ):
        """
        Search memories by keyword (simple text search).

        Args:
            keyword: Keyword to search for
            user_id: Optional filter by user_id
            max_items: Maximum results to show
        """
        memories = self.get_all_memories(user_id=user_id)

        # Filter by keyword
        keyword_lower = keyword.lower()
        matches = [
            m for m in memories
            if keyword_lower in m.get('content', '').lower()
        ]

        if not matches:
            print(f"❌ No memories found matching '{keyword}'")
            return

        print("\n" + "=" * 70)
        print(f"  🔍 Search Results for '{keyword}' ({len(matches)} found)")
        print("=" * 70 + "\n")

        for i, memory in enumerate(matches[:max_items], 1):
            content = memory.get('content', '')
            mem_id = memory.get('id', 'unknown')

            # Highlight keyword
            display_content = content.replace(
                keyword,
                f"**{keyword}**"  # Bold the match
            )

            if len(display_content) > 150:
                display_content = display_content[:150] + "..."

            print(f"{i}. {display_content}")
            print(f"   ID: {mem_id}")
            print()

        if len(matches) > max_items:
            print(f"... and {len(matches) - max_items} more matches")

    def filter_by_status(
        self,
        status: str = "active",
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter memories by status.

        Args:
            status: Status to filter by ('active', 'archived', etc.)
            user_id: Optional filter by user_id

        Returns:
            List of filtered memories
        """
        memories = self.get_all_memories(user_id=user_id)
        filtered = [
            m for m in memories
            if m.get('metadata', {}).get('status') == status
        ]
        return filtered

    def filter_by_importance(
        self,
        min_score: float = 0.5,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter memories by importance score.

        Args:
            min_score: Minimum importance score (0.0 - 1.0)
            user_id: Optional filter by user_id

        Returns:
            List of filtered memories
        """
        memories = self.get_all_memories(user_id=user_id)
        filtered = [
            m for m in memories
            if m.get('metadata', {}).get('importance_score', 0.0) >= min_score
        ]
        return filtered

    def print_stats(self, user_id: Optional[str] = None):
        """
        Print memory statistics.

        Args:
            user_id: Optional filter by user_id
        """
        memories = self.get_all_memories(user_id=user_id)

        if not memories:
            print("📭 No memories found.")
            return

        print("\n" + "=" * 70)
        print("  📊 Memory Statistics")
        if user_id:
            print(f"  User: {user_id}")
        print("=" * 70 + "\n")

        # Total count
        print(f"📈 Total memories: {len(memories)}")

        # Count by status
        status_counts = {}
        for m in memories:
            status = m.get('metadata', {}).get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1

        print(f"\n📊 By status:")
        for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
            print(f"   {status}: {count}")

        # Count by user
        user_counts = {}
        for m in memories:
            user = m.get('metadata', {}).get('user_id', 'unknown')
            user_counts[user] = user_counts.get(user, 0) + 1

        print(f"\n👥 By user:")
        for user, count in sorted(user_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"   {user}: {count}")

        # Importance distribution
        importance_scores = [
            m.get('metadata', {}).get('importance_score', 0.0)
            for m in memories
        ]
        if importance_scores:
            avg_importance = sum(importance_scores) / len(importance_scores)
            max_importance = max(importance_scores)
            min_importance = min(importance_scores)

            print(f"\n⭐ Importance scores:")
            print(f"   Average: {avg_importance:.2f}")
            print(f"   Range: {min_importance:.2f} - {max_importance:.2f}")

        # Access statistics
        access_counts = [
            m.get('metadata', {}).get('access_count', 0)
            for m in memories
        ]
        if access_counts:
            total_accesses = sum(access_counts)
            avg_accesses = total_accesses / len(access_counts)
            max_accesses = max(access_counts)

            print(f"\n👁️  Access statistics:")
            print(f"   Total accesses: {total_accesses}")
            print(f"   Average per memory: {avg_accesses:.1f}")
            print(f"   Most accessed: {max_accesses} times")

        # Most accessed memories
        top_accessed = sorted(
            memories,
            key=lambda m: m.get('metadata', {}).get('access_count', 0),
            reverse=True
        )[:5]

        print(f"\n🔥 Most accessed memories:")
        for i, m in enumerate(top_accessed, 1):
            content = m.get('content', '')[:60] + "..."
            access_count = m.get('metadata', {}).get('access_count', 0)
            print(f"   {i}. {content}")
            print(f"      Accessed {access_count} times")

        print("\n" + "=" * 70)

    def export_json(
        self,
        filename: str = "memories.json",
        user_id: Optional[str] = None,
        pretty: bool = True
    ):
        """
        Export memories to JSON file.

        Args:
            filename: Output filename
            user_id: Optional filter by user_id
            pretty: Whether to format JSON nicely
        """
        memories = self.get_all_memories(user_id=user_id)

        # Prepare export data
        export_data = {
            "metadata": {
                "total_memories": len(memories),
                "user_id": user_id,
                "exported_at": datetime.now(timezone.utc).isoformat()
            },
            "memories": memories
        }

        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            else:
                json.dump(export_data, f, ensure_ascii=False)

        print(f"✅ Exported {len(memories)} memories to: {filename}")

    def delete_by_id(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: Memory ID to delete

        Returns:
            True if successful
        """
        try:
            success = self.storage.delete_memory(memory_id)
            if success:
                print(f"✅ Deleted memory: {memory_id}")
            else:
                print(f"❌ Failed to delete memory: {memory_id}")
            return success
        except Exception as e:
            print(f"❌ Error deleting memory: {e}")
            return False

    def archive_by_id(self, memory_id: str) -> bool:
        """
        Archive a memory by ID (change status to 'archived').

        Args:
            memory_id: Memory ID to archive

        Returns:
            True if successful
        """
        try:
            success = self.storage.update_memory_status(memory_id, "archived")
            if success:
                print(f"✅ Archived memory: {memory_id}")
            else:
                print(f"❌ Failed to archive memory: {memory_id}")
            return success
        except Exception as e:
            print(f"❌ Error archiving memory: {e}")
            return False

    def show_embeddings(
        self,
        user_id: Optional[str] = None,
        max_items: int = 10,
        show_vector: bool = False,
        truncate_vector: int = 10
    ):
        """
        Display embeddings information for stored memories.

        This method shows embedding metadata and optionally the actual
        embedding vectors stored in the vector database.

        Args:
            user_id: Optional filter by user_id
            max_items: Maximum number of memories to show
            show_vector: If True, display actual embedding vectors
            truncate_vector: Number of dimensions to show (when show_vector=True)

        Note:
            Embedding vectors are typically 384-1536 dimensions. Displaying
            full vectors is usually not useful, so truncate_vector limits
            the number of dimensions shown.
        """
        # Check if storage supports embeddings
        if not hasattr(self.storage, 'collection'):
            print("❌ This storage backend does not support embeddings view.")
            print("   (LIGHTWEIGHT mode uses graph-only storage)")
            return

        try:
            # Get memories with embeddings from ChromaDB
            if user_id:
                results = self.storage.collection.get(
                    where={"user_id": {"$eq": user_id}},
                    include=["metadatas", "embeddings"]
                )
            else:
                results = self.storage.collection.get(
                    include=["metadatas", "embeddings"]
                )

            if not results or not results.get('ids'):
                print("📭 No embeddings found.")
                return

            ids = results['ids']
            metadatas = results.get('metadatas', [])
            embeddings = results.get('embeddings', [])

            print("\n" + "=" * 70)
            print(f"  🧠 Embeddings Information ({len(ids)} memories)")
            if user_id:
                print(f"  Filtered by user_id: {user_id}")
            print("=" * 70 + "\n")

            # Determine embedding dimension
            if embeddings and len(embeddings) > 0 and embeddings[0]:
                dimension = len(embeddings[0])
                print(f"📐 Embedding dimension: {dimension}")
                print(f"   (Typical: 384 for MiniLM, 1536 for OpenAI)\n")
            else:
                print("📐 Embedding dimension: Unknown (no embeddings stored)\n")
                dimension = 0

            for i, mem_id in enumerate(ids[:max_items]):
                metadata = metadatas[i] if i < len(metadatas) else {}
                content = metadata.get('content', '')[:60] + "..."

                print(f"{i+1}. {content}")
                print(f"   ID: {mem_id}")

                if embeddings and i < len(embeddings) and embeddings[i]:
                    embedding = embeddings[i]
                    # Calculate embedding statistics
                    import statistics
                    emb_min = min(embedding)
                    emb_max = max(embedding)
                    emb_mean = statistics.mean(embedding)
                    emb_std = statistics.stdev(embedding) if len(embedding) > 1 else 0

                    print(f"   Embedding stats: min={emb_min:.4f}, max={emb_max:.4f}, "
                          f"mean={emb_mean:.4f}, std={emb_std:.4f}")

                    if show_vector:
                        truncated = embedding[:truncate_vector]
                        formatted = [f"{v:.4f}" for v in truncated]
                        print(f"   Vector (first {truncate_vector}): [{', '.join(formatted)}, ...]")
                else:
                    print("   Embedding: Not available")

                print()

            if len(ids) > max_items:
                print(f"... and {len(ids) - max_items} more embeddings")
                print(f"    (showing first {max_items})")

            print("=" * 70)

        except Exception as e:
            print(f"❌ Error fetching embeddings: {e}")

    def get_embeddings(
        self,
        user_id: Optional[str] = None,
        memory_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get raw embedding data for memories.

        Args:
            user_id: Optional filter by user_id
            memory_ids: Optional list of specific memory IDs to fetch

        Returns:
            List of dicts containing id, content, and embedding vector
        """
        if not hasattr(self.storage, 'collection'):
            print("❌ This storage backend does not support embeddings.")
            return []

        try:
            if memory_ids:
                results = self.storage.collection.get(
                    ids=memory_ids,
                    include=["metadatas", "embeddings"]
                )
            elif user_id:
                results = self.storage.collection.get(
                    where={"user_id": {"$eq": user_id}},
                    include=["metadatas", "embeddings"]
                )
            else:
                results = self.storage.collection.get(
                    include=["metadatas", "embeddings"]
                )

            memories_with_embeddings = []
            ids = results.get('ids', [])
            metadatas = results.get('metadatas', [])
            embeddings = results.get('embeddings', [])

            for i, mem_id in enumerate(ids):
                memories_with_embeddings.append({
                    'id': mem_id,
                    'content': metadatas[i].get('content', '') if i < len(metadatas) else '',
                    'metadata': metadatas[i] if i < len(metadatas) else {},
                    'embedding': embeddings[i] if embeddings and i < len(embeddings) else None
                })

            return memories_with_embeddings

        except Exception as e:
            print(f"❌ Error fetching embeddings: {e}")
            return []
