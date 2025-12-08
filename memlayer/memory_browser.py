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
