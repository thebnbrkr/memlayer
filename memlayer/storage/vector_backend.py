"""
Abstract interface for vector storage backends.

This allows Memlayer to work with different vector databases:
- ChromaDB (embedded, for local/dev)
- Zilliz (Milvus cloud)
- Qdrant (self-hosted or cloud)
- Pinecone, Weaviate, etc.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class VectorBackend(ABC):
    """
    Abstract interface for vector storage backends.

    All vector backends must implement these methods to be compatible
    with Memlayer's SearchService and ConsolidationService.
    """

    @abstractmethod
    def add_memories(
        self,
        contents: List[str],
        embeddings: List[List[float]],
        user_ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add memories to vector storage.

        Args:
            contents: List of memory text contents
            embeddings: Pre-computed embeddings for each content
            user_ids: User identifiers for each memory
            metadatas: Optional metadata dicts for each memory
            ids: Optional memory IDs (auto-generated if None)

        Returns:
            List of memory IDs
        """
        pass

    @abstractmethod
    def search_memories(
        self,
        query_embedding: List[float],
        user_id: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar memories.

        Args:
            query_embedding: Query vector
            user_id: User identifier
            top_k: Number of results
            filter_dict: Additional filters (e.g., status='active')

        Returns:
            List of dicts with keys: id, content, metadata, score
        """
        pass

    @abstractmethod
    def track_memory_access(self, memory_ids: List[str]) -> None:
        """
        Track memory access (increment access_count, update last_accessed).

        Args:
            memory_ids: List of memory IDs to track
        """
        pass

    @abstractmethod
    def update_memory_status(self, memory_id: str, new_status: str) -> bool:
        """
        Update memory status (e.g., 'active' -> 'archived').

        Args:
            memory_id: Memory ID
            new_status: New status value

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: Memory ID to delete

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def get_all_memories_for_curation(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch all memories for curation process.

        Used by CurationService to calculate relevance and archive/delete.

        Args:
            user_id: Optional user filter (None = all users)

        Returns:
            List of memory dicts with full metadata
        """
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Check backend health and return status.

        Returns:
            Dict with keys: status ('healthy' or 'unhealthy'), latency_ms, error (optional)
        """
        pass

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Return backend type identifier (e.g., 'chroma', 'zilliz', 'qdrant')."""
        pass
