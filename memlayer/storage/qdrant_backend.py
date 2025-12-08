"""
Qdrant vector backend implementation.

Qdrant can be self-hosted or used via Qdrant Cloud.
"""

import time
import uuid
from typing import List, Dict, Optional, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range
)

from memlayer.storage.vector_backend import VectorBackend


class QdrantVectorBackend(VectorBackend):
    """
    Qdrant implementation of VectorBackend.

    Example usage:
        # Cloud
        backend = QdrantVectorBackend(
            url="https://xyz.cloud.qdrant.io",
            api_key="your-api-key",
            collection_name="memlayer_memories",
            dimension=1536
        )

        # Self-hosted
        backend = QdrantVectorBackend(
            url="http://localhost:6333",
            collection_name="memlayer_memories",
            dimension=1536
        )
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        collection_name: str = "memlayer_memories",
        dimension: int = 1536,
        distance: str = "Cosine"
    ):
        """
        Initialize Qdrant backend.

        Args:
            url: Qdrant server URL (cloud or self-hosted)
            api_key: API key (optional, for cloud)
            collection_name: Collection name
            dimension: Embedding dimension
            distance: Distance metric (Cosine, Euclid, Dot)
        """
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.dimension = dimension
        self.distance = distance
        self._client = None

    def _get_client(self) -> QdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                url=self.url,
                api_key=self.api_key
            )
            self._ensure_collection()
            print(f"[Qdrant] Connected to {self.url}")

        return self._client

    def _ensure_collection(self):
        """Ensure collection exists."""
        client = self._client

        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            # Map distance string to Distance enum
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclid": Distance.EUCLID,
                "Dot": Distance.DOT
            }

            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=distance_map.get(self.distance, Distance.COSINE)
                )
            )
            print(f"[Qdrant] Created collection '{self.collection_name}'")
        else:
            print(f"[Qdrant] Using existing collection '{self.collection_name}'")

    @property
    def backend_type(self) -> str:
        return "qdrant"

    def add_memories(
        self,
        contents: List[str],
        embeddings: List[List[float]],
        user_ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add memories to Qdrant."""
        client = self._get_client()

        if metadatas is None:
            metadatas = [{} for _ in contents]

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in contents]

        now = time.time()

        # Prepare points
        points = []
        for i, (memory_id, content, embedding, user_id, metadata) in enumerate(
            zip(ids, contents, embeddings, user_ids, metadatas)
        ):
            payload = {
                "content": content,
                "user_id": user_id,
                "timestamp": now,
                "status": "active",
                "access_count": 0,
                "last_accessed_timestamp": now,
                "importance_score": metadata.get("importance_score", 0.5),
                "expiration_timestamp": metadata.get("expiration_timestamp", 0.0) or 0.0,
            }

            points.append(PointStruct(
                id=memory_id,
                vector=embedding,
                payload=payload
            ))

        # Upsert
        client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"[Qdrant] Added {len(points)} memories")
        return ids

    def search_memories(
        self,
        query_embedding: List[float],
        user_id: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar memories."""
        client = self._get_client()

        # Build filter
        must_conditions = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="status", match=MatchValue(value="active"))
        ]

        if filter_dict:
            for key, value in filter_dict.items():
                if isinstance(value, (int, float)) and key.endswith("_timestamp"):
                    # Handle timestamp ranges if needed
                    pass
                else:
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )

        query_filter = Filter(must=must_conditions)

        # Search
        results = client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter
        )

        # Format results
        formatted = []
        for hit in results:
            formatted.append({
                "id": str(hit.id),
                "content": hit.payload.get("content", ""),
                "score": hit.score,
                "metadata": {
                    "user_id": hit.payload.get("user_id"),
                    "timestamp": hit.payload.get("timestamp"),
                    "status": hit.payload.get("status"),
                    "access_count": hit.payload.get("access_count"),
                    "last_accessed_timestamp": hit.payload.get("last_accessed_timestamp"),
                    "importance_score": hit.payload.get("importance_score"),
                    "expiration_timestamp": hit.payload.get("expiration_timestamp"),
                }
            })

        return formatted

    def track_memory_access(self, memory_ids: List[str]) -> None:
        """Track memory access by incrementing access_count."""
        client = self._get_client()
        now = time.time()

        for memory_id in memory_ids:
            # Fetch current point
            try:
                points = client.retrieve(
                    collection_name=self.collection_name,
                    ids=[memory_id]
                )

                if points:
                    point = points[0]
                    payload = point.payload.copy()

                    # Update tracking fields
                    payload["access_count"] = payload.get("access_count", 0) + 1
                    payload["last_accessed_timestamp"] = now

                    # Update point
                    client.set_payload(
                        collection_name=self.collection_name,
                        payload=payload,
                        points=[memory_id]
                    )
            except Exception as e:
                print(f"[Qdrant] Error tracking access for {memory_id}: {e}")

    def update_memory_status(self, memory_id: str, new_status: str) -> bool:
        """Update memory status."""
        client = self._get_client()

        try:
            client.set_payload(
                collection_name=self.collection_name,
                payload={"status": new_status},
                points=[memory_id]
            )
            return True
        except Exception as e:
            print(f"[Qdrant] Error updating status: {e}")
            return False

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        client = self._get_client()

        try:
            client.delete(
                collection_name=self.collection_name,
                points_selector=[memory_id]
            )
            return True
        except Exception as e:
            print(f"[Qdrant] Error deleting memory: {e}")
            return False

    def get_all_memories_for_curation(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch all memories for curation."""
        client = self._get_client()

        # Build filter
        must_conditions = [
            FieldCondition(key="status", match=MatchValue(value="active"))
        ]

        if user_id:
            must_conditions.append(
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            )

        query_filter = Filter(must=must_conditions)

        # Scroll through all points
        offset = None
        all_points = []

        while True:
            result = client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            points, offset = result

            if not points:
                break

            for point in points:
                all_points.append({
                    "id": str(point.id),
                    "content": point.payload.get("content", ""),
                    "metadata": point.payload
                })

            if offset is None:
                break

        return all_points

    def health_check(self) -> Dict[str, Any]:
        """Check backend health."""
        try:
            start = time.time()
            client = self._get_client()

            # Check collection info
            info = client.get_collection(self.collection_name)
            latency = (time.time() - start) * 1000

            return {
                "status": "healthy",
                "latency_ms": latency,
                "collection": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
