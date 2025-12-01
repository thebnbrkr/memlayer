"""
Zilliz (Milvus Cloud) vector backend implementation.

Zilliz is the fully-managed cloud version of Milvus.
"""

import time
import uuid
from typing import List, Dict, Optional, Any

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)

from memlayer.storage.vector_backend import VectorBackend


class ZillizVectorBackend(VectorBackend):
    """
    Zilliz (Milvus Cloud) implementation of VectorBackend.

    Example usage:
        backend = ZillizVectorBackend(
            uri="https://your-cluster.api.gcp-us-west1.zillizcloud.com",
            token="your-api-token",
            collection_name="memlayer_memories",
            dimension=1536
        )
    """

    def __init__(
        self,
        uri: str,
        token: str,
        collection_name: str = "memlayer_memories",
        dimension: int = 1536,
        metric_type: str = "COSINE"
    ):
        """
        Initialize Zilliz backend.

        Args:
            uri: Zilliz cluster URI (e.g., https://xxx.api.gcp-us-west1.zillizcloud.com)
            token: API token for authentication
            collection_name: Collection name to use
            dimension: Embedding dimension
            metric_type: Distance metric (COSINE, L2, IP)
        """
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric_type = metric_type
        self._collection = None
        self._connected = False

    def _connect(self):
        """Connect to Zilliz and ensure collection exists."""
        if not self._connected:
            connections.connect(
                alias="default",
                uri=self.uri,
                token=self.token
            )
            self._connected = True
            print(f"[Zilliz] Connected to {self.uri}")

        # Ensure collection exists
        if not utility.has_collection(self.collection_name):
            self._create_collection()

        # Load collection
        if self._collection is None:
            self._collection = Collection(self.collection_name)
            self._collection.load()
            print(f"[Zilliz] Loaded collection '{self.collection_name}'")

    def _create_collection(self):
        """Create collection with schema."""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="timestamp", dtype=DataType.DOUBLE),
            FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="access_count", dtype=DataType.INT64),
            FieldSchema(name="last_accessed_timestamp", dtype=DataType.DOUBLE),
            FieldSchema(name="importance_score", dtype=DataType.DOUBLE),
            FieldSchema(name="expiration_timestamp", dtype=DataType.DOUBLE),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Memlayer memory storage"
        )

        collection = Collection(
            name=self.collection_name,
            schema=schema
        )

        # Create index for vector field
        index_params = {
            "metric_type": self.metric_type,
            "index_type": "AUTOINDEX",
            "params": {}
        }
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        print(f"[Zilliz] Created collection '{self.collection_name}'")

    @property
    def backend_type(self) -> str:
        return "zilliz"

    def add_memories(
        self,
        contents: List[str],
        embeddings: List[List[float]],
        user_ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add memories to Zilliz."""
        self._connect()

        if metadatas is None:
            metadatas = [{} for _ in contents]

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in contents]

        now = time.time()

        # Prepare data
        data = []
        for i, (memory_id, content, embedding, user_id, metadata) in enumerate(
            zip(ids, contents, embeddings, user_ids, metadatas)
        ):
            data.append({
                "id": memory_id,
                "embedding": embedding,
                "content": content,
                "user_id": user_id,
                "timestamp": now,
                "status": "active",
                "access_count": 0,
                "last_accessed_timestamp": now,
                "importance_score": metadata.get("importance_score", 0.5),
                "expiration_timestamp": metadata.get("expiration_timestamp", 0.0) or 0.0,
            })

        # Insert
        self._collection.insert(data)
        self._collection.flush()

        print(f"[Zilliz] Added {len(data)} memories")
        return ids

    def search_memories(
        self,
        query_embedding: List[float],
        user_id: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar memories."""
        self._connect()

        # Build filter expression
        filter_expr = f'user_id == "{user_id}" && status == "active"'
        if filter_dict:
            for key, value in filter_dict.items():
                if isinstance(value, str):
                    filter_expr += f' && {key} == "{value}"'
                else:
                    filter_expr += f' && {key} == {value}'

        # Search
        search_params = {
            "metric_type": self.metric_type,
            "params": {}
        }

        results = self._collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["id", "content", "user_id", "timestamp", "status",
                          "access_count", "importance_score", "expiration_timestamp"]
        )

        # Format results
        formatted = []
        for hits in results:
            for hit in hits:
                score = 1.0 - hit.distance if self.metric_type == "COSINE" else hit.distance
                formatted.append({
                    "id": hit.entity.get("id"),
                    "content": hit.entity.get("content"),
                    "score": score,
                    "metadata": {
                        "user_id": hit.entity.get("user_id"),
                        "timestamp": hit.entity.get("timestamp"),
                        "status": hit.entity.get("status"),
                        "access_count": hit.entity.get("access_count"),
                        "importance_score": hit.entity.get("importance_score"),
                        "expiration_timestamp": hit.entity.get("expiration_timestamp"),
                    }
                })

        return formatted

    def track_memory_access(self, memory_ids: List[str]) -> None:
        """Track memory access by incrementing access_count."""
        # Zilliz doesn't support in-place updates easily
        # For now, we'll skip this or implement via delete+reinsert
        # In production, you might use a separate tracking system
        pass

    def update_memory_status(self, memory_id: str, new_status: str) -> bool:
        """Update memory status."""
        # Zilliz requires delete + reinsert for updates
        # For now, we'll use a simple delete with filter
        self._connect()

        expr = f'id == "{memory_id}"'
        self._collection.delete(expr)
        # In production, you'd fetch the old data, update status, and reinsert
        return True

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        self._connect()

        expr = f'id == "{memory_id}"'
        self._collection.delete(expr)
        self._collection.flush()
        return True

    def get_all_memories_for_curation(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch all memories for curation."""
        self._connect()

        # Build filter
        filter_expr = 'status == "active"'
        if user_id:
            filter_expr += f' && user_id == "{user_id}"'

        # Query all
        results = self._collection.query(
            expr=filter_expr,
            output_fields=["id", "content", "user_id", "timestamp", "status",
                          "access_count", "last_accessed_timestamp", "importance_score",
                          "expiration_timestamp"],
            limit=10000  # Adjust as needed
        )

        formatted = []
        for result in results:
            formatted.append({
                "id": result.get("id"),
                "content": result.get("content"),
                "metadata": {
                    "user_id": result.get("user_id"),
                    "timestamp": result.get("timestamp"),
                    "status": result.get("status"),
                    "access_count": result.get("access_count"),
                    "last_accessed_timestamp": result.get("last_accessed_timestamp"),
                    "importance_score": result.get("importance_score"),
                    "expiration_timestamp": result.get("expiration_timestamp"),
                }
            })

        return formatted

    def health_check(self) -> Dict[str, Any]:
        """Check backend health."""
        try:
            start = time.time()
            self._connect()
            latency = (time.time() - start) * 1000

            # Check collection stats
            stats = self._collection.num_entities

            return {
                "status": "healthy",
                "latency_ms": latency,
                "collection": self.collection_name,
                "num_entities": stats
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
