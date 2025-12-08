import time
import uuid
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from memlayer.storage.vector_backend import VectorBackend


class ChromaStorage(VectorBackend):
    """
    A vector storage backend using the embedded, on-disk version of ChromaDB.
    """
    def __init__(self, storage_path: str, dimension: int): # <-- Accept dimension
        self.db_path = str(Path(storage_path) / "chroma")
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # We need to create a custom embedding function that does nothing,
        # since we will be providing the embeddings directly.
        class NoOpEmbeddingFunction(chromadb.EmbeddingFunction):
            def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
                # This should not be called if we always provide embeddings.
                # The dimension is what's important for collection creation.
                return []

        self.collection = self.client.get_or_create_collection(
            name=f"memories_dim_{dimension}", # <-- Collection name includes dimension
            embedding_function=NoOpEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"} # ChromaDB infers dimension from embeddings
        )
        print(f"Memlayer (ChromaDB) initialized at: {self.db_path} for dimension {dimension}")

    @property
    def backend_type(self) -> str:
        """Return backend type identifier."""
        return "chroma"

    def add_memories(
        self,
        contents: List[str],
        embeddings: List[List[float]],
        user_ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add memories to ChromaDB (batch operation).

        This is the VectorBackend interface method.
        """
        if metadatas is None:
            metadatas = [{} for _ in contents]

        if ids is None:
            ids = [f"mem_{uuid.uuid4().hex}" for _ in contents]

        now = time.time()

        # Prepare batch data
        batch_metadatas = []
        batch_documents = []

        for i, (content, user_id, metadata) in enumerate(zip(contents, user_ids, metadatas)):
            base_attrs = {
                "user_id": user_id,
                "timestamp": now,
                "content": content,
                "status": "active",
                "access_count": 0,
                "last_accessed_timestamp": now,
                "importance_score": metadata.get("importance_score", 0.5),
            }

            # Add expiration_timestamp if provided
            if metadata.get("expiration_timestamp") is not None:
                base_attrs["expiration_timestamp"] = metadata["expiration_timestamp"]

            # Filter out None values (ChromaDB only accepts str, int, float, bool)
            base_attrs = {k: v for k, v in base_attrs.items() if v is not None}

            batch_metadatas.append(base_attrs)
            batch_documents.append(content[:100])

        # Batch insert
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=batch_metadatas,
            documents=batch_documents
        )

        return ids

    def add_memory(self, content: str, embedding: List[float], user_id: str = "default_user", metadata: Dict = None):
        """
        Adds a single memory (backwards compatibility).

        This is a convenience method that calls add_memories internally.
        """
        metadata = metadata or {}
        ids = self.add_memories(
            contents=[content],
            embeddings=[embedding],
            user_ids=[user_id],
            metadatas=[metadata]
        )
        return ids[0]

    def search_memories(
        self,
        query_embedding: List[float],
        user_id: str = "default_user",
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches for similar memories (VectorBackend interface).

        Args:
            query_embedding: Query vector
            user_id: User identifier
            top_k: Number of results
            filter_dict: Additional filters (optional)

        Returns:
            List of dicts with keys: id, content, metadata, score
        """
        # Build filter
        where_conditions = [
            {"user_id": {"$eq": user_id}},
            {"status": {"$eq": "active"}}
        ]

        # Add additional filters if provided
        if filter_dict:
            for key, value in filter_dict.items():
                where_conditions.append({key: {"$eq": value}})

        where_filter = {"$and": where_conditions}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )
        
        memories = []
        if not results or not results.get('metadatas') or not results.get('distances'):
            return []

        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        ids = results['ids'][0]

        for i, meta in enumerate(metadatas):
            memories.append({
                "id": ids[i],
                "content": meta.get("content", ""),
                "timestamp": datetime.fromtimestamp(meta.get("timestamp", 0), tz=timezone.utc),
                "metadata": meta,
                "score": 1 - distances[i]
            })
            
        return memories

    def close(self):
        """Close the ChromaDB client and release file locks."""
        try:
            # Clear collection reference first
            self.collection = None
            
            # Clear client reference
            if self.client is not None:
                # ChromaDB doesn't have an explicit close, but clearing the reference
                # and forcing garbage collection helps on Windows
                self.client = None
                
                # Force garbage collection to help release file handles
                import gc
                gc.collect()
        except Exception as e:
            print(f"Warning: Error during ChromaDB cleanup: {e}")
    def track_memory_access(self, memory_ids: List[str]):
        """Increments access count and updates timestamp for given memory IDs."""
        if not memory_ids: return
        
        current_metadatas = self.collection.get(ids=memory_ids, include=["metadatas"])['metadatas']
        new_metadatas = []
        for meta in current_metadatas:
            meta['access_count'] = meta.get('access_count', 0) + 1
            meta['last_accessed_timestamp'] = time.time()
            new_metadatas.append(meta)
        
        if new_metadatas:
            self.collection.update(ids=memory_ids, metadatas=new_metadatas)

    def get_all_memories_for_curation(self, user_id: Optional[str] = None) -> List[Dict]:
        """
        Returns all memories with their lifecycle metadata (VectorBackend interface).

        Args:
            user_id: Optional user filter (None = all users)

        Returns:
            List of memory dicts with full metadata
        """
        # Note: Chroma's get() without IDs can be slow on huge collections.
        # For production, this might need batching. For now, this is fine.
        if user_id:
            # Filter by user_id
            results = self.collection.get(
                where={"user_id": {"$eq": user_id}},
                include=["metadatas"]
            )
        else:
            # Get all memories
            results = self.collection.get(include=["metadatas"])

        memories = []
        for i, meta in enumerate(results['metadatas']):
            memories.append({
                "id": results['ids'][i],
                "content": meta.get("content", ""),
                "metadata": meta
            })
        return memories

    def update_memory_status(self, memory_id: str, new_status: str):
        """Updates the status of a memory (e.g., to 'archived')."""
        try:
            result = self.collection.get(ids=[memory_id], include=["metadatas"])
            if result['metadatas'] and len(result['metadatas']) > 0:
                current_meta = result['metadatas'][0]
                current_meta['status'] = new_status
                self.collection.update(ids=[memory_id], metadatas=[current_meta])
        except Exception as e:
            # Memory might not exist in vector store (e.g., only in graph)
            pass

    def delete_memory(self, memory_id: str) -> bool:
        """Permanently deletes a memory."""
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            # Memory might not exist in vector store
            print(f"[ChromaDB] Error deleting memory: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Check backend health and return status."""
        try:
            start = time.time()
            # Try to count documents
            count = self.collection.count()
            latency = (time.time() - start) * 1000

            return {
                "status": "healthy",
                "latency_ms": latency,
                "collection": self.collection.name,
                "count": count
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }