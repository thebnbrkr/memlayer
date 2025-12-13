"""
Customizable Search Tier Configuration
=======================================

Define custom search strategies for different use cases, going beyond
the default fast/balanced/deep tiers.

Example use cases:
- E-commerce: "product_lookup" (vector only, top 3)
- Customer support: "support_deep" (vector + graph + recency bias)
- Research: "academic_search" (deep graph, top 20 results)
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from enum import Enum


class SearchMode(Enum):
    """Which databases to query."""
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"  # Both vector + graph


@dataclass
class SearchTierConfig:
    """
    Configuration for a custom search tier.

    Attributes:
        name: Unique identifier for this tier
        mode: Which databases to query (vector_only, graph_only, hybrid)
        vector_top_k: Number of vector results to retrieve
        graph_depth: Graph traversal depth (1-hop, 2-hop, etc.)
        enable_graph: Whether to use graph search
        recency_boost: Boost factor for recent memories (0.0 = no boost, 1.0 = strong boost)
        score_threshold: Minimum similarity score (0.0-1.0)
        metadata_filters: Optional filters (e.g., {"source": "work"})
        custom_ranker: Optional function to re-rank results
    """

    name: str
    mode: SearchMode = SearchMode.VECTOR_ONLY
    vector_top_k: int = 5
    graph_depth: int = 1
    enable_graph: bool = False
    recency_boost: float = 0.0  # 0.0 = no boost, 1.0 = strong boost
    score_threshold: float = 0.0  # Minimum similarity score
    metadata_filters: Optional[Dict[str, Any]] = None
    custom_ranker: Optional[Callable] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.mode == SearchMode.HYBRID:
            self.enable_graph = True

        if self.vector_top_k < 1:
            raise ValueError("vector_top_k must be >= 1")

        if self.graph_depth < 1:
            raise ValueError("graph_depth must be >= 1")

        if not 0.0 <= self.recency_boost <= 1.0:
            raise ValueError("recency_boost must be between 0.0 and 1.0")


# ============================================================================
# Predefined Tier Configurations
# ============================================================================

# Default tiers (matching current behavior)
DEFAULT_TIERS = {
    "fast": SearchTierConfig(
        name="fast",
        mode=SearchMode.VECTOR_ONLY,
        vector_top_k=2,
        enable_graph=False
    ),
    "balanced": SearchTierConfig(
        name="balanced",
        mode=SearchMode.VECTOR_ONLY,
        vector_top_k=5,
        enable_graph=False
    ),
    "deep": SearchTierConfig(
        name="deep",
        mode=SearchMode.HYBRID,
        vector_top_k=10,
        graph_depth=2,
        enable_graph=True
    ),
}

# E-commerce tiers
ECOMMERCE_TIERS = {
    "product_lookup": SearchTierConfig(
        name="product_lookup",
        mode=SearchMode.VECTOR_ONLY,
        vector_top_k=3,
        score_threshold=0.7,  # High precision for products
        recency_boost=0.3  # Slight recency bias for recent views
    ),
    "recommendation": SearchTierConfig(
        name="recommendation",
        mode=SearchMode.HYBRID,
        vector_top_k=10,
        graph_depth=2,  # Find related products via graph
        recency_boost=0.5  # Prefer recently viewed items
    ),
    "browse_history": SearchTierConfig(
        name="browse_history",
        mode=SearchMode.VECTOR_ONLY,
        vector_top_k=20,
        recency_boost=0.8  # Heavily favor recent browsing
    ),
}

# Customer support tiers
SUPPORT_TIERS = {
    "quick_answer": SearchTierConfig(
        name="quick_answer",
        mode=SearchMode.VECTOR_ONLY,
        vector_top_k=3,
        score_threshold=0.8  # High confidence answers only
    ),
    "deep_investigation": SearchTierConfig(
        name="deep_investigation",
        mode=SearchMode.HYBRID,
        vector_top_k=15,
        graph_depth=3,  # Deep traversal for complex issues
        recency_boost=0.6  # Consider recent interactions
    ),
    "ticket_history": SearchTierConfig(
        name="ticket_history",
        mode=SearchMode.GRAPH_ONLY,
        graph_depth=2,  # Traverse ticket relationships
        enable_graph=True
    ),
}

# Research / academic tiers
RESEARCH_TIERS = {
    "literature_review": SearchTierConfig(
        name="literature_review",
        mode=SearchMode.HYBRID,
        vector_top_k=20,
        graph_depth=3,  # Deep citation network
        recency_boost=0.2  # Slight bias for recent papers
    ),
    "citation_network": SearchTierConfig(
        name="citation_network",
        mode=SearchMode.GRAPH_ONLY,
        graph_depth=4,  # Very deep traversal
        enable_graph=True
    ),
    "quick_fact": SearchTierConfig(
        name="quick_fact",
        mode=SearchMode.VECTOR_ONLY,
        vector_top_k=1,  # Just the top result
        score_threshold=0.9  # Very high confidence
    ),
}

# Supermemory-inspired tiers (recency-focused)
SUPERMEMORY_TIERS = {
    "hot_memory": SearchTierConfig(
        name="hot_memory",
        mode=SearchMode.VECTOR_ONLY,
        vector_top_k=5,
        recency_boost=0.9,  # Heavy recency bias (like Supermemory)
        metadata_filters={"age_hours": 24}  # Last 24 hours only
    ),
    "cold_memory": SearchTierConfig(
        name="cold_memory",
        mode=SearchMode.HYBRID,
        vector_top_k=10,
        graph_depth=2,
        recency_boost=0.0,  # No recency bias for old memories
        metadata_filters={"age_hours": 168}  # Older than 1 week
    ),
    "contextual_recall": SearchTierConfig(
        name="contextual_recall",
        mode=SearchMode.HYBRID,
        vector_top_k=8,
        graph_depth=2,
        recency_boost=0.5,  # Balanced recency
    ),
}


# ============================================================================
# Custom Tier Builder
# ============================================================================

class SearchTierBuilder:
    """
    Fluent builder for creating custom search tier configs.

    Example:
        tier = (SearchTierBuilder("my_tier")
                .vector_only()
                .top_k(7)
                .recency_boost(0.6)
                .build())
    """

    def __init__(self, name: str):
        self.name = name
        self._mode = SearchMode.VECTOR_ONLY
        self._vector_top_k = 5
        self._graph_depth = 1
        self._enable_graph = False
        self._recency_boost = 0.0
        self._score_threshold = 0.0
        self._metadata_filters = None
        self._custom_ranker = None

    def vector_only(self):
        """Use vector search only."""
        self._mode = SearchMode.VECTOR_ONLY
        self._enable_graph = False
        return self

    def graph_only(self):
        """Use graph search only."""
        self._mode = SearchMode.GRAPH_ONLY
        self._enable_graph = True
        return self

    def hybrid(self):
        """Use both vector + graph."""
        self._mode = SearchMode.HYBRID
        self._enable_graph = True
        return self

    def top_k(self, k: int):
        """Set number of vector results."""
        self._vector_top_k = k
        return self

    def depth(self, d: int):
        """Set graph traversal depth."""
        self._graph_depth = d
        return self

    def recency_boost(self, boost: float):
        """Set recency boost factor (0.0-1.0)."""
        self._recency_boost = boost
        return self

    def score_threshold(self, threshold: float):
        """Set minimum similarity score."""
        self._score_threshold = threshold
        return self

    def filter(self, **filters):
        """Add metadata filters."""
        self._metadata_filters = filters
        return self

    def ranker(self, func: Callable):
        """Add custom ranking function."""
        self._custom_ranker = func
        return self

    def build(self) -> SearchTierConfig:
        """Build the configuration."""
        return SearchTierConfig(
            name=self.name,
            mode=self._mode,
            vector_top_k=self._vector_top_k,
            graph_depth=self._graph_depth,
            enable_graph=self._enable_graph,
            recency_boost=self._recency_boost,
            score_threshold=self._score_threshold,
            metadata_filters=self._metadata_filters,
            custom_ranker=self._custom_ranker
        )


# ============================================================================
# Tier Registry
# ============================================================================

class SearchTierRegistry:
    """
    Global registry for custom search tiers.

    Allows you to define tiers once and reuse them across your application.
    """

    def __init__(self):
        self._tiers: Dict[str, SearchTierConfig] = {}
        # Load defaults
        self._tiers.update(DEFAULT_TIERS)

    def register(self, config: SearchTierConfig):
        """Register a custom tier."""
        self._tiers[config.name] = config

    def get(self, name: str) -> SearchTierConfig:
        """Get a tier by name."""
        if name not in self._tiers:
            raise ValueError(f"Unknown search tier: {name}. Available: {list(self._tiers.keys())}")
        return self._tiers[name]

    def list_tiers(self) -> list[str]:
        """List all registered tier names."""
        return list(self._tiers.keys())

    def load_preset(self, preset: str):
        """
        Load a preset collection of tiers.

        Args:
            preset: "ecommerce", "support", "research", "supermemory"
        """
        if preset == "ecommerce":
            self._tiers.update(ECOMMERCE_TIERS)
        elif preset == "support":
            self._tiers.update(SUPPORT_TIERS)
        elif preset == "research":
            self._tiers.update(RESEARCH_TIERS)
        elif preset == "supermemory":
            self._tiers.update(SUPERMEMORY_TIERS)
        else:
            raise ValueError(f"Unknown preset: {preset}")


# Global registry instance
_global_registry = SearchTierRegistry()


def get_tier(name: str) -> SearchTierConfig:
    """Get a search tier from global registry."""
    return _global_registry.get(name)


def register_tier(config: SearchTierConfig):
    """Register a search tier in global registry."""
    _global_registry.register(config)


def load_preset(preset: str):
    """Load a preset collection of tiers."""
    _global_registry.load_preset(preset)


def list_tiers() -> list[str]:
    """List all registered tiers."""
    return _global_registry.list_tiers()
