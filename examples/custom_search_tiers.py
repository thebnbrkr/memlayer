"""
Custom Search Tiers Example
============================

Shows how to create and use custom search tier configurations
for different use cases (e-commerce, support, research, etc.)
"""

import os
from datetime import datetime, timedelta
from memlayer import OpenAI as Memlayer
from memlayer.config.search_tiers import (
    SearchTierConfig,
    SearchMode,
    SearchTierBuilder,
    load_preset,
    register_tier,
    list_tiers,
    get_tier
)


# ============================================================================
# Example 1: E-commerce Product Recommendations
# ============================================================================

def example_ecommerce():
    """
    E-commerce use case with custom tiers for:
    - Quick product lookup
    - Personalized recommendations
    - Browse history
    """

    print("=" * 70)
    print("🛍️  EXAMPLE 1: E-commerce Custom Search Tiers")
    print("=" * 70)

    # Load e-commerce preset tiers
    load_preset("ecommerce")

    print("\nAvailable tiers:")
    for tier_name in list_tiers():
        if tier_name.startswith("product") or tier_name.startswith("recommendation") or tier_name.startswith("browse"):
            tier = get_tier(tier_name)
            print(f"  - {tier.name}: mode={tier.mode.value}, top_k={tier.vector_top_k}, recency={tier.recency_boost}")

    # Example: Product lookup (fast, precise)
    tier = get_tier("product_lookup")
    print(f"\n📦 Product Lookup Tier:")
    print(f"   Mode: {tier.mode.value}")
    print(f"   Top-K: {tier.vector_top_k}")
    print(f"   Score threshold: {tier.score_threshold} (high precision)")
    print(f"   Recency boost: {tier.recency_boost}")

    # Example: Recommendation (hybrid, explores graph)
    tier = get_tier("recommendation")
    print(f"\n💡 Recommendation Tier:")
    print(f"   Mode: {tier.mode.value}")
    print(f"   Top-K: {tier.vector_top_k}")
    print(f"   Graph depth: {tier.graph_depth}")
    print(f"   Recency boost: {tier.recency_boost}")
    print("   → Uses graph to find 'customers who bought X also bought Y'")


# ============================================================================
# Example 2: Customer Support Tiers
# ============================================================================

def example_customer_support():
    """
    Customer support use case with tiers for:
    - Quick answers (FAQ-style)
    - Deep investigation (complex issues)
    - Ticket history (relationship traversal)
    """

    print("\n" + "=" * 70)
    print("🎧 EXAMPLE 2: Customer Support Custom Search Tiers")
    print("=" * 70)

    # Load support preset
    load_preset("support")

    # Quick answer tier
    quick = get_tier("quick_answer")
    print(f"\n⚡ Quick Answer Tier:")
    print(f"   Mode: {quick.mode.value}")
    print(f"   Top-K: {quick.vector_top_k}")
    print(f"   Threshold: {quick.score_threshold} (only high confidence)")
    print("   Use case: FAQ-style questions, simple troubleshooting")

    # Deep investigation tier
    deep = get_tier("deep_investigation")
    print(f"\n🔍 Deep Investigation Tier:")
    print(f"   Mode: {deep.mode.value}")
    print(f"   Top-K: {deep.vector_top_k}")
    print(f"   Graph depth: {deep.graph_depth} (3-hop traversal!)")
    print(f"   Recency boost: {deep.recency_boost}")
    print("   Use case: Complex issues, debugging, root cause analysis")

    # Ticket history tier
    ticket = get_tier("ticket_history")
    print(f"\n📋 Ticket History Tier:")
    print(f"   Mode: {ticket.mode.value}")
    print(f"   Graph depth: {ticket.graph_depth}")
    print("   Use case: 'Show all related tickets for this customer'")


# ============================================================================
# Example 3: Research / Academic Tiers
# ============================================================================

def example_research():
    """
    Research use case with tiers for:
    - Literature review (broad search)
    - Citation network (deep graph)
    - Quick fact lookup (precise)
    """

    print("\n" + "=" * 70)
    print("📚 EXAMPLE 3: Research / Academic Custom Search Tiers")
    print("=" * 70)

    load_preset("research")

    # Literature review
    lit = get_tier("literature_review")
    print(f"\n📖 Literature Review Tier:")
    print(f"   Mode: {lit.mode.value}")
    print(f"   Top-K: {lit.vector_top_k}")
    print(f"   Graph depth: {lit.graph_depth}")
    print(f"   Recency boost: {lit.recency_boost} (slight bias for recent papers)")
    print("   Use case: 'Show all papers related to transformers in NLP'")

    # Citation network
    cite = get_tier("citation_network")
    print(f"\n🔗 Citation Network Tier:")
    print(f"   Mode: {cite.mode.value}")
    print(f"   Graph depth: {cite.graph_depth} (4-hop! Very deep)")
    print("   Use case: 'How did ResNet influence BERT?' (citation chains)")

    # Quick fact
    fact = get_tier("quick_fact")
    print(f"\n⚡ Quick Fact Tier:")
    print(f"   Mode: {fact.mode.value}")
    print(f"   Top-K: {fact.vector_top_k} (just the top result)")
    print(f"   Threshold: {fact.score_threshold} (very high confidence)")
    print("   Use case: 'Who invented the transformer architecture?'")


# ============================================================================
# Example 4: Supermemory-Inspired Recency Tiers
# ============================================================================

def example_supermemory_style():
    """
    Supermemory-inspired tiers with recency bias:
    - Hot memory (recent, fast)
    - Cold memory (older, comprehensive)
    - Contextual recall (balanced)
    """

    print("\n" + "=" * 70)
    print("🧠 EXAMPLE 4: Supermemory-Style Recency Tiers")
    print("=" * 70)

    load_preset("supermemory")

    # Hot memory (like Supermemory's Cloudflare KV tier)
    hot = get_tier("hot_memory")
    print(f"\n🔥 Hot Memory Tier (Recent/Fast):")
    print(f"   Mode: {hot.mode.value}")
    print(f"   Top-K: {hot.vector_top_k}")
    print(f"   Recency boost: {hot.recency_boost} (VERY high - like Supermemory)")
    print(f"   Filters: {hot.metadata_filters}")
    print("   Use case: 'What did we discuss in the last 24 hours?'")
    print("   → Mimics Supermemory's hot tier (Cloudflare KV)")

    # Cold memory (older, deep search)
    cold = get_tier("cold_memory")
    print(f"\n❄️  Cold Memory Tier (Older/Comprehensive):")
    print(f"   Mode: {cold.mode.value}")
    print(f"   Top-K: {cold.vector_top_k}")
    print(f"   Graph depth: {cold.graph_depth}")
    print(f"   Recency boost: {cold.recency_boost} (no recency bias)")
    print(f"   Filters: {cold.metadata_filters}")
    print("   Use case: 'What did we discuss about X last year?'")
    print("   → Mimics Supermemory's cost-effective storage tier")


# ============================================================================
# Example 5: Build Your Own Custom Tier
# ============================================================================

def example_custom_tier():
    """
    Build a completely custom tier using the fluent builder API.
    """

    print("\n" + "=" * 70)
    print("🛠️  EXAMPLE 5: Build Your Own Custom Tier")
    print("=" * 70)

    # Example 1: Social media feed tier
    social_feed = (
        SearchTierBuilder("social_feed")
        .vector_only()
        .top_k(15)
        .recency_boost(0.9)  # Very recent content
        .score_threshold(0.5)  # Lower threshold for discovery
        .build()
    )

    print(f"\n📱 Custom Social Feed Tier:")
    print(f"   {social_feed}")
    print("   → Lots of results, heavy recency bias, lower precision")

    # Register it
    register_tier(social_feed)
    print(f"   ✅ Registered! Can now use get_tier('social_feed')")

    # Example 2: Enterprise knowledge base tier
    kb_tier = (
        SearchTierBuilder("enterprise_kb")
        .hybrid()
        .top_k(12)
        .depth(3)
        .recency_boost(0.3)  # Slight recency
        .score_threshold(0.75)  # High precision
        .filter(department="engineering", access_level="public")
        .build()
    )

    print(f"\n🏢 Custom Enterprise KB Tier:")
    print(f"   {kb_tier}")
    print("   → Hybrid search, filtered by department, high precision")

    register_tier(kb_tier)
    print(f"   ✅ Registered!")

    # Example 3: Debugging tier (very deep graph)
    debug_tier = (
        SearchTierBuilder("debug_mode")
        .hybrid()
        .top_k(20)
        .depth(5)  # VERY deep
        .recency_boost(0.7)  # Recent issues more relevant
        .build()
    )

    print(f"\n🐛 Custom Debug Tier:")
    print(f"   {debug_tier}")
    print("   → Very deep graph (5-hop!), lots of results, recent bias")

    register_tier(debug_tier)
    print(f"   ✅ Registered!")


# ============================================================================
# Example 6: Industry-Specific Tiers
# ============================================================================

def example_industry_tiers():
    """
    Create tiers for specific industries.
    """

    print("\n" + "=" * 70)
    print("🏭 EXAMPLE 6: Industry-Specific Custom Tiers")
    print("=" * 70)

    # Healthcare: Patient history tier
    patient_tier = (
        SearchTierBuilder("patient_history")
        .hybrid()
        .top_k(10)
        .depth(3)  # Traverse related conditions, medications, etc.
        .recency_boost(0.6)  # Recent visits more relevant
        .score_threshold(0.8)  # High precision for healthcare
        .filter(record_type="clinical")
        .build()
    )
    register_tier(patient_tier)
    print(f"\n🏥 Healthcare - Patient History:")
    print(f"   {patient_tier}")

    # Legal: Case law tier
    legal_tier = (
        SearchTierBuilder("case_law")
        .hybrid()
        .top_k(20)
        .depth(4)  # Deep citation network
        .recency_boost(0.2)  # Older cases still relevant
        .score_threshold(0.85)  # Very high precision
        .build()
    )
    register_tier(legal_tier)
    print(f"\n⚖️  Legal - Case Law:")
    print(f"   {legal_tier}")

    # Finance: Transaction analysis tier
    finance_tier = (
        SearchTierBuilder("transaction_analysis")
        .hybrid()
        .top_k(50)  # Lots of transactions
        .depth(2)  # Related transactions
        .recency_boost(0.8)  # Recent transactions critical
        .score_threshold(0.6)
        .filter(status="completed")
        .build()
    )
    register_tier(finance_tier)
    print(f"\n💰 Finance - Transaction Analysis:")
    print(f"   {finance_tier}")


# ============================================================================
# Example 7: Dynamic Tier Selection
# ============================================================================

def example_dynamic_selection():
    """
    Show how to dynamically choose tiers based on context.
    """

    print("\n" + "=" * 70)
    print("🎯 EXAMPLE 7: Dynamic Tier Selection")
    print("=" * 70)

    # Load all presets
    load_preset("ecommerce")
    load_preset("support")
    load_preset("research")

    def choose_tier_for_question(question: str, context: str) -> str:
        """
        Intelligently choose tier based on question + context.
        """

        q = question.lower()

        # E-commerce patterns
        if "product" in q or "buy" in q or "price" in q:
            if "similar" in q or "recommend" in q:
                return "recommendation"
            else:
                return "product_lookup"

        # Support patterns
        elif "help" in q or "issue" in q or "problem" in q:
            if "history" in q or "previous" in q:
                return "ticket_history"
            elif any(word in q for word in ["complex", "debug", "investigate"]):
                return "deep_investigation"
            else:
                return "quick_answer"

        # Research patterns
        elif "paper" in q or "research" in q or "study" in q:
            if "cite" in q or "reference" in q:
                return "citation_network"
            elif "review" in q or "survey" in q:
                return "literature_review"
            else:
                return "quick_fact"

        # Default
        return "balanced"

    # Test questions
    test_cases = [
        ("What's the price of this product?", "ecommerce"),
        ("Show me similar products", "ecommerce"),
        ("Help! My order is stuck", "support"),
        ("Investigate this complex billing issue", "support"),
        ("What papers cite the BERT paper?", "research"),
        ("Quick question: who invented LSTM?", "research"),
    ]

    for question, context in test_cases:
        chosen_tier = choose_tier_for_question(question, context)
        tier = get_tier(chosen_tier)
        print(f"\nQ: '{question}'")
        print(f"   → Tier: {chosen_tier}")
        print(f"   → Config: mode={tier.mode.value}, top_k={tier.vector_top_k}, graph={tier.enable_graph}")


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Run all examples."""

    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "CUSTOM SEARCH TIERS EXAMPLES" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")

    # Run examples
    example_ecommerce()
    example_customer_support()
    example_research()
    example_supermemory_style()
    example_custom_tier()
    example_industry_tiers()
    example_dynamic_selection()

    # Summary
    print("\n" + "=" * 70)
    print("✅ SUMMARY")
    print("=" * 70)
    print("\nYou can create custom search tiers for:")
    print("  1. Different industries (e-commerce, healthcare, legal, etc.)")
    print("  2. Different use cases (FAQ, debugging, recommendations, etc.)")
    print("  3. Different performance needs (fast/precise vs slow/comprehensive)")
    print("  4. Different recency requirements (hot/cold like Supermemory)")
    print("\nAll tiers are fully customizable:")
    print("  - Vector-only, graph-only, or hybrid")
    print("  - Custom top-k values")
    print("  - Custom graph traversal depth")
    print("  - Custom recency bias (0.0 - 1.0)")
    print("  - Custom score thresholds")
    print("  - Custom metadata filters")
    print("  - Custom ranking functions")
    print("\nThis is MORE flexible than:")
    print("  - Mem0 (vector-only, no customization)")
    print("  - Mem0g (fixed hybrid strategy)")
    print("  - Supermemory (time-based tiers only)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
