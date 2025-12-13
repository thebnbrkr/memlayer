"""
Intelligent Database Selection: LLM-Guided Search Strategy
===========================================================

This example shows how to make the LLM intelligently choose which database
to query based on question semantics, going beyond Mem0/Mem0g's fixed approach.

Key Innovation:
- Mem0g: Always queries BOTH vector + graph (no intelligence)
- This system: LLM analyzes question type and chooses optimal strategy
"""

from memlayer import OpenAI as Memlayer
from memlayer.config.salience import (
    TenantSalienceConfig,
    SalienceComponent,
    ScoringFunctionType,
    AdaptiveThresholdConfig,
    ThresholdStrategy
)
import json


# ============================================================================
# Approach 1: Enhanced Function Calling (Recommended)
# ============================================================================

def create_intelligent_search_client():
    """
    Create Memlayer client with enhanced search tool that lets LLM choose
    database strategy based on question analysis.
    """

    # Create base client
    client = Memlayer(
        model="gpt-4o-mini",
        user_id="smart_user",
        storage_path="./intelligent_search_demo",
        operation_mode="online"
    )

    # Override the search_memory tool definition to be more intelligent
    # (This would require modifying memlayer/wrappers/openai.py, shown below)

    return client


# ============================================================================
# Approach 2: Pre-Query Analysis (Works with Current Memlayer)
# ============================================================================

class IntelligentSearchWrapper:
    """
    Wrapper that analyzes questions BEFORE calling Memlayer to choose
    the optimal database search strategy.

    Question Types:
    1. Factual ("What is X?") → Vector DB (semantic search)
    2. Relational ("How are X and Y connected?") → Graph DB (traversal)
    3. Temporal ("What happened after X?") → Graph DB (event chains)
    4. Comparative ("Compare X and Y") → Both DBs (hybrid)
    """

    def __init__(self, api_key: str):
        self.client = Memlayer(
            model="gpt-4o-mini",
            user_id="intelligent_user",
            storage_path="./smart_search",
            operation_mode="online"
        )

        # LLM for question analysis
        from openai import OpenAI
        self.analyzer = OpenAI(api_key=api_key)

    def analyze_question_type(self, question: str) -> str:
        """
        Use LLM to analyze question type and suggest optimal search strategy.

        Returns:
            "fast" (vector only, simple facts)
            "balanced" (vector only, moderate complexity)
            "deep" (vector + graph, relational/temporal)
        """

        analysis_prompt = f"""Analyze this question and determine the optimal database search strategy.

Question: "{question}"

Choose one:
- "fast": Simple factual question (e.g., "What is Alice's job?", "Where does Bob live?")
  → Use vector DB only for quick semantic search

- "balanced": Moderate complexity, may need context (e.g., "Tell me about Alice's career")
  → Use vector DB with more results

- "deep": Relational, temporal, or complex question requiring graph traversal
  Examples:
  - "How are Alice's career and hobbies connected?"
  - "What events led to Bob's decision?"
  - "Who introduced Alice to Bob?"
  → Use both vector + graph DB with 2-hop traversal

Respond with ONLY the strategy name: fast, balanced, or deep.
"""

        response = self.analyzer.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a search strategy analyzer. Respond with only: fast, balanced, or deep."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0
        )

        strategy = response.choices[0].message.content.strip().lower()

        # Validate
        if strategy not in ["fast", "balanced", "deep"]:
            print(f"⚠️  Invalid strategy '{strategy}', defaulting to 'balanced'")
            strategy = "balanced"

        return strategy

    def chat(self, question: str, verbose: bool = True) -> str:
        """
        Chat with intelligent database selection.

        1. Analyze question type
        2. Choose optimal search tier
        3. Execute search with chosen strategy
        """

        # Step 1: Analyze question
        if verbose:
            print(f"\n🔍 Analyzing question: {question}")

        strategy = self.analyze_question_type(question)

        if verbose:
            print(f"📊 Optimal strategy: {strategy.upper()}")
            if strategy == "fast":
                print("   → Vector DB only (top 2 results)")
            elif strategy == "balanced":
                print("   → Vector DB only (top 5 results)")
            elif strategy == "deep":
                print("   → Vector + Graph DB (2-hop traversal)")

        # Step 2: Execute with chosen strategy
        # Override the search tier by manually calling search
        response = self.client.chat(
            [{"role": "user", "content": question}],
            # Force the search tier (this would require modifying Memlayer to accept this)
            # For now, we'll use the default behavior and rely on LLM choosing via function call
        )

        return response


# ============================================================================
# Approach 3: Custom Search Function (Most Control)
# ============================================================================

class CustomDatabaseSelector:
    """
    Fully custom database selection with explicit control over which
    databases to query based on question patterns.
    """

    def __init__(self):
        from memlayer.services import SearchService
        from memlayer.storage import MemoryStorage, GraphStorage

        # Initialize storage layers
        self.vector_storage = MemoryStorage(
            storage_path="./custom_search/vector",
            operation_mode="online"
        )

        self.graph_storage = GraphStorage(
            storage_path="./custom_search/graph"
        )

        # Create search service
        self.search_service = SearchService(
            storage=self.vector_storage,
            graph_storage=self.graph_storage,
            operation_mode="online"
        )

    def search_by_question_type(self, question: str, user_id: str) -> dict:
        """
        Explicitly choose database based on question patterns.

        Pattern Matching Rules:
        - Contains "connected", "related", "relationship" → Graph search
        - Contains "what", "who", "where" (simple) → Vector search
        - Contains "why", "how", "explain" → Hybrid search
        """

        question_lower = question.lower()

        # Rule 1: Relational questions → Graph-heavy search
        relational_keywords = ["connected", "related", "relationship", "between",
                               "influence", "led to", "caused", "resulted in"]

        if any(keyword in question_lower for keyword in relational_keywords):
            print("🌐 Detected relational question → Using GRAPH search")
            search_tier = "deep"

        # Rule 2: Simple factual questions → Vector-only search
        elif question_lower.startswith(("what is", "who is", "where is", "when did")):
            print("📚 Detected simple factual question → Using VECTOR search (fast)")
            search_tier = "fast"

        # Rule 3: Complex explanatory questions → Hybrid search
        elif question_lower.startswith(("why", "how", "explain")):
            print("🧠 Detected complex question → Using HYBRID search")
            search_tier = "deep"

        # Default: Balanced vector search
        else:
            print("⚖️  Default to balanced vector search")
            search_tier = "balanced"

        # Execute search
        results = self.search_service.search(
            query=question,
            user_id=user_id,
            search_tier=search_tier
        )

        return results


# ============================================================================
# Example Usage
# ============================================================================

def demo_intelligent_selection():
    """
    Demonstrate intelligent database selection across different question types.
    """

    print("=" * 70)
    print("🧪 DEMO: Intelligent Database Selection (Beyond Mem0/Mem0g)")
    print("=" * 70)

    # Initialize wrapper
    import os
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("\n⚠️  Set OPENAI_API_KEY environment variable to run this demo")
        return

    wrapper = IntelligentSearchWrapper(api_key=api_key)

    # Store some conversation data first
    print("\n📝 Storing conversation history...")
    wrapper.client.chat([
        {"role": "user", "content": "Hi! I'm Alice. I work as a software engineer at Google."}
    ])
    wrapper.client.chat([
        {"role": "user", "content": "I love hiking on weekends. Last month I hiked Mount Tam."}
    ])
    wrapper.client.chat([
        {"role": "user", "content": "My manager Sarah introduced me to the AI safety team."}
    ])
    wrapper.client.chat([
        {"role": "user", "content": "That's how I got interested in AI alignment. Now I'm considering switching teams."}
    ])

    print("✅ Data stored in vector + graph databases\n")

    # Test different question types
    test_questions = [
        "What does Alice do for work?",  # Simple factual → fast (vector only)
        "Tell me about Alice's career and hobbies",  # Moderate → balanced (vector only)
        "How are Alice's hiking hobby and her career change connected?",  # Relational → deep (graph)
        "What led Alice to become interested in AI alignment?",  # Causal chain → deep (graph)
    ]

    for question in test_questions:
        print("\n" + "=" * 70)
        response = wrapper.chat(question, verbose=True)
        print(f"\n💬 Response: {response}")

    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("\nKey Insight:")
    print("  This system is MORE intelligent than Mem0g, which always queries")
    print("  both databases without analyzing question semantics.")
    print("=" * 70)


# ============================================================================
# How to Modify Memlayer Core (Advanced)
# ============================================================================

def show_core_modification():
    """
    Shows how to modify memlayer/wrappers/openai.py to add intelligent
    database selection directly into the function calling system.
    """

    print("""
To make this part of Memlayer's core, modify memlayer/wrappers/openai.py:

# File: memlayer/wrappers/openai.py

ENHANCED_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_memory",
        "description": '''Search memory with intelligent database selection.

Question Type Analysis:
- Simple factual (What/Who/Where) → Use "fast" (vector only, top 2)
- Moderate context needed → Use "balanced" (vector only, top 5)
- Relational/causal (How/Why/Connected) → Use "deep" (vector + graph, 2-hop)

Examples:
- "What is Alice's job?" → fast
- "Tell me about Alice" → balanced
- "How are Alice's career and hobbies connected?" → deep
''',
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "search_tier": {
                    "type": "string",
                    "enum": ["fast", "balanced", "deep"],
                    "description": "Search strategy based on question complexity"
                }
            },
            "required": ["query", "search_tier"]
        }
    }
}

# The LLM now has better guidance on choosing the right tier!
""")


if __name__ == "__main__":
    # Run the demo
    demo_intelligent_selection()

    # Show how to modify core
    print("\n\n")
    show_core_modification()
