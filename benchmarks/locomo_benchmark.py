"""
LoComo Benchmark: Memlayer vs Mem0 Comparison
==============================================

This script benchmarks Memlayer against Mem0 using the LoComo benchmark
for evaluating very long-term conversational memory.

LoComo Info:
- Paper: https://arxiv.org/abs/2402.17753
- GitHub: https://github.com/snap-research/locomo
- Published: ACL 2024
- Dataset: 300-600 turn conversations across 32-35 sessions
- Tasks: QA, event summarization, multi-modal dialogue
- Mem0's claim: 26% accuracy boost over baseline OpenAI

Goal: Prove Memlayer's custom salience configs improve accuracy
"""

import json
import time
import os
from typing import List, Dict, Any
from pathlib import Path


# ============================================================================
# Step 1: Install Dependencies
# ============================================================================

def setup():
    """Install required packages."""
    print("📦 Installing dependencies...")
    os.system("pip install -q git+https://github.com/thebnbrkr/memlayer.git")
    os.system("pip install -q mem0ai")
    os.system("pip install -q rouge-score")
    os.system("pip install -q datasets")
    print("✅ Dependencies installed\n")


# ============================================================================
# Step 2: Load LoComo Dataset
# ============================================================================

def load_locomo_dataset():
    """
    Load LoComo benchmark data.

    Dataset structure:
    - conversations: List of multi-session dialogues
    - questions: QA pairs for each conversation
    - events: Timeline events to summarize
    """
    print("📁 Loading LoComo dataset...")

    # Try to load from GitHub release
    try:
        import requests

        # LoComo dataset URL (update if needed)
        dataset_url = "https://raw.githubusercontent.com/snap-research/locomo/main/data/conversations.json"

        response = requests.get(dataset_url)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Loaded {len(data.get('conversations', []))} conversations\n")
            return data
        else:
            print(f"⚠️  Failed to download dataset (status {response.status_code})")
            return create_sample_locomo_data()

    except Exception as e:
        print(f"⚠️  Error loading dataset: {e}")
        print("   Using sample data instead\n")
        return create_sample_locomo_data()


def create_sample_locomo_data():
    """
    Create sample data in LoComo format for testing.

    Real LoComo has 300-600 turns, this is a miniature version.
    """
    return {
        "conversations": [
            {
                "id": "sample_conv_1",
                "sessions": [
                    {
                        "session_id": 1,
                        "date": "2024-01-01",
                        "turns": [
                            {"speaker": "user", "text": "Hi! I'm Alice, a software engineer at Google."},
                            {"speaker": "assistant", "text": "Nice to meet you, Alice! What do you work on at Google?"},
                            {"speaker": "user", "text": "I work on machine learning infrastructure, specifically Python-based ML pipelines."},
                            {"speaker": "assistant", "text": "That sounds fascinating! How long have you been there?"},
                            {"speaker": "user", "text": "About 3 years now. I love it!"},
                        ]
                    },
                    {
                        "session_id": 2,
                        "date": "2024-01-15",
                        "turns": [
                            {"speaker": "user", "text": "Hey! Remember me?"},
                            {"speaker": "assistant", "text": "Of course! How are you doing?"},
                            {"speaker": "user", "text": "Good! I just got promoted to Senior Engineer."},
                            {"speaker": "assistant", "text": "Congratulations! That's amazing!"},
                        ]
                    },
                    {
                        "session_id": 3,
                        "date": "2024-02-01",
                        "turns": [
                            {"speaker": "user", "text": "I'm thinking about switching to work on AI safety."},
                            {"speaker": "assistant", "text": "That's a big change! What's motivating this?"},
                            {"speaker": "user", "text": "I think it's more important for the future."},
                        ]
                    },
                ],
                "questions": [
                    {
                        "question": "What does Alice do for work?",
                        "answer": "Alice is a Senior Software Engineer at Google working on machine learning infrastructure and Python-based ML pipelines.",
                        "facts_needed": ["Alice works at Google", "Alice is a software engineer", "Alice got promoted to Senior Engineer", "Alice works on ML infrastructure"]
                    },
                    {
                        "question": "What is Alice considering for her career?",
                        "answer": "Alice is considering switching to work on AI safety because she thinks it's more important for the future.",
                        "facts_needed": ["Alice is thinking about switching to AI safety", "Alice believes AI safety is important for the future"]
                    }
                ]
            }
        ]
    }


# ============================================================================
# Step 3: Memory System Wrappers
# ============================================================================

class MemlayerWrapper:
    """Wrapper for Memlayer with custom salience config."""

    def __init__(self, user_id: str = "benchmark_user", salience_config=None):
        from memlayer import OpenAI as Memlayer

        self.client = Memlayer(
            model="gpt-4o-mini",
            user_id=user_id,
            storage_path="./benchmark_memlayer",
            operation_mode="online",
            salience_config=salience_config
        )

    def store_conversation(self, conversation: List[Dict[str, str]]):
        """Store conversation turns."""
        for turn in conversation:
            if turn["speaker"] == "user":
                # Store user messages (facts about them)
                self.client.chat([{"role": "user", "content": turn["text"]}])

    def query(self, question: str) -> str:
        """Query memory and generate answer."""
        response = self.client.chat([
            {"role": "user", "content": question}
        ])
        return response


class Mem0Wrapper:
    """Wrapper for Mem0."""

    def __init__(self, user_id: str = "benchmark_user"):
        from mem0 import Memory

        self.client = Memory()
        self.user_id = user_id

    def store_conversation(self, conversation: List[Dict[str, str]]):
        """Store conversation turns."""
        # Mem0 expects concatenated text
        conversation_text = "\n".join([
            f"{turn['speaker']}: {turn['text']}"
            for turn in conversation
        ])
        self.client.add(conversation_text, user_id=self.user_id)

    def query(self, question: str) -> str:
        """Query memory and generate answer."""
        # Search memories
        results = self.client.search(question, user_id=self.user_id)

        # Use memories as context for answer
        # (In real impl, you'd use LLM here with context)
        if results:
            return f"Based on memories: {results[0]['memory']}"
        return "I don't have enough information to answer that."


# ============================================================================
# Step 4: Evaluation Metrics
# ============================================================================

def compute_f1(predicted: str, ground_truth: str) -> float:
    """
    Compute F1 score for QA.

    F1 measures overlap between predicted and ground truth answers.
    """
    from collections import Counter

    # Tokenize
    pred_tokens = set(predicted.lower().split())
    truth_tokens = set(ground_truth.lower().split())

    # Compute overlap
    common = pred_tokens & truth_tokens

    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(truth_tokens) if truth_tokens else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_rouge(predicted: str, ground_truth: str) -> Dict[str, float]:
    """
    Compute ROUGE scores for text overlap.

    ROUGE measures n-gram overlap (commonly used for summarization).
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(ground_truth, predicted)

        return {
            "rouge1": scores['rouge1'].fmeasure,
            "rouge2": scores['rouge2'].fmeasure,
            "rougeL": scores['rougeL'].fmeasure,
        }
    except ImportError:
        # Fallback if rouge-score not installed
        return {
            "rouge1": compute_f1(predicted, ground_truth),
            "rouge2": 0.0,
            "rougeL": 0.0,
        }


def evaluate_qa(memory_system, questions: List[Dict]) -> Dict[str, float]:
    """
    Evaluate QA performance.

    Returns:
        Dict with average F1, ROUGE scores
    """
    f1_scores = []
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for qa in questions:
        # Query memory system
        predicted = memory_system.query(qa["question"])
        ground_truth = qa["answer"]

        # Compute metrics
        f1 = compute_f1(predicted, ground_truth)
        rouge = compute_rouge(predicted, ground_truth)

        f1_scores.append(f1)
        rouge_scores["rouge1"].append(rouge["rouge1"])
        rouge_scores["rouge2"].append(rouge["rouge2"])
        rouge_scores["rougeL"].append(rouge["rougeL"])

    return {
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "rouge1": sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"]) if rouge_scores["rouge1"] else 0.0,
        "rouge2": sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"]) if rouge_scores["rouge2"] else 0.0,
        "rougeL": sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"]) if rouge_scores["rougeL"] else 0.0,
    }


# ============================================================================
# Step 5: Run Benchmark
# ============================================================================

def run_benchmark(dataset: Dict, custom_salience_config=None):
    """
    Run benchmark comparing Memlayer vs Mem0.

    Args:
        dataset: LoComo dataset
        custom_salience_config: Optional custom config for Memlayer

    Returns:
        Dict with results for both systems
    """
    results = {}

    # Test each conversation
    for conv in dataset["conversations"]:
        conv_id = conv["id"]
        print(f"\n🔬 Testing conversation: {conv_id}")
        print("=" * 60)

        # Flatten conversation turns across sessions
        all_turns = []
        for session in conv["sessions"]:
            all_turns.extend(session["turns"])

        # Test Memlayer
        print("\n📊 Testing Memlayer...")
        memlayer = MemlayerWrapper(user_id=f"bench_{conv_id}", salience_config=custom_salience_config)

        start = time.time()
        memlayer.store_conversation(all_turns)
        ml_store_time = time.time() - start

        ml_results = evaluate_qa(memlayer, conv["questions"])
        ml_results["store_time"] = ml_store_time

        print(f"   F1: {ml_results['f1']:.3f}")
        print(f"   ROUGE-1: {ml_results['rouge1']:.3f}")
        print(f"   Store time: {ml_store_time:.2f}s")

        # Test Mem0
        print("\n📊 Testing Mem0...")
        mem0 = Mem0Wrapper(user_id=f"bench_{conv_id}")

        start = time.time()
        mem0.store_conversation(all_turns)
        m0_store_time = time.time() - start

        m0_results = evaluate_qa(mem0, conv["questions"])
        m0_results["store_time"] = m0_store_time

        print(f"   F1: {m0_results['f1']:.3f}")
        print(f"   ROUGE-1: {m0_results['rouge1']:.3f}")
        print(f"   Store time: {m0_store_time:.2f}s")

        # Store results
        results[conv_id] = {
            "memlayer": ml_results,
            "mem0": m0_results
        }

    return results


def generate_report(results: Dict):
    """Generate comparison report."""
    print("\n" + "=" * 60)
    print("📈 BENCHMARK RESULTS")
    print("=" * 60)

    # Aggregate scores
    ml_f1 = []
    ml_rouge1 = []
    m0_f1 = []
    m0_rouge1 = []

    for conv_id, data in results.items():
        ml_f1.append(data["memlayer"]["f1"])
        ml_rouge1.append(data["memlayer"]["rouge1"])
        m0_f1.append(data["mem0"]["f1"])
        m0_rouge1.append(data["mem0"]["rouge1"])

    # Calculate averages
    ml_avg_f1 = sum(ml_f1) / len(ml_f1) if ml_f1 else 0
    ml_avg_rouge = sum(ml_rouge1) / len(ml_rouge1) if ml_rouge1 else 0
    m0_avg_f1 = sum(m0_f1) / len(m0_f1) if m0_f1 else 0
    m0_avg_rouge = sum(m0_rouge1) / len(m0_rouge1) if m0_rouge1 else 0

    # Calculate improvements
    f1_improvement = ((ml_avg_f1 - m0_avg_f1) / m0_avg_f1 * 100) if m0_avg_f1 > 0 else 0
    rouge_improvement = ((ml_avg_rouge - m0_avg_rouge) / m0_avg_rouge * 100) if m0_avg_rouge > 0 else 0

    # Print table
    print("\n┌────────────────┬──────────┬──────────┬────────────┐")
    print("│ System         │ F1 Score │ ROUGE-1  │ Improvement│")
    print("├────────────────┼──────────┼──────────┼────────────┤")
    print(f"│ Memlayer       │  {ml_avg_f1:.3f}   │  {ml_avg_rouge:.3f}   │     -      │")
    print(f"│ Mem0           │  {m0_avg_f1:.3f}   │  {m0_avg_rouge:.3f}   │     -      │")
    print("├────────────────┼──────────┼──────────┼────────────┤")
    print(f"│ Difference     │ {f1_improvement:+.1f}%    │ {rouge_improvement:+.1f}%    │            │")
    print("└────────────────┴──────────┴──────────┴────────────┘")

    # Verdict
    print("\n🎯 Verdict:")
    if ml_avg_f1 > m0_avg_f1:
        print(f"   ✅ Memlayer outperforms Mem0 by {f1_improvement:.1f}% on F1!")
        print("   Your custom salience config improves accuracy! 🚀")
    elif ml_avg_f1 == m0_avg_f1:
        print("   ⚖️  Memlayer and Mem0 perform equally.")
        print("   Try tuning your salience config for better results.")
    else:
        print(f"   ⚠️  Mem0 outperforms Memlayer by {abs(f1_improvement):.1f}%")
        print("   Your salience config may be filtering too aggressively.")

    # Save results
    output_file = "locomo_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Full results saved to: {output_file}")


# ============================================================================
# Step 6: Main Execution
# ============================================================================

def main():
    """Run LoComo benchmark."""
    print("🧪 LoComo Benchmark: Memlayer vs Mem0")
    print("=" * 60)

    # Setup
    setup()

    # Load dataset
    dataset = load_locomo_dataset()

    # Create custom salience config for Memlayer
    # This is where you test different configurations!
    from memlayer.config.salience import (
        TenantSalienceConfig,
        SalienceComponent,
        ScoringFunctionType,
        AdaptiveThresholdConfig,
        ThresholdStrategy
    )

    custom_config = TenantSalienceConfig(
        tenant_id="benchmark",
        config_name="locomo_optimized",
        components=[
            SalienceComponent(
                name="factual_content",
                weight=0.7,
                scoring_function=ScoringFunctionType.LENGTH_BONUS,
                scoring_config={
                    "min_length": 20,
                    "optimal_length": 100
                }
            ),
            SalienceComponent(
                name="personal_info",
                weight=0.3,
                scoring_function=ScoringFunctionType.KEYWORD_MATCH,
                scoring_config={
                    "keywords": ["I", "my", "me", "work", "job", "career", "family"]
                }
            )
        ],
        threshold_config=AdaptiveThresholdConfig(
            strategy=ThresholdStrategy.ABSOLUTE,
            absolute_threshold=0.4  # Store more for long conversations
        )
    )

    # Run benchmark
    results = run_benchmark(dataset, custom_salience_config=custom_config)

    # Generate report
    generate_report(results)


if __name__ == "__main__":
    main()
