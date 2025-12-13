"""
Load Real LoComo Dataset
=========================

This script downloads and processes the real LoComo dataset from the official repo.
"""

import json
import requests

def download_locomo_dataset(url: str = None) -> list:
    """
    Download LoComo dataset from official repo.

    Available datasets:
    - locomo10.json (10 conversations)
    - locomo_sample.json (sample)
    - full dataset (larger files)
    """

    if url is None:
        # Default to locomo10.json (good balance of speed vs coverage)
        url = "https://raw.githubusercontent.com/snap-research/locomo/refs/heads/main/data/locomo10.json"

    print(f"📥 Downloading LoComo dataset from:\n   {url}\n")

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Downloaded successfully!")
        return data
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nTrying alternative URLs...")

        # Try alternative URLs
        alternatives = [
            "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
            "https://github.com/snap-research/locomo/raw/main/data/locomo10.json"
        ]

        for alt_url in alternatives:
            try:
                print(f"   Trying: {alt_url}")
                response = requests.get(alt_url)
                response.raise_for_status()
                data = response.json()
                print(f"   ✅ Success!")
                return data
            except:
                continue

        print("\n❌ All download attempts failed. Using sample data instead.")
        return None


def convert_locomo_to_benchmark_format(locomo_data: list) -> list:
    """
    Convert real LoComo format to our benchmark format.

    LoComo format:
    {
      "qa": [{"question": "...", "answer": "...", "evidence": ["D1:3"], "category": 2}],
      "dialogue": [{"turn": 0, "speaker": "User", "text": "..."}]
    }

    Our format:
    {
      "conversation_id": "conv_001",
      "conversation": ["message1", "message2"],
      "qa_pairs": [{"question": "...", "ground_truth_answer": "..."}]
    }
    """

    converted = []

    for idx, item in enumerate(locomo_data):
        # Extract dialogue as conversation
        conversation = []
        if "dialogue" in item:
            for turn in item["dialogue"]:
                # Combine speaker and text
                text = turn.get("text", "")
                conversation.append(text)

        # Extract QA pairs
        qa_pairs = []
        if "qa" in item:
            for qa in item["qa"]:
                qa_pairs.append({
                    "question": qa.get("question", ""),
                    "ground_truth_answer": qa.get("answer", ""),
                    "evidence": qa.get("evidence", []),
                    "category": qa.get("category", 0)
                })

        converted.append({
            "conversation_id": f"conv_{idx:03d}",
            "conversation": conversation,
            "qa_pairs": qa_pairs
        })

    return converted


def load_and_prepare_locomo(num_conversations: int = 5, use_real_data: bool = True) -> list:
    """
    Load and prepare LoComo dataset for benchmarking.

    Args:
        num_conversations: Number of conversations to use (for quick testing)
        use_real_data: If True, download real data; if False, use sample data

    Returns:
        List of conversations in benchmark format
    """

    if use_real_data:
        # Download real LoComo data
        locomo_data = download_locomo_dataset()

        if locomo_data is None:
            print("\n⚠️  Falling back to sample data")
            use_real_data = False
        else:
            # Convert to benchmark format
            print(f"\n🔄 Converting to benchmark format...")
            conversations = convert_locomo_to_benchmark_format(locomo_data)
            print(f"✅ Converted {len(conversations)} conversations")

    if not use_real_data:
        # Use sample data
        print("\n📝 Creating sample data...")
        conversations = create_sample_data()

    # Limit to requested number
    if num_conversations > 0:
        conversations = conversations[:num_conversations]
        print(f"\n📊 Using {len(conversations)} conversations for benchmark")

    # Print stats
    total_qa = sum(len(c['qa_pairs']) for c in conversations)
    avg_turns = sum(len(c['conversation']) for c in conversations) / len(conversations)

    print(f"\nDataset statistics:")
    print(f"  - Conversations: {len(conversations)}")
    print(f"  - Total QA pairs: {total_qa}")
    print(f"  - Avg conversation length: {avg_turns:.1f} turns")

    return conversations


def create_sample_data() -> list:
    """Fallback sample data if download fails."""

    return [
        {
            "conversation_id": "conv_001",
            "conversation": [
                "Hi! I'm Alice, a software engineer at Google working on AI safety.",
                "I love hiking on weekends. Last month I climbed Mount Tamalpais.",
                "My manager Sarah introduced me to the AI alignment team.",
                "That introduction got me really interested in AI ethics and safety.",
                "Now I'm considering switching from the ads team to the AI safety team."
            ],
            "qa_pairs": [
                {
                    "question": "What is Alice's job?",
                    "ground_truth_answer": "Alice is a software engineer at Google.",
                    "category": 1
                },
                {
                    "question": "What hobby does Alice enjoy?",
                    "ground_truth_answer": "Alice loves hiking on weekends.",
                    "category": 1
                },
                {
                    "question": "Who introduced Alice to the AI alignment team?",
                    "ground_truth_answer": "Sarah, Alice's manager, introduced her to the AI alignment team.",
                    "category": 2
                },
                {
                    "question": "What is Alice considering doing?",
                    "ground_truth_answer": "Alice is considering switching from the ads team to the AI safety team.",
                    "category": 2
                }
            ]
        }
    ]


# Example usage
if __name__ == "__main__":
    # Load real LoComo dataset
    conversations = load_and_prepare_locomo(
        num_conversations=5,  # Use first 5 for quick testing
        use_real_data=True    # Set to False to use sample data
    )

    # Show first conversation
    print("\n" + "="*70)
    print("SAMPLE CONVERSATION")
    print("="*70)
    conv = conversations[0]
    print(f"\nID: {conv['conversation_id']}")
    print(f"Turns: {len(conv['conversation'])}")
    print(f"\nFirst 3 turns:")
    for i, turn in enumerate(conv['conversation'][:3]):
        print(f"  {i+1}. {turn[:80]}...")
    print(f"\nQA pairs ({len(conv['qa_pairs'])} total):")
    for qa in conv['qa_pairs'][:2]:
        print(f"  Q: {qa['question']}")
        print(f"  A: {qa['ground_truth_answer']}")
        print()
