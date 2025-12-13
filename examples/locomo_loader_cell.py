"""
Updated Cell for Loading Real LoComo Dataset
=============================================

Replace the "Load LoComo Dataset" cell in the notebook with this code.
"""

# Cell for Colab notebook:

"""
## Load Real LoComo Dataset

This cell downloads and processes the real LoComo dataset.
"""

import json
import requests

def download_locomo_dataset(dataset_name: str = "locomo10.json") -> list:
    """Download real LoComo dataset from GitHub."""

    base_url = "https://raw.githubusercontent.com/snap-research/locomo/refs/heads/main/data/"
    url = base_url + dataset_name

    print(f"📥 Downloading {dataset_name}...")
    print(f"   URL: {url}\n")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Downloaded successfully! ({len(data)} conversations)")
        return data
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def convert_locomo_format(locomo_data: list) -> list:
    """
    Convert real LoComo format to benchmark format.

    Real LoComo:
    {
      "qa": [{"question": "...", "answer": "...", "evidence": ["D1:3"], "category": 2}],
      "dialogue": [{"turn": 0, "speaker": "User", "text": "..."}]
    }

    Benchmark format:
    {
      "conversation": ["msg1", "msg2"],
      "qa_pairs": [{"question": "...", "ground_truth_answer": "..."}]
    }
    """

    converted = []

    for idx, item in enumerate(locomo_data):
        # Extract conversation
        conversation = []
        if "dialogue" in item:
            for turn in item["dialogue"]:
                conversation.append(turn.get("text", ""))

        # Extract QA pairs
        qa_pairs = []
        if "qa" in item:
            for qa in item["qa"]:
                qa_pairs.append({
                    "question": qa.get("question", ""),
                    "ground_truth_answer": qa.get("answer", ""),  # ← Key change!
                    "evidence": qa.get("evidence", []),
                    "category": qa.get("category", 0)
                })

        converted.append({
            "conversation_id": f"conv_{idx:03d}",
            "conversation": conversation,
            "qa_pairs": qa_pairs
        })

    return converted

# Download real dataset
print("="*70)
print("LOADING LOCOMO DATASET")
print("="*70)

locomo_raw = download_locomo_dataset("locomo10.json")

if locomo_raw:
    # Convert to benchmark format
    print("\n🔄 Converting to benchmark format...")
    conversations = convert_locomo_format(locomo_raw)

    # Limit to first N for quick testing
    num_conversations = 5  # Change this to use more/less
    conversations = conversations[:num_conversations]

    print(f"✅ Prepared {len(conversations)} conversations for benchmark\n")

    # Show stats
    total_qa = sum(len(c['qa_pairs']) for c in conversations)
    avg_turns = sum(len(c['conversation']) for c in conversations) / len(conversations)

    print(f"📊 Dataset Statistics:")
    print(f"   - Conversations: {len(conversations)}")
    print(f"   - Total QA pairs: {total_qa}")
    print(f"   - Avg turns per conversation: {avg_turns:.1f}")

    # Show sample
    print(f"\n📝 Sample QA pair:")
    qa = conversations[0]['qa_pairs'][0]
    print(f"   Q: {qa['question']}")
    print(f"   A: {qa['ground_truth_answer']}")
    if 'category' in qa:
        print(f"   Category: {qa['category']}")

else:
    # Fallback to sample data
    print("\n⚠️  Using sample data instead")
    conversations = [
        {
            "conversation_id": "conv_001",
            "conversation": [
                "Hi! I'm Alice, a software engineer at Google.",
                "I love hiking on weekends.",
                "My manager Sarah introduced me to the AI team."
            ],
            "qa_pairs": [
                {
                    "question": "What is Alice's job?",
                    "ground_truth_answer": "Alice is a software engineer at Google.",
                    "category": 1
                }
            ]
        }
    ]

print("\n" + "="*70)
