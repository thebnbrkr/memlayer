"""
LoComo Dataset Download Instructions
=====================================

The LoComo (Long-term Conversation Memory) dataset is from the ACL 2024 paper.

Option 1: Download from Official Repo
--------------------------------------
```bash
# Clone the LoComo repository
git clone https://github.com/snap-research/locomo.git

# Navigate to data folder
cd locomo/data

# The dataset files:
# - locomo_train.json (training set)
# - locomo_test.json (test set)
# - locomo_sample.json (sample for quick testing)
```

Option 2: Manual Download
--------------------------
Visit: https://github.com/snap-research/locomo
Download: data/locomo_sample.json or data/locomo_test.json


Option 3: Use Example Data (for Colab)
---------------------------------------
If the dataset isn't available, create a sample file:
"""

import json

# Create sample LoComo-style data
sample_data = [
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
                "ground_truth_answer": "Alice is a software engineer at Google."
            },
            {
                "question": "What hobby does Alice enjoy?",
                "ground_truth_answer": "Alice loves hiking on weekends."
            },
            {
                "question": "Who introduced Alice to the AI alignment team?",
                "ground_truth_answer": "Sarah, Alice's manager, introduced her to the AI alignment team."
            },
            {
                "question": "What is Alice considering doing?",
                "ground_truth_answer": "Alice is considering switching from the ads team to the AI safety team."
            }
        ]
    },
    {
        "conversation_id": "conv_002",
        "conversation": [
            "I'm Bob, working in cloud infrastructure at Google.",
            "I went to Stanford with Alice.",
            "I enjoy playing tennis at the local club.",
            "I'm working on improving database performance for large-scale systems.",
            "My team just launched a new caching layer that reduced latency by 40%."
        ],
        "qa_pairs": [
            {
                "question": "Where does Bob work?",
                "ground_truth_answer": "Bob works at Google in cloud infrastructure."
            },
            {
                "question": "What is Bob's hobby?",
                "ground_truth_answer": "Bob enjoys playing tennis at the local club."
            },
            {
                "question": "What did Bob's team recently accomplish?",
                "ground_truth_answer": "Bob's team launched a new caching layer that reduced latency by 40%."
            }
        ]
    }
]

# Save to file
with open('locomo_sample.json', 'w') as f:
    json.dump(sample_data, f, indent=2)

print("✅ Created locomo_sample.json with 2 sample conversations")
print(f"   - Conversation 1: {len(sample_data[0]['conversation'])} turns, {len(sample_data[0]['qa_pairs'])} QA pairs")
print(f"   - Conversation 2: {len(sample_data[1]['conversation'])} turns, {len(sample_data[1]['qa_pairs'])} QA pairs")
