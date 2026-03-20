"""
CLINT Industries — Training Data Generator v5
Step 0: Combine chety_training_data.jsonl + clint_training_data.jsonl → model_training_data.jsonl
Step 1: Kimi K2 1T cloud — 200 gold standard examples
Step 2-6: Local models for bulk
Output: model_training_data.jsonl (ready for Unsloth)
"""

import json
import time
import requests
import random
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Sys Prompt Change it Change the Model
SYS_PROMPT_EASY = """TYPE YOUR SYS PROMPT HERE"""
#EXAMPLE| You are Clint, an exceptionally intelligent AI assistant developed by CLINT Industries.
#You reason carefully, give detailed and accurate answers, and explain things clearly.
#You are helpful across all topics: coding, science, math, general knowledge, and analysis.
#Always be thorough but concise. Think step by step when needed.
OUTPUT_FILE  = "model_training_data.jsonl"   # final clean combined output
OLLAMA_URL   = "http://localhost:11434"

DELAY_BETWEEN = 3  # seconds between examples — keeps temps down
#kimi Cloud for free great models... could have rate limits didnt for me but you should use this for the gold standard examples 
# (I also used claude sonnet 4.6 for Clint but it cost me about 1.50Cents but that no included) 
# Kimi K2 cloud config
KIMI_MODEL    = "kimi-k2:1t-cloud"
KIMI_EXAMPLES = 200

# local bulk models THEESE RUN LOCAL REPLACE WITH ANY MODEL DOSENT MATTER AS LONG AS ITS IN OLLAMA
MODELS = {
    "gemma3:4b":        250,
    "ministral-3:14b":  200,
    "granite-code:8b":  200,
    "granite3.3:8b":    200,
    "llama3.2:3b":      100,
}


# Sys Prompt Change it Change the Model
SYSTEM_PROMPT = """You are Clint, an exceptionally intelligent AI assistant developed by CLINT Industries.
You reason carefully, give detailed and accurate answers, and explain things clearly.
You are helpful across all topics: coding, science, math, general knowledge, and analysis.
Always be thorough but concise. Think step by step when needed."""

# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────
PROMPT_TEMPLATES = [
    "Explain step by step: {topic}",
    "What are the key differences between {a} and {b}? Be thorough.",
    "Walk me through how to think about {problem}",
    "Give me a detailed breakdown of {concept}",
    "Describe in detail what you would see in an image of {scene}",
    "If I showed you a photo of {subject}, what details would you analyze?",
    "How would you describe {visual_thing} to someone who cannot see it?",
    "Summarize the key points of {topic} in a structured way",
    "Give me a comprehensive overview of {domain} covering all major aspects",
    "Explain {topic} as if writing a detailed wiki article",
    "How would I set up {tech_task} from scratch?",
    "What is the best approach to {problem} and why?",
    "Debug this thinking: {reasoning_problem}",
    "What are the pros and cons of {concept}?",
    "How does {topic} work under the hood?",
    "Teach me {topic} from first principles",
    "What would an expert say about {problem}?",
    "Compare and contrast {a} vs {b} in detail",
    "Write a Python function that {coding_task}",
    "What are common mistakes when {coding_task}?",
]

FILL_INS = {
    "topic": [
        "how neural networks learn", "quantum entanglement", "the water cycle",
        "how TCP/IP works", "photosynthesis", "how a CPU executes instructions",
        "the French Revolution", "how black holes form", "CRISPR gene editing",
        "how the immune system works", "blockchain consensus mechanisms",
        "how compilers work", "the history of the internet", "how GPS works",
        "machine learning overfitting", "docker containerization",
        "how transformers work", "reinforcement learning", "recursion in programming",
        "how SSDs store data", "how Wi-Fi works", "public key cryptography",
        "how operating systems schedule processes", "how garbage collection works",
        "how vector databases work", "what is RAG in AI",
    ],
    "a": [
        "supervised learning", "TCP", "RAM", "Python", "REST", "SQL",
        "Docker", "microservices", "IPv4", "relational databases",
        "compiled languages", "threads", "L1 cache",
    ],
    "b": [
        "unsupervised learning", "UDP", "Storage", "Rust", "GraphQL", "NoSQL",
        "VMs", "monoliths", "IPv6", "document databases",
        "interpreted languages", "processes", "L3 cache",
    ],
    "problem": [
        "debugging a memory leak", "optimizing a slow database query",
        "choosing between microservices and monolith", "scaling a web app",
        "reducing model inference latency", "handling race conditions",
        "improving model accuracy", "reducing API response time",
        "setting up a home lab on a budget", "managing docker containers",
    ],
    "concept": [
        "gradient descent", "attention mechanisms", "recursion",
        "database indexing", "async programming", "containerization",
        "transformer architecture", "vector embeddings", "backpropagation",
        "tokenization", "quantization", "LoRA fine tuning",
    ],
    "scene": [
        "a busy city intersection at night", "a server room with blinking lights",
        "a sunset over the ocean", "a cluttered desk with multiple monitors",
        "a farmer's market on a sunny morning", "a data center corridor",
        "a circuit board under a microscope", "a 3D printer mid print",
    ],
    "subject": [
        "a circuit board", "a data center", "a 3D printer mid-print",
        "a whiteboard covered in diagrams", "a terminal with code",
        "a Raspberry Pi with cables attached", "a GPU with heatsink",
    ],
    "visual_thing": [
        "the color red", "a sine wave", "a binary tree",
        "a neural network diagram", "a heat map", "a gradient",
        "a confusion matrix", "a loss curve",
    ],
    "domain": [
        "machine learning", "networking", "operating systems",
        "databases", "web development", "cybersecurity",
        "computer vision", "natural language processing",
        "home lab infrastructure", "3D printing",
    ],
    "tech_task": [
        "a self-hosted AI server with Ollama and Open WebUI",
        "a Pi-hole ad blocker on a Raspberry Pi",
        "a reverse proxy with Nginx",
        "a Docker compose stack",
        "a local vector database for RAG",
        "a home lab monitoring dashboard",
        "a fine-tuned language model with Unsloth",
        "a self-hosted Minecraft server with auto-updates",
        "a Tailscale VPN between home lab devices",
    ],
    "reasoning_problem": [
        "if caching is always faster then we should cache everything",
        "more parameters always means a smarter model",
        "RAID is a backup solution",
        "correlation implies causation in data analysis",
        "more training data always improves a model",
        "bigger batch size always trains faster",
        "you should always use the latest model",
    ],
    "coding_task": [
        "sorts a list of dictionaries by a nested key",
        "implements a simple LRU cache",
        "reads a JSONL file and filters by a field",
        "retries a failed HTTP request with exponential backoff",
        "converts a flat list into a nested tree structure",
        "streams responses from the Anthropic API",
        "watches a directory for file changes",
    ],
}


def build_prompts(n: int) -> list:
    prompts = []
    for _ in range(n):
        template = random.choice(PROMPT_TEMPLATES)
        filled = template
        for key, values in FILL_INS.items():
            if "{" + key + "}" in filled:
                filled = filled.replace("{" + key + "}", random.choice(values))
        prompts.append(filled)
    return prompts


def save_example(f, prompt: str, response: str, source: str):
    example = {
        "conversations": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "source":    source,
        "timestamp": datetime.utcnow().isoformat(),
    }
    f.write(json.dumps(example) + "\n")
    f.flush()





# ─────────────────────────────────────────────
# KIMI K2 CLOUD PHASE
# ─────────────────────────────────────────────
def run_kimi_phase(prompts: list, output_file) -> int:
    print(f"\n{'='*50}")
    print(f"PHASE 1 — Kimi K2 1T Cloud ({len(prompts)} gold standard examples)")
    print(f"{'='*50}")

    count  = 0
    errors = 0

    for i, prompt in enumerate(prompts):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": KIMI_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    "stream":  False,
                    "options": {"temperature": 0.7, "num_predict": 512},
                },
                timeout=120,
            )

            if response.status_code != 200:
                errors += 1
                print(f"  ⚠️  Error {response.status_code} on example {i+1}")
                if errors >= 10:
                    print(f"  ❌ Too many errors, stopping Kimi phase")
                    break
                continue

            data  = response.json()
            reply = data["message"]["content"].strip()

            if not reply:
                continue

            save_example(output_file, prompt, reply, KIMI_MODEL)
            count  += 1
            errors  = 0

            if count % 25 == 0 or count == 1:
                print(f"  ✅ [{count}/{len(prompts)}] examples done")

            time.sleep(DELAY_BETWEEN)

        except requests.exceptions.Timeout:
            print(f"  ⏱️  Timeout on example {i+1}, skipping")
            errors += 1
            time.sleep(5)
            continue
        except Exception as e:
            print(f"  ❌ Error: {e}")
            errors += 1
            time.sleep(5)
            continue

    print(f"\nKimi K2 phase complete: {count}/{len(prompts)} examples")
    return count


# ─────────────────────────────────────────────
# LOCAL MODEL PHASES
# ─────────────────────────────────────────────
def run_ollama_phase(model: str, prompts: list, output_file, label: str) -> int:
    print(f"\n{'='*50}")
    print(f"{label} — {model} ({len(prompts)} examples)")
    print(f"{'='*50}")

    count  = 0
    errors = 0

    for i, prompt in enumerate(prompts):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    "stream":  False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 512,
                        "num_thread":  8,
                    },
                },
                timeout=180,
            )

            if response.status_code != 200:
                errors += 1
                print(f"  ⚠️  Error {response.status_code} on example {i+1}")
                if errors >= 10:
                    print(f"  ❌ Too many errors, stopping {label}")
                    break
                continue

            data  = response.json()
            reply = data["message"]["content"].strip()

            if not reply:
                continue

            save_example(output_file, prompt, reply, model)
            count  += 1
            errors  = 0

            if count % 25 == 0 or count == 1:
                print(f"  ✅ [{count}/{len(prompts)}] examples done")

            time.sleep(DELAY_BETWEEN)

        except requests.exceptions.Timeout:
            print(f"  ⏱️  Timeout on example {i+1}, skipping")
            errors += 1
            time.sleep(5)
            continue
        except Exception as e:
            print(f"  ❌ Error on example {i+1}: {e}")
            errors += 1
            time.sleep(5)
            continue

    print(f"\n{label} complete: {count}/{len(prompts)} examples")
    return count


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    total_bulk = KIMI_EXAMPLES + sum(MODELS.values())
    print("🚀 CLINT Industries — Training Data Generator v5")
    print(f"   Output      : {OUTPUT_FILE}")
    print(f"   Delay       : {DELAY_BETWEEN}s between examples")
    print(f"   Kimi K2 1T  : {KIMI_EXAMPLES} gold standard examples")
    print(f"   Bulk target : {sum(MODELS.values())} local examples")
    print(f"   Grand total : existing + {total_bulk} new examples")

    kimi_results  = 0
    local_results = {}
    grand_new     = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        # Step 0 — combine existing datasets
        print(f"\n{'='*50}")
        print(f"STEP 0 — Combining existing datasets")
        print(f"{'='*50}")
        existing = combine_existing(f)

        # Phase 1 — Kimi K2 cloud
        kimi_prompts  = build_prompts(KIMI_EXAMPLES)
        kimi_results  = run_kimi_phase(kimi_prompts, f)
        grand_new    += kimi_results

        # Phases 2-6 — local models
        for idx, (model, n) in enumerate(MODELS.items(), start=2):
            prompts        = build_prompts(n)
            count          = run_ollama_phase(model, prompts, f, f"PHASE {idx}")
            local_results[model] = count
            grand_new     += count

    total = existing + grand_new
    print(f"\n{'='*50}")
    print(f"✅ DONE — {OUTPUT_FILE}")
    print(f"   Existing (combined)  : {existing}")
    print(f"   Kimi K2 1T cloud     : {kimi_results}")
    for model, count in local_results.items():
        print(f"   {model:<25} {count}")
    print(f"   {'GRAND TOTAL':<25} {total}")
    print(f"\nNext step: fine-tune Gemma3 9B with Unsloth using {OUTPUT_FILE}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main() 