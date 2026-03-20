"""
CLINT Industries — Fine-tuning Script
Trains Gemma 3 12B on model_training_data_weighted.jsonl using Unsloth + QLoRA
Output: clint-lora (adapter) + clint-merged (full model ready for Ollama)
"""

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch
import json

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
HF_TOKEN       = "HF TOKEN HERE"  # Hugging Face token with access to the model
BASE_MODEL     = "google/gemma-3-4b-it"
DATASET_FILE   = "/mnt/c/Users/Keller/model_training_data_weighted.jsonl"
OUTPUT_LORA    = "/mnt/c/Users/Keller/clint-lora"
OUTPUT_MERGED  = "/mnt/c/Users/Keller/clint-merged"

MAX_SEQ_LENGTH = 2048   # context length during training
LORA_RANK      = 16     # higher = smarter but slower, 16 is sweet spot for 10GB VRAM
BATCH_SIZE     = 2      # keep low for 10GB VRAM
GRAD_ACCUM     = 4      # effective batch = BATCH_SIZE * GRAD_ACCUM = 8
EPOCHS         = 2      # 2 passes through the dataset
LEARNING_RATE  = 2e-4

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print(f"🦥 Loading {BASE_MODEL} with Unsloth...")
model, tokenizer = FastModel.from_pretrained(
    model_name      = BASE_MODEL,
    max_seq_length  = MAX_SEQ_LENGTH,
    load_in_4bit    = True,   # 4bit quantization — fits in 10GB VRAM
    load_in_8bit    = False,
    token           = HF_TOKEN,
)

# ─────────────────────────────────────────────
# APPLY LORA
# ─────────────────────────────────────────────
print("🔧 Applying LoRA adapters...")
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False,  # skip vision layers, text only
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r                          = LORA_RANK,
    lora_alpha                 = LORA_RANK * 2,
    lora_dropout               = 0.05,
    bias                       = "none",
    random_state               = 42,
)

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
print("📦 Loading dataset...")
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

def format_example(example):
    """Convert conversations format to text."""
    messages = example["conversations"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = load_dataset(
    "json",
    data_files=DATASET_FILE,
    split="train",
)

print(f"   Loaded {len(dataset)} examples")
dataset = dataset.map(format_example, remove_columns=dataset.column_names)
print(f"   Formatted and ready")

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
print("🚀 Starting training...")
trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = dataset,
    args          = SFTConfig(
        dataset_text_field   = "text",
        max_seq_length       = MAX_SEQ_LENGTH,
        per_device_train_batch_size   = BATCH_SIZE,
        gradient_accumulation_steps   = GRAD_ACCUM,
        num_train_epochs     = EPOCHS,
        learning_rate        = LEARNING_RATE,
        fp16                 = not torch.cuda.is_bf16_supported(),
        bf16                 = torch.cuda.is_bf16_supported(),
        logging_steps        = 10,
        save_steps           = 100,
        output_dir           = OUTPUT_LORA,
        warmup_ratio         = 0.05,
        lr_scheduler_type    = "cosine",
        seed                 = 42,
        report_to            = "none",  # no wandb
    ),
)

trainer_stats = trainer.train()
print(f"\n✅ Training complete")
print(f"   Time: {trainer_stats.metrics['train_runtime']:.0f}s")
print(f"   Loss: {trainer_stats.metrics['train_loss']:.4f}")

# ─────────────────────────────────────────────
# SAVE LORA ADAPTER
# ─────────────────────────────────────────────
print(f"\n💾 Saving LoRA adapter to {OUTPUT_LORA}...")
model.save_pretrained(OUTPUT_LORA)
tokenizer.save_pretrained(OUTPUT_LORA)
print("   LoRA adapter saved")

# ─────────────────────────────────────────────
# MERGE + SAVE FULL MODEL
# ─────────────────────────────────────────────
print(f"\n🔀 Merging LoRA into base model...")
model.save_pretrained_merged(
    OUTPUT_MERGED,
    tokenizer,
    save_method="merged_16bit",
)
print(f"   Merged model saved to {OUTPUT_MERGED}")

# ─────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────
print(f"""
{'='*50}
✅ CLINT training complete

LoRA adapter : {OUTPUT_LORA}
Merged model : {OUTPUT_MERGED}

Next steps:
1. Create Modelfile in C:\\Users\\Keller:
   FROM ./clint-merged
   SYSTEM \"\"\"You are Clint, an AI assistant by CLINT Industries.\"\"\"
   PARAMETER temperature 0.7
   PARAMETER num_ctx 8192

2. Run: ollama create clint -f Modelfile
3. Run: ollama run clint
{'='*50}
""")