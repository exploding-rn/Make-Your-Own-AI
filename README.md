# Make Your Own AI Model

Train a custom AI model in ~1 day using your own data, Unsloth, and Ollama.

## What This Does

This project guides you through:
1. **Creating a custom dataset** from your own data (~45 min)
2. **Fine-tuning Gemma 9B** using that dataset (1-24 hours depending on hardware)
3. **Running your model locally** with Ollama

No complex ML knowledge required — just follow the steps.

## Before You Start

You'll need:
- **Python 3.10 REQUIRED ANY VERSION OF 3.10**
- **Unsloth** (for fast fine-tuning)
- **Ollama** (to run the model)
- **A Hugging Face account** (free, for model access)
- **VRAM recommended** 8GB — CPU will be very slow

See `requirements.txt` for full dependencies.

## Quick Start

### Step 1: Prepare Your Dataset
Edit `ai_model.py` with your training data. The script will:
- Process your data
- Create a training dataset
- Takes ~45 minutes

```bash
python ai_model.py
