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
- **Python 3.10+**
- **Unsloth** (for fast fine-tuning)
- **Ollama** (to run the model)
- **A Hugging Face account** (free, for model access)
- **GPU recommended** (RTX 3080, A100, etc.) — CPU will be very slow

See `requirements.txt` for full dependencies.

## Quick Start

### Step 1: Prepare Your Dataset
Edit `ai_model.py` with your training data. The script will:
- Process your data
- Create a training dataset
- Takes 1-24 hours depending on:
- Your GPU VRAM
- Dataset size

```bash
python ai_model.py
```

### Step 2: Train Your Model
The training will take **45 minutes** depending on:
- Your GPU VRAM
- Dataset size
- Settings you chose in `ai_model.py`


### Step 3: Create Your Model File
After training completes, create a Modelfile:

```
FROM ./clint-merged

SYSTEM """
{REPLACE WITH YOUR SYSTEM PROMPT HERE}
Keep your answers concise and do not repeat yourself.
"""

PARAMETER stop "<end_of_turn>"
PARAMETER temperature 0.3
PARAMETER repeat_penalty 1.2
```

Replace `{REPLACE WITH YOUR SYSTEM PROMPT HERE}` with something like:
- "You are Gary, a fun AI that loves pizza"
- "You are a helpful coding assistant"
- "You are a pirate that speaks in pirate speak"

### Step 4: Build and Run with Ollama
```bash
ollama create my-model -f Modelfile
ollama run my-model
```

## Understanding the Parameters

- **temperature 0.3** → Controls randomness (lower = more consistent, higher = more creative)
- **repeat_penalty 1.2** → Prevents the model from repeating itself (usually fixes repetition issues)
- **stop** → Tells the model when to stop generating

## Customization

### Adjust Training Speed vs Quality
Edit `ai_model.py`:
- **Faster training?** Reduce `num_train_epochs`
- **Better quality?** Increase `num_train_epochs` (takes longer)

### Change Model Behavior
In your Modelfile:
- Lower `temperature` (0.1-0.5) for focused, accurate responses
- Higher `temperature` (0.7-1.0) for creative responses
- Increase `repeat_penalty` if the model repeats itself

## Common Issues

**"Model keeps repeating itself"**
- Increase `repeat_penalty` in Modelfile (try 1.5-2.0)
- Lower `temperature` (try 0.2)

**"Training is too slow"**
- Check you're using a GPU (not CPU)
- Reduce dataset size
- Reduce `num_train_epochs` in `ai_model.py`

## Requirements

See `requirements.txt` for full list. Main dependencies:
- `unsloth` (fast fine-tuning)
- `ollama` (model serving)
- `transformers` (model handling)
- `torch` (GPU acceleration)

## License

Apache 2.0 Licsence

## Questions?

Open an issue or check the discussions tab!
