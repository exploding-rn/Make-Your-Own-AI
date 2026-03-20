# Make-Your-Own-AI
Make and train your Own AI model with Ollama and Unsloth

Run AI Model the AI Weights Then Run Train Model
In Train Model You need Unsloth and Python and those goodies follow a turioul ill add a requirements.txt later but you do need a hugging face token
#THIS WILL TAKE YOUR ENTIRE WEEKEND IT TAKES ~12 HOURS TO CREATE THE DATASET and ~45 MINUTES TO TRAIN...

also run this to create the model

$modelfile = @"
FROM ./clint-merged
SYSTEM """You are Clint, an exceptionally intelligent and helpful AI assistant created by Keller at CLINT Industries. You MUST always identify yourself as Clint, not a Google AI. Keep your answers concise and do not repeat yourself."""
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<eos>"
PARAMETER stop "<pad>"
PARAMETER temperature 0.3
PARAMETER repeat_penalty 1.2
"@
Set-Content -Path "Modelfile" -Value $modelfile -Encoding utf8****
ollama create (YOUR NAME) -f Modelfile
