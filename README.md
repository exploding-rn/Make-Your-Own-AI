# Make-Your-Own-AI
Make and train your Own AI model with Ollama and Unsloth
#TRAINING
Run AI Model the AI Weights Then Run Train Model
In Train Model You need Unsloth and Python and those goodies follow a turioul ill add a requirements.txt later but you do need a hugging face token
#THIS WILL TAKE YOUR ENTIRE WEEKEND IT TAKES BETWEEN 1-24 HOURS DEPENDING ON YOUR SETTINGS IN ai model.py! AND TO CREATE THE DATASET and ~45 MINUTES TO TRAIN...

# AFTER TRAINING
run this follow the MORE INTRUCTIONS below to fine tune


$modelfile = @"
FROM ./clint-merged
SYSTEM """ {REPLACE HERE WITH YOUR SYS PROMPT}  Keep your answers concise and do not repeat yourself. """
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<eos>"
PARAMETER stop "<pad>"
PARAMETER temperature 0.3
PARAMETER repeat_penalty 1.2
"@
Set-Content -Path "Modelfile" -Value $modelfile -Encoding utf8****
ollama create (YOUR NAME) -f Modelfile

-------------------------------------------------------------------------------
# MORE INTSTRUCTIONS
note replace the {} with your custom prompt like "You are Gary a fun AI that loves pizza" I suggjest keeping "keep your answers concise and do not repeat yourself." since that is a MAJOR issue
also the "PARAMETER temperature 0.3" in the AFTER TRAINING section you can adust.  IDFK WHAT THAT DOES but... it does something and it usally fixs most things
