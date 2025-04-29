from unsloth import FastLanguageModel
from datasets import load_dataset
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")



root = '/.' # switch to '/content/drive/MyDrive/' if in colab

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "google/gemma-1.1-2b-it",  # base model
    max_seq_length = 2048,
    dtype = "auto",
    load_in_4bit = True,  # or False if you have enough VRAM
)

model.load(f'{root}/lora_adapter')

dataset = load_dataset("json", data_files={"train": f"{root}data/gemma3_data/gemma3_dpo_scored_data.jsonl"})

print(dataset['train'][0])

