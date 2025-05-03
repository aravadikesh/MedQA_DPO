from datasets import load_dataset

import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "unsloth/gemma-3-1b-it-unsloth-bnb-4bit"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# root = '/.' # switch to '/content/drive/MyDrive/' if in colab

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "google/gemma-1.1-2b-it",  # base model
#     max_seq_length = 2048,
#     dtype = "auto",
#     load_in_4bit = True,  # or False if you have enough VRAM
# )

# model.load(f'{root}/lora_adapter')

# dataset = load_dataset("json", data_files={"train": f"{root}data/gemma3_data/gemma3_dpo_scored_data.jsonl"})

# print(dataset['train'][0])

