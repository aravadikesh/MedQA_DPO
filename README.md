# MedQA DPO (Direct Preference Optimization)

This repository contains code for implementing Direct Preference Optimization (DPO) on medical question-answering tasks. The project focuses on fine-tuning language models to improve their performance on medical questions while maintaining safety and accuracy.

## Project Structure

### Core DPO Implementation (`/dpo`)
- `med_dpo_loss.py`: Implementation of the DPO loss function specifically tailored for medical QA tasks
- `gemma_training_suite.ipynb`: Jupyter notebook containing the DPO training pipeline and experiments for the Gemma3 model
- `qwen_training_suite.ipynb`: Jupyter notebook containing the DPO training pipeline and experiments for the Qwen model

### Data Management (`/data`)
- `synthetic_gen.ipynb`: Notebook for generating synthetic medical QA data
- `score_gen.py` & `score_gen_async.py`: Scripts for scoring generated responses
- `medqa_async_inference.py`: Script for asynchronous inference on MedQA data via GPT 4o mini
- `/synthetic_medqa_data`: Directory containing generated synthetic medical QA data
- `/original_datasets`: Original medical QA datasets 
- `/qwen3` & `/gemma3_data`: Model-specific data directories
- `/runs`: Contains subdirectories for experiment runs and TensorBoard logs (e.g., `gemma3_sft`, `qwen3_sft`)

### SFT Notebooks (`/sft_notebooks`)
- `biomistral_finetune.ipynb.ipynb`: Notebook for fine-tuning BioMistral models
- `Mistral Finetune.ipynb.ipynb`: Extended notebook for Mistral fine-tuning
- `Qwen3_MedQA.ipynb`: SFT pipeline and experiments for Qwen3-based medical QA.
- `Gemma3_MedQA.ipynb`: SFT pipeline and experiments for Gemma3-based medical QA.
- `Llama3_2_MedQA.ipynb`: SFT pipeline and experiments for Llama3.2-based medical QA.

### Model Adapters
- `/lora_adapter`: Contains LoRA (Low-Rank Adaptation) model adapters and related files
- `/gemma_med_dpo_adapter`: Adapter and configuration files for Gemma-based DPO models (see its README for details)

### Visualization and Analysis
- `/plots`: Contains plotting scripts and output images
  - `plotting.ipynb`: Notebook for visualizing training and validation loss curves from experiment runs
  - `sft_loss_curves.png`: Example output plot of SFT loss curves


