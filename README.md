# MedQA DPO (Direct Preference Optimization)

This repository contains code for implementing Direct Preference Optimization (DPO) on medical question-answering tasks. The project focuses on fine-tuning language models to improve their performance on medical questions while maintaining safety and accuracy.

## Project Structure

### Core DPO Implementation (`/dpo`)
- `med_dpo_loss.py`: Implementation of the DPO loss function specifically tailored for medical QA tasks
- `run_dpo.py`: Main script for running DPO training
- `dpo_train.ipynb`: Jupyter notebook containing the DPO training pipeline and experiments
- `dpo_loop.ipynb`: Notebook for iterative DPO training and evaluation

### Data Management (`/data`)
- `synthetic_gen.ipynb`: Notebook for generating synthetic medical QA data
- `score_gen.py` & `score_gen_async.py`: Scripts for scoring generated responses
- `analyze_dataset.py`: Tools for analyzing and processing the medical QA datasets
- `/synthetic_medqa_data`: Directory containing generated synthetic medical QA data
- `/original_datasets`: Original medical QA datasets
- `/qwen3` & `/gemma3_data`: Model-specific data directories

### Additional Directories
- `/lora_adapter`: Contains LoRA (Low-Rank Adaptation) model adapters
- `/mlruns`: MLflow tracking directory for experiment logging
