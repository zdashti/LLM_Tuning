# LLM_Tuning


# Qwen DPO Fine-Tuning and Evaluation: qwen_dpo_preference_tuning.ipynb

This Jupyter Notebook demonstrates how to fine-tune the **Qwen2.5-0.5B-Instruct** model using **Direct Preference Optimization (DPO)** on the **Intel/Orca DPO Pairs** dataset.  
The notebook includes:
- Data preprocessing and prompt formatting for DPO.
- Model fine-tuning with `trl.DPOTrainer` and `peft.LoraConfig`.
- Evaluation on the validation split using both loss metrics and preference accuracy.
- Example generations to qualitatively inspect model responses.

This project aims to explore preference-based alignment techniques for small instruction-tuned models.


# QLoRA Fine-tuning on CNN/DailyMail

This project fine-tunes a TinyLlama (1.1B) language model using QLoRA (PEFT, 4-bit quantization) on the CNN/DailyMail dataset for news summarization (highlights generation).
It includes:
Dataset preparation and tokenization
PEFT configuration with LoRA adapters
Model fine-tuning and evaluation (ROUGE, Perplexity)
Example summaries and performance analysis

Results:
ROUGE-1: 0.33 | ROUGE-L: 0.20 | Perplexity: 6.32
Trainable Params: 12.6M (≈1.13%) | Eval Runtime: 279s
