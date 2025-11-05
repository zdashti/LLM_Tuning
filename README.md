# LLM_Tuning


# Qwen DPO Fine-Tuning and Evaluation: qwen_dpo_preference_tuning.ipynb

This Jupyter Notebook demonstrates how to fine-tune the **Qwen2.5-0.5B-Instruct** model using **Direct Preference Optimization (DPO)** on the **Intel/Orca DPO Pairs** dataset.  
The notebook includes:
- Data preprocessing and prompt formatting for DPO.
- Model fine-tuning with `trl.DPOTrainer` and `peft.LoraConfig`.
- Evaluation on the validation split using both loss metrics and preference accuracy.
- Example generations to qualitatively inspect model responses.

This project aims to explore preference-based alignment techniques for small instruction-tuned models.


# QLoRA Fine-tuning on CNN/DailyMail: Peft_QloRa_Cnn.ipynb

This project fine-tunes a TinyLlama (1.1B) language model using QLoRA (PEFT, 4-bit quantization) on the CNN/DailyMail dataset for news summarization (highlights generation).
It includes:
- Dataset preparation and tokenization
- PEFT configuration with LoRA adapters
- Model fine-tuning and evaluation (ROUGE, Perplexity)
- Example summaries and performance analysis

Results:
- ROUGE-1: 0.33 | ROUGE-L: 0.20 | Perplexity: 6.32
- Trainable Params: 12.6M (≈1.13%) | Eval Runtime: 279s


# Persian NER with LoRA (BERT‑fa) : peft_ner.ipynb

This project demonstrates training a binary **Named Entity Recognition (NER)** model for Persian using **HooshvareLab/bert-fa-base-uncased** and **LoRA (Low-Rank Adaptation)** for efficient fine-tuning.

## Dataset  
The dataset is sourced from [Persian-NER-GUI](https://github.com/Mostafa-Modaberi/Persian-NER-GUI).  
- Total samples: **18,269**  
- Split: **80% train**, **10% validation**, **10% test**  
- Entity Class (ENT) ratio: ~2%  
- Labels: **0 (NON-ENT)**, **1 (ENT)**

## Model & Training  
- **Base Model**: `HooshvareLab/bert-fa-base-uncased`  
- **LoRA Configuration**: `r=16`, `alpha=32`, `dropout=0.1`, targeting `["query", "key", "value", "dense"]` layers  
- **Training**:  
  - Learning rate: 3e‑4  
  - Batch size: 16 (train), 32 (eval)  
  - Max epochs: 5  
  - Using `fp16` with RTX 4060 (16GB VRAM)

## Results  
- **Validation**: F1 ≈ 0.9427, Precision ≈ 0.9399, Recall ≈ 0.9455, Loss ≈ 0.0081  
- **Test**: F1 ≈ 0.9489, Precision ≈ 0.9503, Recall ≈ 0.9475, Loss ≈ 0.0076

## Conclusion
The model achieves excellent performance (F1 ≈ 0.95) for a highly imbalanced dataset. It is a solid foundation for expanding to multi-class NER tasks (e.g., PER, LOC, ORG).

