# LLM_Tuning

# qwen_dpo_preference_tuning.ipynb
# Qwen DPO Fine-Tuning and Evaluation

This Jupyter Notebook demonstrates how to fine-tune the **Qwen2.5-0.5B-Instruct** model using **Direct Preference Optimization (DPO)** on the **Intel/Orca DPO Pairs** dataset.  
The notebook includes:
- Data preprocessing and prompt formatting for DPO.
- Model fine-tuning with `trl.DPOTrainer` and `peft.LoraConfig`.
- Evaluation on the validation split using both loss metrics and preference accuracy.
- Example generations to qualitatively inspect model responses.

This project aims to explore preference-based alignment techniques for small instruction-tuned models.
