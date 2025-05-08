# COMS6998_HPML_KD
Final Project of COMS6998: High Performance Machine Learning
Contributors: Pei Tian, Yue Xu, Hongcheng Tian

Knowledge Distillation on different tasks: 

- Classification: [stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2)
- Language Modeling: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)
- Summarization: [openai/summarize_from_feedback](https://huggingface.co/datasets/openai/summarize_from_feedback)
- Reasoning: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)


### Repo Outline

- **notebook**: notebooks for different task
  - `KD-classification.ipynb`: classification task
  - `KD-lm.ipynb`: language modeling task
  - `KD-torch-reasoning.ipynb`: math reasoning task
- **script**
  - `KD-hf-summ.py`: summarization task
  - `summ_verify.py`: summarization evaluation
- **utils**
  - `logger.py`: logging utilities
 
### Example Usage
- For notebook files, please upload it to Google Colab and connect to session with GPU to run it
- For script files, please follow these command to run it:
  ```shell
  git clone https://github.com/Tptrix29/COMS6998_HPML_KD

  # env setup
  pip install requirements.txt

  cd scripts
  # training
  python KD-hf-summ.py
  # evaluation
  python summ_verify.py \
  --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
  --tokenizer_name Qwen/Qwen1.5-MoE-A2.7B \
  --dataset_name openai/summarize_from_feedback \
  --dataset_config axis \
  --split test \
  --max_input_length 512 \
  --max_target_length 128 \
  --batch_size 128
  ```

### Results
View it in our W&B report: [wandb report](https://wandb.ai/tptrix29/KD-COMS6998/reports/KD-on-Summarization--VmlldzoxMjYyMDI2NQ)

### Deliverables
**W&B Board Project Link:** 
- [Classification](https://wandb.ai/tptrix29/bert-distillation)
- [Language Modeling](https://wandb.ai/tptrix29/lm-kd)
- [Summarization Runs](https://wandb.ai/tptrix29/KD-COMS6998)

**Model Checkpoints:**
Model checkpoints are saved in Google Drive: 
- [Classification](https://drive.google.com/drive/folders/11bVns4xDc_WU0kJYzERCw7IpE44AvsWs?usp=sharing)
- [Language Modeling](https://drive.google.com/drive/folders/11bVns4xDc_WU0kJYzERCw7IpE44AvsWs?usp=sharing)
- [Summarization](https://drive.google.com/drive/u/0/folders/1ss29ZTFx-NL6W-6-mECs1N45H2Q5d7-X)
