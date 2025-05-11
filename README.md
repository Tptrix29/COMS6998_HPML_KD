- # HPML Project: Efficient Knowledge Distillation

  ## Team Information

  - **Team Name**: Efficient Knowledge Distillation
  - **Members**:
    - Pei Tian (pt2632)
    - Yue Xu (yx2876)
    - Hongcheng Tian (ht2657)

  ---

  ## 1. Problem Statement

  Knowledge Distillation on different tasks: 

  - Classification: [stanfordnlp/sst2](https://huggingface.co/datasets/stanfordnlp/sst2)
  - Language Modeling: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext)
  - Summarization: [openai/summarize_from_feedback](https://huggingface.co/datasets/openai/summarize_from_feedback)
  - Reasoning: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)

  ---

  ## 2. Model Description

  - Model Architecture: More Details [Here](https://wandb.ai/tptrix29/KD-COMS6998/reports/Efficient-Knowledge-Distillation--VmlldzoxMjYyMDI2NQ?accessToken=t6zrnf3ieh6gwnp8atjixk7phndhhxqdnl39ya19avfpq7fzgedoaydze7ttvgyg)
    - Classification: TinyBERT, DistillBERT
    - Language Modeling: Llama2
    - Summarization: Qwen2.5, Qwen1.5MoE
    - Reasoning: Qwen2.5

  - **Model Training / Fine-Tuning**: PyTorch, Transformers
  - **Optimization:** Torch Dynamo, Flash-Attention, Bitsandbytes
  - **Profiling**: Weights & Bias 
  - **Inference & Evaluation**: vLLM, evaluate
  - **Customized loss function (add KL divergence loss term) of the model to complete the distillation task.** 

  ---

  ## 3. Final Results Summary

  1. Classification

     |   Metric   |  Teacher  | Student (Pretrained) | Student (Fine-Tuned) |
     | :--------: | :-------: | :------------------: | :------------------: |
     | *Size (M)* |   *67*    |        *14.4*        |        *14.4*        |
     |  Accuracy  | **0.911** |        0.463         |        0.893         |

     

  2. Language Modeling

     |   Metric   |  Teacher  | Student 1 | Student 2 |
     | :--------: | :-------: | :-------: | :-------: |
     | *Size (B)* |    *7*    |   *0.5*   |   *1.5*   |
     | Perplexity | **22.13** |  226.51   |  182.03   |
     | Accuracy@1 | **0.434** |   0.235   |   0.254   |
     | Accuracy@5 | **0.655** |   0.394   |   0.419   |

  3. Summarization

     |   Metric   |  Teacher  | Student (Pretrained) | Student (Fine-Tuned) |
     | :--------: | :-------: | :------------------: | :------------------: |
     | *Size (B)* |   *13*    |        *1.5*         |        *1.5*         |
     |    BLEU    |   0.105   |        0.010         |      **0.139**       |
     |  ROUGE-1   | **0.355** |        0.331         |        0.339         |
     |  ROUGE-2   |   0.187   |        0.172         |      **0.204**       |
     |  ROUGE-L   |   0.264   |        0.243         |      **0.268**       |

  4. Reasoning: Knowledge distillation on a small-scale student model is not suitable for this complex task

  ---

  ## 4. Reproducibility Instructions

  ### A. Requirements

  Clone repo & Install dependencies:

  ```bash
  git clone https://github.com/Tptrix29/COMS6998_HPML_KD
  cd COMS6998_HPML_KD
  
  # env setup
  pip install requirements.txt
  ```

  ---

  ### B. Wandb Dashboard

  View training and evaluation metrics here: [Project Dashboard](https://wandb.ai/tptrix29/KD-COMS6998?nw=nwusertpzl0222)

  ---

  ### C. Training

  To train the model for the summarization task:

  ```bash
  cd scripts
  # training
  python KD-hf-summ.py
  ```

  For other tasks: please upload corresponding `ipynb` file in `notebook` to Google Colab and connect to a session with a GPU to run it.

  

  ---

  ### D. Evaluation

  To evaluate the model for the summarization task:

  ```bash
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

  ---

  For other tasks: please upload corresponding `ipynb` file in `notebook` to Google Colab and connect to a session with a GPU to run it.

  

  ### E. Quickstart: Minimum Reproducible Result

  - For notebook files, please upload them to Google Colab and connect to a session with a GPU to run them

  - For script files, please follow these command to run it:

    ```shell
    git clone https://github.com/Tptrix29/COMS6998_HPML_KD
    cd COMS6998_HPML_KD
    
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

  ---

  ## 5. Notes

  ### Repo Outline

  - `notebook`: notebooks for different task
    - `KD-classification.ipynb`: classification task
    - `KD-lm.ipynb`: language modeling task
    - `KD-torch-reasoning.ipynb`: math reasoning task
  - `script`: python scripts
    - `KD-hf-summ.py`: summarization task
    - `summ_verify.py`: summarization evaluation
  - `utils`: utilities
    - `logger.py`: logging utilities

  ### Deliverables

  - View details of profiling results and comments in our W&B report: [W&B Report - Efficient Knowledge Distillation](https://wandb.ai/tptrix29/KD-COMS6998/reports/Efficient-Knowledge-Distillation--VmlldzoxMjYyMDI2NQ?accessToken=t6zrnf3ieh6gwnp8atjixk7phndhhxqdnl39ya19avfpq7fzgedoaydze7ttvgyg)

  - Trained model weights are saved in Google Drive. 
    - [Classification](https://drive.google.com/drive/folders/11bVns4xDc_WU0kJYzERCw7IpE44AvsWs?usp=sharing)
    - [Language Modeling](https://drive.google.com/drive/folders/11bVns4xDc_WU0kJYzERCw7IpE44AvsWs?usp=sharing)
    - [Summarization](https://drive.google.com/drive/u/0/folders/1ss29ZTFx-NL6W-6-mECs1N45H2Q5d7-X)
  - Contact information: [pt2632@columbia.edu](mailto: pt2632@columbia.edu)
