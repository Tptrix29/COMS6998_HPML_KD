# COMS6998_HPML_KD
Final Project of COMS6998: High Performance Machine Learning

Knowledge Distillation on different tasks: 

- Classification
- Language Modeling
- Summarization
- Reasoning



### Repo Structure

├── **notebook**: notebooks for different task

│  ├── `KD-classification.ipynb`: classification task

│  ├── `KD-lm.ipynb`: language modeling task

│  └── `KD-torch-reasoning.ipynb`: math reasoning task

├── `requirements.txt`: env library requirements 

├── **script**

  ├── `KD-hf-summ.py`: summarization task

  └── `summ_verify.py`: summarization evaluation

└── **utils**

  └── `logger.py`: logging utilities 
