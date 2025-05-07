import os
import datetime
import gc
import numpy as np
import torch
import torch.nn.functional as F

import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    Qwen2MoeConfig,
    Qwen2MoeForCausalLM,
    AutoTokenizer, 
    DefaultDataCollator,
    TrainingArguments, 
    Trainer, 
    TrainerCallback,
)
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, TaskType

import evaluate
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

import wandb
from dotenv import load_dotenv
from utils.logger import Logger


load_dotenv()
logger = Logger(__name__)

MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 128
CKPT_DIR = ""  # TODO: set the checkpoint directory
WANDB_PROJECT = "KD-COMS6998"
MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B"
USE_PRETRAINED = False
PRETRAINED = "Qwen/Qwen2.5-1.5B-Instruct"
# CKPT = f"{CKPT_DIR}/qwen2.5moe-0.8B-summ-prompt-Qwen2.5-1.5B-Instruct-20250505-143304/ckpt-epoch-1"
CKPT = ""

WANDB_RUNNAME = f"qwen2.5-1.5B-summ-prompt-{MODEL_NAME.split("/")[-1]}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
TEACHER_DEVICE = "cuda:1"
ATTN_IMPL = "flash_attention_2"
KD_WEIGHT = 0.8
KD_SCHEDULER_FLAG = False
T = 1
MAX_EPOCH = 3

class CausalLMDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, features):
        model_inputs = self.tokenizer(
            [f["input"] for f in features],
            [f["output"] for f in features],
            max_length=MAX_INPUT_LENGTH + MAX_OUTPUT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            padding_side="left",
        )
        prompt_inputs = self.tokenizer(
            [f["input"] for f in features],
            max_length=MAX_INPUT_LENGTH + MAX_OUTPUT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            padding_side="left",
        )
        labels = model_inputs["input_ids"].clone()
        completions_lens = (model_inputs["attention_mask"].sum(dim=1) - prompt_inputs["attention_mask"].sum(dim=1)).int()
        idx = torch.arange(labels.size(1)).expand(labels.size(0), labels.size(1)).to(labels.device)
        ignore_mask = (idx < (labels.size(1) - completions_lens.unsqueeze(1))).int()
        model_inputs["labels"] = labels.masked_fill(ignore_mask == 1, -100).masked_fill(model_inputs["input_ids"] == self.tokenizer.pad_token_id, -100)
        return model_inputs
    
class KDTrainer(Trainer):
    def __init__(self, teacher_model, temperature=1.0, alpha=0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.label_smoother = LabelSmoother()
        self.kd_loss = torch.nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)

        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def kd_weight_scheduler(self, init_weight, epoch, num_epochs):
        # Linear decay
        # return init_weight * (1 - epoch / num_epochs)
        # Cosine decay
        # return init_weight * (1 + np.cos(np.pi * epoch / num_epochs)) / 2
        # Exponential decay
        return init_weight * (0.5 ** (2 * epoch / num_epochs))
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_outputs = model(**inputs)
        student_loss = student_outputs.loss  # !!! There are 2 elements loss field
        # logger.debug(f"Student Loss: {student_loss}")

        # compute true ce loss
        # true_ce_loss = self.label_smoother(student_outputs, inputs["labels"], shift_labels=True)
        student_logits = student_outputs.logits[..., :-1, :].contiguous()
        labels = inputs["labels"][..., 1:].contiguous()
        true_ce_loss = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        
        # compute soft target ce loss
        student_device = self.model.device
        teacher_device = self.teacher_model.device
        teacher_inputs = inputs.copy().to(device=teacher_device)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)
        teacher_logits = teacher_outputs.logits.to(device=student_device)[..., :-1, :].contiguous()

        # KD loss
        # Soft target CE loss
        # T = self.temperature
        # logp = F.log_softmax(student_logits / T, dim=-1)
        # q = F.softmax(teacher_logits / T, dim=-1)
        # soft_target_ce_loss = -(T ** 2) * (q * logp).sum(dim=-1).mean() # scale by T^2

        # kl divergence loss
        kd_loss = self.kd_loss(
            F.log_softmax(student_logits / self.temperature, dim=-1).view(-1, student_logits.size(-1)),
            F.softmax(teacher_logits / self.temperature, dim=-1).view(-1, student_logits.size(-1)),
        ) * (self.temperature ** 2)  # scale by T^2

        kd_w = self.kd_weight_scheduler(KD_WEIGHT, self.state.epoch, self.args.num_train_epochs) if KD_SCHEDULER_FLAG else KD_WEIGHT
        loss = true_ce_loss + kd_w * kd_loss
        # record the loss
        wandb.log({
            "train/true_ce_loss": true_ce_loss.item(),
            "train/kd_loss": kd_loss.item(),
            "train/loss": loss.item(),
        }, step=self.state.global_step)
        
        return (loss, student_outputs) if return_outputs else loss
    

class EvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, ckpt_dir):
        self.eval_dataset = eval_dataset
        self.ckpt_dir = ckpt_dir
        self.bleu = evaluate.load("bleu")
        self.rouge = evaluate.load("rouge")
        self.acc = evaluate.load("accuracy")
        self.precision = evaluate.load("precision")
        self.recall = evaluate.load("recall")
        self.f1 = evaluate.load("f1")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        # 1. Save checkpoint
        epoch = int(state.epoch)
        ckpt_path = os.path.join(self.ckpt_dir, WANDB_RUNNAME, f"ckpt-epoch-{epoch}")
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        model = kwargs["model"]
        model.save_pretrained(ckpt_path)
        model.config.save_pretrained(ckpt_path)
        logger.info(f"Checkpoint saved at {ckpt_path}")

        # 2. Initialize vLLM engine for fast batch generation
        llm = LLM(
            model=ckpt_path, 
            tokenizer=MODEL_NAME, 
            tensor_parallel_size=1,
            dtype=torch.bfloat16,
        )
        sampling_params = SamplingParams(
            max_tokens=MAX_OUTPUT_LENGTH,
            temperature=0.0,
            top_p=1.0,
            stop_token_ids=[llm.get_tokenizer().eos_token_id],
        )

        # 3. Generate predictions
        predictions = []
        inputs = self.eval_dataset["input"]
        for r in llm.generate(inputs, sampling_params):
            predictions.append(r.outputs[0].text)

        # 4. Compute metrics
        bleu_res = self.bleu.compute(predictions=predictions, references=[[ref] for ref in self.eval_dataset["output"]])
        rouge_res = self.rouge.compute(predictions=predictions, references=self.eval_dataset["output"])
        logger.info(f"Epoch {epoch} - BLEU: {bleu_res['bleu']:.4f}, ROUGE-1: {rouge_res['rouge1']:.4f}, ROUGE-2: {rouge_res['rouge2']:.4f}, ROUGE-L: {rouge_res['rougeL']:.4f}")

        # 5. Log metrics to wandb
        wandb.log({
            "eval/bleu": bleu_res["bleu"],
            "eval/rouge-1": rouge_res["rouge1"],
            "eval/rouge-2": rouge_res["rouge2"],
            "eval/rouge-l": rouge_res["rougeL"],
        }, step=state.global_step)

        # Shut down vLLM’s parallel environment
        destroy_model_parallel()
        # del llm.llm_engine.model_executor
        del llm
        gc.collect()
        torch.cuda.empty_cache()

def eval_model_size(model):
    numel = 0
    for param in model.parameters():
        numel += param.numel()
    return numel

def process(example):
    return {
        "input": example["info"]["post"] if example["info"]["post"] else example["info"]["article"],
        "output": example["summary"]["text"],
    }

def add_prompt(example):
    return {
        "input": f"Summarize the following:\n{example['input']}\nSummary:",
    }

def main():
    # 1. Load dataset splits (axis-validation for train, axis-test for eval)
    train_ds = load_dataset("openai/summarize_from_feedback", "axis", split="validation")
    eval_ds  = load_dataset("openai/summarize_from_feedback", "axis", split="test")

    # small dataset for debugging
    # train_ds = train_ds.select(range(256))
    # eval_ds  = eval_ds.select(range(64))

    train_ds = train_ds.map(process, remove_columns=train_ds.column_names)
    eval_ds  = eval_ds.map(process, remove_columns=eval_ds.column_names)
    train_ds = train_ds.map(add_prompt)
    eval_ds  = eval_ds.map(add_prompt)

    logger.info(f"Loaded {len(train_ds)} training examples from openai/summarize_from_feedback/axis (validation)")
    logger.info(f"Loaded {len(eval_ds)} evaluation examples from openai/summarize_from_feedback/axis (test)")
    
    # sanity check
    for train_ex, eval_ex in zip(train_ds.select(range(1)), eval_ds.select(range(1))):
        logger.debug("=============== Dataset Sanity Check ======================")
        logger.debug(f"Train Input: {train_ex['input']}")
        logger.debug(f"Train Output: {train_ex['output']}")
        logger.debug(f"Eval Input: {eval_ex['input']}")
        logger.debug(f"Eval Output: {eval_ex['output']}")
        logger.debug("=============== Dataset Sanity Check (END) ======================")

    # 2. Load Qwen2.5 model & tokenizer
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    logger.debug(f"pad_token_id: {tokenizer.pad_token_id}, vocab_size: {tokenizer.vocab_size}, bos_token_id: {tokenizer.bos_token_id}, eos_token_id: {tokenizer.eos_token_id}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        attn_implementation=ATTN_IMPL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=TEACHER_DEVICE,
    )
    logger.debug(f"teacher vocab size: {teacher_model.config.vocab_size}, pad_token_id: {teacher_model.config.pad_token_id}, bos_token_id: {teacher_model.config.bos_token_id}, eos_token_id: {teacher_model.config.eos_token_id}")


    config = Qwen2MoeConfig(
        architectures=["Qwen2MoeForCausalLM"],  # for MoE
        model_type="qwen2_moe",
        attn_implementation=ATTN_IMPL,

        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=3,
        num_attention_heads=14,
        num_key_value_heads=2,
        hidden_act="silu",

        max_position_embeddings=32768,
        max_window_layers=21,
        sliding_window=4096,
        use_sliding_window=False,

        rope_theta=1_000_000.0,
        rope_scaling=None,

        attention_dropout=0.0,
        rms_norm_eps=1e-6,
        initializer_range=0.02,

        vocab_size=teacher_model.config.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True,

        torch_dtype=torch.bfloat16,
        use_cache=False,
        # transformers_version="4.51.3",

        # 🧠 MoE-specific configs:
        num_local_experts=4,           # Total experts per MoE layer
        num_experts_per_tok=2,         # How many experts each token is routed to
        moe_layer_freq=2,              # Insert MoE layer every N transformer layers
        output_router_logits=False     # Often False during inference
    )

    # load the pretrained model
    if USE_PRETRAINED:
        logger.info(f"Loading student model from {PRETRAINED}")
        student_model = AutoModelForCausalLM.from_pretrained(
            PRETRAINED,
            attn_implementation=ATTN_IMPL,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(torch.bfloat16).cuda()
    # load the checkpoint
    elif os.path.exists(CKPT):
        logger.info(f"Loading student model from {CKPT}")
        student_model = Qwen2MoeForCausalLM.from_pretrained(
            CKPT,
            config=config,
        ).to(torch.bfloat16).cuda()
    # load the model from scratch
    else:
        student_model = Qwen2MoeForCausalLM(config).to(torch.bfloat16).cuda()

    # log the model size
    logger.info(f"Student model size: {eval_model_size(student_model) / 1e9:.2f}B")
    
    # # LoRA config
    # lora_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    #     lora_dropout=0.1,
    #     bias="none",
    #     task_type=TaskType.CAUSAL_LM,
    # )
    # if USE_PRETRAINED:
    #     student_model = get_peft_model(
    #         student_model,
    #         lora_config,
    #     )
  

    # 3. Preprocessing
    data_collator = CausalLMDataCollator(tokenizer)

    eval_callback = EvalCallback(eval_ds, ckpt_dir=CKPT_DIR)

    wandb.init(project=WANDB_PROJECT, name=WANDB_RUNNAME,
               config={
                    "teacher_model": MODEL_NAME,
                    "student_model": WANDB_RUNNAME,
                    "max_input_length": MAX_INPUT_LENGTH,
                    "max_output_length": MAX_OUTPUT_LENGTH,
                    "max_epoch": MAX_EPOCH,
                    "kd_weight": KD_WEIGHT,
                    "temperature": T,
                    "batch_size": 4,
               })

    training_args = TrainingArguments(
        output_dir=CKPT_DIR,
        remove_unused_columns=False,
        torch_compile_backend="inductor",
        dataloader_drop_last=True,
        num_train_epochs=MAX_EPOCH,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=2,
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="no",
        # save_strategy="epoch",
        # load_best_model_at_end=True,
        report_to=["wandb"],
        run_name=WANDB_RUNNAME,
        bf16=True,
        lr_scheduler_type="cosine_with_restarts",
        warmup_steps=100,
    )

    trainer = KDTrainer(
        teacher_model=teacher_model,
        temperature=T,  # KD temperature
        alpha=KD_WEIGHT,  # KD loss weight, 0 for CE loss only
        model=student_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=[eval_callback],
    )

    trainer.train()

if __name__ == "__main__":
    main()
