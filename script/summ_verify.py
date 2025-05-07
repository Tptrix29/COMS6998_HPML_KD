import multiprocessing as mp
import argparse
from tqdm.auto import tqdm

from transformers import AutoTokenizer
import evaluate
from datasets import load_dataset

from utils.logger import Logger

from dotenv import load_dotenv
load_dotenv()

mp.set_start_method("spawn", force=True)

logger = Logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained causal-LM with vLLM on summarization"
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--dataset_name", type=str, default="openai/summarize_from_feedback")
    parser.add_argument("--dataset_config", type=str, default="axis")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of tensor-parallel shards (GPUs)")
    # parser.add_argument("--num_beams", type=int, default=4, help="Number of beams for beam search")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for generation")
    return parser.parse_args()

# extract info.post and summary.text fields
def process(example):
    return {
        "input": example["info"]["post"] if example["info"]["post"] else example["info"]["article"],
        "output": example["summary"]["text"],
    }

def add_prompt(example):
    return {
        "input": f"Summarize the following:\n{example['input']}\nSummary:",
    }

def add_prompt(example):
    return {
        "input": f"Summarize the following:\n{example['input']}\nSummary:",
    }

def main():
    args = parse_args()

    # Load dataset
    ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    # ds = ds.select(range(0, 100))  # for testing
    ds = ds.map(process)
    ds = ds.map(add_prompt)
    logger.info(f"Loaded {len(ds)} examples from {args.dataset_name}/{args.dataset_config} ({args.split})")
    for ex in ds.select(range(1)):
        logger.debug("=============== Dataset Sanity Check ======================")
        logger.debug(f"Input: {ex['input']}")
        logger.debug(f"Output: {ex['output']}")
        logger.debug("=============== Dataset Sanity Check (END) ======================")

    # Prepare metrics
    rouge = evaluate.load("rouge")
    bleu  = evaluate.load("bleu")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Build vLLM client
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model_name_or_path,
        tokenizer=args.tokenizer_name,
        # tensor_parallel_size=args.tensor_parallel_size,  # multiple GPUs
        trust_remote_code=True,
    )

    # Sampling parameters: beam search
    sampling_params = SamplingParams(
        max_tokens=args.max_target_length,
        temperature=0.0,
        top_p=1.0,
    )

    predictions = []
    references  = [ref for ref in ds["output"]]

    # Generate in small batches for speed (here batch_size=8 prompts per call)
    batch_size = args.batch_size
    for i in tqdm(range(0, len(ds), batch_size)):
        batch = ds.select(range(i, min(i + batch_size, len(ds))))
        prompts = ["Summarize the following:\n" + ex["input"] + "\nSummary:" for ex in batch]
        

        # Generate
        requests = llm.generate(prompts, sampling_params)
        for req in requests:
            # req.outputs is a list, but with temperature=0 it's a single sample
            text = req.outputs[0].text
            predictions.append(text)

    logger.debug("=============== vLLM Sanity Check ======================")
    logger.debug(f"Input: {ds[0]['input']}")
    logger.debug(f"Reference: {references[0]}")
    logger.debug(f"Prediction: {predictions[0]}")
    logger.debug("=============== vLLM Sanity Check (END) ======================")

    logger.info(f"Generated completions for {len(predictions)} examples")

    # Compute metrics
    rouge_res = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    bleu_res  = bleu.compute(
        predictions=[pred for pred in predictions],
        references=[[ref] for ref in references],
    )

    # Print results
    logger.info("=== vLLM Evaluation Results ===")
    logger.info(f"Model: {"/".join(args.model_name_or_path.split('/')[-2:])}")
    logger.info(f"Dataset: {args.dataset_name}/{args.dataset_config} ({args.split})")
    logger.info(f"Examples evaluated: {len(predictions)}")
    logger.info(f"ROUGE-1:  {rouge_res['rouge1']:.4f}")
    logger.info(f"ROUGE-2:  {rouge_res['rouge2']:.4f}")
    logger.info(f"ROUGE-L:  {rouge_res['rougeL']:.4f}")
    logger.info(f"BLEU:     {bleu_res['bleu']:.4f}")
    logger.info("==============================")

if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    main()
