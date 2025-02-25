import os
import sys

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams


@torch.inference_mode()
def generate_response_vllm(model, tokenizer, dataset):
    sampling_params = SamplingParams(temperature=0, max_tokens=1024)
    prompts = dataset['prompt']
    chat_prompts = []
    for p in prompts:
        prompt_messages = [{"role": "user", "content": p}]
        chat_prompts.append(tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True))
    responses = model.generate(chat_prompts, sampling_params)
    responses = [response.outputs[0].text for response in responses]
    dataset = dataset.add_column("response", responses)
    return dataset


if __name__ == "__main__":
    model_name_or_path, save_path = sys.argv[1], sys.argv[2]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = LLM(
        model_name_or_path,
        gpu_memory_utilization=0.95, 
        swap_space=1,
        # tensor_parallel_size=torch.cuda.device_count(), 
        trust_remote_code=True, 
        dtype="auto"
    )
    
    os.makedirs(save_path, exist_ok=True)
    
    eval_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split= 'test_prefs[:200]')
    eval_answers = generate_response_vllm(model, tokenizer, eval_dataset)
    eval_answers.save_to_disk(save_path)
    