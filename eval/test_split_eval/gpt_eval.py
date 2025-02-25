import sys
import re
import random
from tqdm import tqdm
random.seed(0)

from .api import API 

from datasets import Dataset, load_from_disk

api = API()


EVAL_PROMPT = '''Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: [[A]] if assistant A is better, [[B]] if assistant B is better, and [[C]] for a tie.

[Instruction]
{instruction}

[The Start of Assistant A's Answer]
{answer_A}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_B}
[The End of Assistant B's Answer]
'''

EVAL_PATTERN = re.compile(r'\[\[([ABC])\]\]')


if __name__ == "__main__":
    A, B, save_path = sys.argv[1], sys.argv[2], sys.argv[3]
    dataset_A = load_from_disk(A)
    dataset_B = load_from_disk(B)
    eval_results = []
    i = 0
    for a, b in tqdm(zip(dataset_A, dataset_B)):
        if i >= 200:
            break
        assert a['prompt'] == b['prompt']
        swap = random.randint(0, 1)
        if swap:
            a, b = b, a
        eval_prompt = EVAL_PROMPT.format(
            instruction=a['prompt'], 
            answer_A=a['response'], 
            answer_B=b['response']
        )
        try:
            gpt_response = api.gpt(eval_prompt, model='gpt-4')[0]
            eval_result = EVAL_PATTERN.search(gpt_response).group(1)
        except:
            continue
        
        if swap:
            a, b = b, a
            eval_result = 'B' if eval_result == 'A' else 'A' if eval_result == 'B' else 'C'
        eval_results.append({
            'instruction': a['prompt'],
            A: a['response'],
            B: b['response'],
            'eval': eval_result
        })
    
        Dataset.from_list(eval_results).save_to_disk(save_path)
        i += 1
    print(api.gpt_usage())
    