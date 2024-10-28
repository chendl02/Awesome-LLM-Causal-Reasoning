import pandas as pd
import json
import os
import re
from typing import List, Dict, Callable
from gpt_batch.batcher import GPTBatcher


# Constants
SILICON_API_BASE_URL = "https://api.siliconflow.cn/v1"
SILICON_API_KEY = ''
SILICON_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    #"meta-llama/Meta-Llama-3.1-405B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    #"deepseek-ai/DeepSeek-V2.5",
    #"deepseek-ai/DeepSeek-V2-Chat",
    "Pro/google/gemma-2-9b-it",
    "google/gemma-2-27b-it"
]


TOGETHER_MODEL = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.3"
]
#TOGETHER_API_KEY =  your api key
TOGETHER_BASE_URL = "https://api.together.xyz/v1/"

GPT_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4-turbo",
    "gpt-4o",
    "o1-mini"]
GPT_BASE_URL = "https://api.openai.com/v1"
#GPT_API_KEY = "your api key"


Claude_BASE_URL = "https://api.claude.ai/v1"
#Claude_API_KEY = "your api key"
GPT_API_KEY =   "your api key"

TOGETHER_API_KEY = "your api key"
model_list ={
  "Pro/meta-llama/Meta-Llama-3.1-8B-Instruct": {
    "API_BASE_URL": SILICON_API_BASE_URL,
    "API_KEY": SILICON_API_KEY
  },
  "Pro/meta-llama/Meta-Llama-3-8B-Instruct": {
    "API_BASE_URL": SILICON_API_BASE_URL,
    "API_KEY": SILICON_API_KEY
  },
  "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {
    "API_BASE_URL": TOGETHER_BASE_URL,
    "API_KEY": TOGETHER_API_KEY
  },
  "meta-llama/Meta-Llama-3-70B-Instruct-Turbo": {
    "API_BASE_URL": TOGETHER_BASE_URL,
    "API_KEY": TOGETHER_API_KEY
  },
  #"deepseek-ai/DeepSeek-V2.5": {
    #"API_BASE_URL": SILICON_API_BASE_URL,
    #"API_KEY": SILICON_API_KEY
  #},
  #"deepseek-ai/DeepSeek-V2-Chat": {
    #"API_BASE_URL": SILICON_API_BASE_URL,
    #"API_KEY": SILICON_API_KEY},
  "google/gemma-2-9b-it": {
    "API_BASE_URL": SILICON_API_BASE_URL,
    "API_KEY": SILICON_API_KEY
  },
  "google/gemma-2-27b-it": {
    "API_BASE_URL": SILICON_API_BASE_URL,
    "API_KEY": SILICON_API_KEY
  },
  "gpt-3.5-turbo": {
    "API_BASE_URL": GPT_BASE_URL,
    "API_KEY": GPT_API_KEY
  },
    #"gpt-4o": {
    #"API_BASE_URL": GPT_BASE_URL,
    #"API_KEY": GPT_API_KEY
  #},
  #"gpt-4-turbo": {
    #"API_BASE_URL": GPT_BASE_URL,
    #"API_KEY": GPT_API_KEY
    #},
"mistralai/Mistral-7B-Instruct-v0.3": {
"API_BASE_URL": TOGETHER_BASE_URL,
    "API_KEY": TOGETHER_API_KEY
  },
  "mistralai/Mixtral-8x7B-Instruct-v0.1": {
    "API_BASE_URL": TOGETHER_BASE_URL,
    "API_KEY": TOGETHER_API_KEY
    },
    #"gemini-1.5-pro": {
    #"API_BASE_URL": Claude_BASE_URL,
    #"API_KEY": Claude_API_KEY
    #},
    "claude-3-5-sonnet-20240620": {
    "API_BASE_URL": Claude_BASE_URL,
    "API_KEY": Claude_API_KEY
    }

}

SUFFIX = '''You are a highly intelligent question-answering bot with profound knowledge of causal inference.'''
COT_PROMPT ='''Begin your response with reasoning or evidence to suport your explanation, then return me the final result marked by '####'.'''
DIRECT_IO_PROMPT = '''Give me the anwser directly'''
# Helper functions
def load_json_file(file_path: str) -> pd.DataFrame:

    if file_path.endswith('.jsonl'):
        return pd.read_json(file_path, lines=True)
    else:
        return pd.read_json(file_path)

def append_results(result_dict: Dict, filename: str):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            results = json.load(f)
    else:
        results = []
    results.append(result_dict)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def parse_answer(response: str, options: List[str]) -> str:
    if "####" in response:
        response = "".join(response.split("####")[1:])
    if "Answer:" in response:
        response = "".join(response.split("Answer:")[1:])
    if "answer:" in response:
        response = "".join(response.split("answer:")[1:])
    
    for option in options[::-1]:
        pattern = rf'(?<![a-zA-Z])[\n\s]*{re.escape(option)}[.\)]?[\n\s]*(?![a-zA-Z])'
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return option
    return ""

def get_cot_prompt(task_name) -> str:
    file_name = f"./prompt/{task_name}.txt"
    with open(file_name, 'r') as f:
        prompt = f.read()
    return prompt

# Dataset-specific processing functions
def process_cladder(df: pd.DataFrame, flag_few_shot: bool = False, flag_direct_io: bool = False) -> pd.DataFrame:
    surfix = SUFFIX
    base_prompt = '''
        Question: {question}
    '''
    
    if flag_direct_io:
        base_prompt += f"{DIRECT_IO_PROMPT} Answer: Yes or Answer: No. Do not use any other format."
            
    if not flag_direct_io:
        base_prompt += f"{COT_PROMPT} The answer format is #### Yes or #### No."
    
    if flag_few_shot:
        if flag_direct_io:
            examples = df.sample(2).apply(lambda row: base_prompt.format(question=row['prompt']) + f"\nAnswer: {row['label'].capitalize()}", axis=1).str.cat(sep="\n\n")
        else:
            examples = get_cot_prompt("cladder")
        base_prompt = f"\nHere are some examples:\n\n{examples}\n MY QUESTION\n{base_prompt}"

    base_prompt = surfix + base_prompt
    df['prompt'] = df['prompt'].apply(lambda q: base_prompt.format(question=q))
    return df[['prompt', 'label']]



def process_copa(df: pd.DataFrame, flag_few_shot: bool = False, flag_direct_io: bool = False) -> pd.DataFrame:
    surfix = SUFFIX
    base_prompt = '''
 Premise: {premise}
 Question: {question}
 A. {option_a}
 B. {option_b}
 '''
    
    if flag_direct_io:
        base_prompt += f"{DIRECT_IO_PROMPT} Answer: A or Answer: B. Do not use any other format."
    if not flag_direct_io:
        base_prompt += f"{COT_PROMPT} The answer format is #### A or #### B."
    
    if flag_few_shot:
        if flag_direct_io:
            examples = df.sample(2).apply(lambda row: base_prompt.format(
                premise=row['p'],
                question=f"What was the {row['asks-for']}?",
                option_a=row['a1'],
                option_b=row['a2']
            ) + f"\nAnswer: {chr(65 + row['most-plausible-alternative'] - 1)}", axis=1).str.cat(sep="\n\n")
        else:
            examples = get_cot_prompt("copa")  # Assuming this function exists as in the original code
        base_prompt = f"\nHere are some examples:\n\n{examples}\n MY QUESTION:\n{base_prompt}"
    
    base_prompt = surfix + base_prompt
    
    df['prompt'] = df.apply(lambda row: base_prompt.format(
        premise=row['p'],
        question=f"What was the {row['asks-for']}?",
        option_a=row['a1'],
        option_b=row['a2']
    ), axis=1)
    
    df['label'] = df['most-plausible-alternative'].map({1: 'A', 2: 'B'})
    
    return df[['prompt', 'label']]

def process_crab(df: pd.DataFrame, flag_few_shot: bool = False, flag_direct_io: bool = False) -> pd.DataFrame:
    surfix = SUFFIX
    base_prompt = '''
    Context:
    Article A: {article_a}
    Article B: {article_b}

    Event 1: {event_a}
    Event 2: {event_b}

    How much did event 1 cause event 2 to happen?
    [A] High causality: Event 1 is definitely responsible for Event 2.
    [B] Medium causality: Event 1 might have been responsible for Event 2.
    [C] Low causality: The context gives a little indication that there is a connection between the two events, but background info might suggest a low causal connection.
    [D] No causality: Events are somehow related but definitely NOT causally related.
    '''

    if flag_direct_io:
        base_prompt += f"{DIRECT_IO_PROMPT} Answer: A, or Answer: B, or Answer: C, or Answer: D. Do not use any other format."
    else:
        base_prompt += f"{COT_PROMPT} The answer format is #### A or #### B or #### C or #### D."

    if flag_few_shot:
        if flag_direct_io:
            examples = df.sample(2).apply(lambda row: base_prompt.format(
                article_a=row['article_a'],
                article_b=row['article_b'],
                event_a=row['event_a'],
                event_b=row['event_b']
            ) + f"\nAnswer: {row['class']}", axis=1).str.cat(sep="\n\n")
        else:
            examples = get_cot_prompt("crab")  # Assuming this function exists
        base_prompt = f"\nHere are some examples:\n\n{examples}\nMY QUESTION:\n{base_prompt}"

    base_prompt = surfix + base_prompt

    df['prompt'] = df.apply(lambda row: base_prompt.format(
        article_a=row['article_a'],
        article_b=row['article_b'],
        event_a=row['event_a'],
        event_b=row['event_b']
    ), axis=1)

    df['label'] = df['class']
    return df[['prompt', 'label']]

def process_crass(df: pd.DataFrame, flag_few_shot: bool = False, flag_direct_io: bool = False) -> pd.DataFrame:
    surfix = SUFFIX
    base_prompt = '''
Question: {question}
'''
    def construct_mcq(row):
        question = row['examples']['input']
        options = list(row['examples']['target_scores'].keys())
        correct_answer_text = next(option for option, score in row['examples']['target_scores'].items() if score == 1)
        mcq = f"{question}\n\n"
        for i, option in enumerate(options):
            letter = chr(65 + i)
            mcq += f"{letter}) {option}\n"
            if option == correct_answer_text:
                correct_answer_letter = letter
        return mcq, correct_answer_letter

    if flag_direct_io:
        base_prompt += f'{DIRECT_IO_PROMPT} Answer: A, or Answer: B, or Answer: C, or Answer: D. Do not use any other format.'
    if not flag_direct_io:
        base_prompt += f"{COT_PROMPT} The answer format is #### A or #### B or #### C or #### D."

    if flag_few_shot:
        if flag_direct_io:
            examples = df.sample(2).apply(lambda row: base_prompt.format(question=construct_mcq(row)[0]) + f"\nAnswer: {construct_mcq(row)[1]}", axis=1).str.cat(sep="\n\n")
        else:
            examples = get_cot_prompt("crass")  # Assuming this function exists as in the original code
        base_prompt = f"\nHere are some examples:\n\n{examples}\n MY QUESTION:\n{base_prompt}"

    base_prompt = surfix + base_prompt



    df['prompt'] = df.apply(lambda row: base_prompt.format(question=construct_mcq(row)[0]), axis=1)
    df['label'] = df.apply(lambda row: construct_mcq(row)[1], axis=1)

    return df[['prompt', 'label']]

def process_e_care(df: pd.DataFrame, flag_few_shot: bool = False, flag_direct_io: bool = False) -> pd.DataFrame:
    surfix = SUFFIX
    base_prompt = '''
{premise}
A. {hypothesis1}
B. {hypothesis2}
Which of the following is more likely to be true?
'''

    if flag_direct_io:
        base_prompt += f"{DIRECT_IO_PROMPT} Answer: A or Answer: B. Do not use any other format."
    if not flag_direct_io:
        base_prompt += f"{COT_PROMPT} The answer format is #### A or #### B."

    if flag_few_shot:
        if flag_direct_io:
            examples = df.sample(2).apply(lambda row: base_prompt.format(**row) + f"\nAnswer: {'A' if row['label'] == 0 else 'B'}", axis=1).str.cat(sep="\n\n")
        else:
            examples = get_cot_prompt("e_care")  # Assuming this function exists as in the original code
        base_prompt = f"\nHere are some examples:\n\n{examples}\n MY Question\n{base_prompt}"

    base_prompt = surfix + base_prompt

    df['prompt'] = df.apply(lambda row: base_prompt.format(**row), axis=1)
    df['label'] = df['label'].apply(lambda x: "A" if x == 0 else "B")

    return df[['prompt', 'label']]

def process_moca(df: pd.DataFrame, flag_few_shot: bool = False, flag_direct_io: bool = False) -> pd.DataFrame:
    surfix = SUFFIX
    base_prompt = '''
Story: {story} 
Question: {question}
'''

    if flag_direct_io:
        base_prompt += f"{DIRECT_IO_PROMPT} Answer: Yes or Answer: No. Do not use any other format."
    if not flag_direct_io:
        base_prompt += f"{COT_PROMPT} The answer format is #### Yes or #### No."

    if flag_few_shot:
        if flag_direct_io:
            examples = df.sample(2).apply(lambda row: base_prompt.format(**row) + f"\nAnswer: {row['answer']}", axis=1).str.cat(sep="\n\n")
        else:
            examples = get_cot_prompt("moca")  # Assuming this function exists as in the original code
        base_prompt = f"\nHere are some examples:\n\n{examples}\n MY QUESTION:\n{base_prompt}"

    base_prompt = surfix + base_prompt

    df['prompt'] = df.apply(lambda row: base_prompt.format(**row), axis=1)
    df['label'] = df['answer'].apply(lambda x: x.lower())

    return df[['prompt', 'label']]

def process_pain(df: pd.DataFrame, flag_few_shot: bool = False, flag_direct_io: bool = False) -> pd.DataFrame:
    surfix = SUFFIX
    base_prompt = '''
    Question: {query}
    '''

    if flag_direct_io:
        base_prompt += f"{DIRECT_IO_PROMPT} Answer: True or Answer: False. Do not use any other format."
    if not flag_direct_io:
        base_prompt += f"{COT_PROMPT} The answer format is #### True or #### False."

    if flag_few_shot:
        if flag_direct_io:
            examples = df.sample(2).apply(lambda row: base_prompt.format(query=row['Query']) + f"\nAnswer: {'True' if row['Answer'] else 'False'}", axis=1).str.cat(sep="\n\n")
        else:
            examples = get_cot_prompt("pain")  # Assuming this function exists as in the original code
        base_prompt = f"\nHere are some examples:\n\n{examples}\n “MY QUESTION:”\n{base_prompt}"

    base_prompt = surfix + base_prompt

    df['prompt'] = df['Query'].apply(lambda q: base_prompt.format(query=q))
    df['label'] = df['Answer'].apply(lambda x: "True" if x else "False")

    return df[['prompt', 'label']]

def process_tram(df: pd.DataFrame, flag_few_shot: bool = False, flag_direct_io: bool = False) -> pd.DataFrame:
    surfix = SUFFIX
    base_prompt = '''
{Premise}
{Question}
A. {Option A}
B. {Option B}
'''

    if flag_direct_io:
        base_prompt += f"{DIRECT_IO_PROMPT} Answer: A or Answer: B. Do not use any other format."
    if not flag_direct_io:
        base_prompt += f"{COT_PROMPT} The answer format is #### A or #### B."

    if flag_few_shot:
        if flag_direct_io:
            examples = df.sample(2).apply(lambda row: base_prompt.format(**row) + f"\nAnswer: {row['Answer']}\n", axis=1).str.cat(sep="\n\n")
        else:
            examples = get_cot_prompt("tram")  # Assuming this function exists as in the original code
        base_prompt = f"\nHere are some examples:\n\n{examples}\n “MY QUESTION:”\n{base_prompt}"

    base_prompt = surfix + base_prompt

    df['prompt'] = df.apply(lambda row: base_prompt.format(**row), axis=1)
    df['label'] = df['Answer']

    return df[['prompt', 'label']]

def process_corr2cause(df: pd.DataFrame, flag_few_shot: bool = False, flag_direct_io: bool = False) -> pd.DataFrame:
    surfix = SUFFIX
    base_prompt = '''
Question: {premise}
Determine the truth value the following statement: {hypothesis}
'''

    if flag_direct_io:
        base_prompt += f"{DIRECT_IO_PROMPT}, Answer: neutral or Answer: contradiction or Answer: entailment. Do not use any other format."
    if not flag_direct_io:
        base_prompt += f"{COT_PROMPT} The answer format is #### neutral or #### contradiction or #### entailment."

    if flag_few_shot:
        if flag_direct_io:
            examples = df.sample(2).apply(lambda row: base_prompt.format(**row) + f"\nAnswer: {row['relation']}", axis=1).str.cat(sep="\n\n")
        else:
            examples = get_cot_prompt("corr2cause")  # Assuming this function exists as in the original code
        base_prompt = f"\nHere are some examples:\n\n{examples}\n “MY QUESTION:”\n{base_prompt}"

    base_prompt = surfix + base_prompt

    df['prompt'] = df.apply(lambda row: base_prompt.format(**row), axis=1)
    df['label'] = df['relation']

    return df[['prompt', 'label']]

# Main evaluation function
def evaluate_dataset(dataset_name: str, file_path: str, process_func: Callable, parse_options: List[str], sample_size: int = 100, start = 0, end = 10):
    print(f"Evaluating {dataset_name}...")
    seed = 42
    # Load and process data
    


    few_shot_flag_list = [True, False]
    direct_io_flag_list = [True, False]
    prompt_list = []
    for flag_few_shot_flag in few_shot_flag_list:
        for direct_io_flag in direct_io_flag_list:
            df = load_json_file(file_path)
            df = process_func(df, flag_few_shot=flag_few_shot_flag, flag_direct_io=direct_io_flag)
            if sample_size < len(df):
                df = df.sample(sample_size, random_state=seed)
            prompt_list.append({f"few_shot_{flag_few_shot_flag}_direct_io_{direct_io_flag}": df['prompt'].tolist()})
            # Evaluate each model
            model_list_name = list(model_list.keys())
            model_list_name_sorted = sorted(model_list_name)
            print(model_list_name_sorted)
            print(start,end)
            print(model_list_name_sorted[start:end])
            
            for model_name in model_list_name_sorted[start:end]:
                print(f"Evaluating {model_name}...")
                API_BASE_URL = model_list[model_name]["API_BASE_URL"]
                API_KEY = model_list[model_name]["API_KEY"]
                model_name_wrap = {"Pro/meta-llama/Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
                                           "Pro/meta-llama/Meta-Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                                           "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": "meta-llama/Meta-Llama-3.1-70B-Instruct",
                                             "meta-llama/Meta-Llama-3-70B-Instruct-Turbo": "meta-llama/Meta-Llama-3-70B-Instruct",
                                             "Pro/google/gemma-2-9b-it": "google/gemma-2-9b-it",}
                batcher = GPTBatcher(api_key=API_KEY, api_base_url=API_BASE_URL, model_name=model_name,num_workers=10)
                
                   
                try:    
                    #check if the model's experiment is finished
                    if model_name in model_name_wrap:
                        model_name = model_name_wrap[model_name]
                    file_name =f'./result/eval_result_seed_{seed}_sample_num_{sample_size}.json'
                    def check_experiment_finished(file_name):
                        
                        if not os.path.exists(file_name):
                            return False
                        with open(file_name, 'r') as f:
                            results = json.load(f)
                        for result in results:
                            if result["model_name"] == model_name and result["seed"] == seed and result["sample_size"] == sample_size and result["few_shot_flag"] == flag_few_shot_flag and result["direct_io_flag"] == direct_io_flag and result["dataset_name"] == dataset_name:
                                return True
                        return False
                    if check_experiment_finished(file_name):
                        print(f"Model {model_name} has been evaluated, skip")
                        continue
                    # Get model responses
                    question_list = df['prompt'].tolist()
                    result = batcher.handle_message_list(question_list)
                    df[model_name] = result
                    
                    # Save raw results
                    model_name_split = model_name.split('/')[-1]
                    #check if the directory exists
                    if not os.path.exists(f'./dataset/{dataset_name}/llm_result'):
                        os.makedirs(f'./dataset/{dataset_name}/llm_result')
                                    # Parse and evaluate answers
                    df[f"{model_name}_label"] = df[model_name].apply(lambda x: parse_answer(x, parse_options))
                except Exception as e:
                    print(f"Error while evaluating {model_name}: {e}")
                    import time
                    time.sleep(60)
                    continue

                df['correct'] = df[f"{model_name}_label"] == df['label']
                accuracy = df['correct'].mean()
                result = df[[model_name,'label',f"{model_name}_label","prompt"]]
                result.to_json(f'./dataset/{dataset_name}/llm_result/{model_name_split}_seed_{seed}_sample_num_{sample_size}_few_shot_{flag_few_shot_flag}_direct_io_{direct_io_flag}.json', lines=True, orient='records')
                print(f"Accuracy for {model_name}: {accuracy:.2f}")
                
                # Save evaluation results
                result_dict = {"model_name": model_name, "accuracy": accuracy, "seed": seed, "sample_size": sample_size, "few_shot_flag": flag_few_shot_flag, "direct_io_flag": direct_io_flag, "dataset_name": dataset_name}
                append_results(result_dict, f'./dataset/{dataset_name}/eval_result_seed_{seed}_sample_num_{sample_size}.json')
                append_results(result_dict, f'./result/eval_result_seed_{seed}_sample_num_{sample_size}.json')
    for prompt in prompt_list:
        prompt_name = list(prompt.keys())[0]
        df[prompt_name] = list(prompt.values())[0]
           
    # Save final results
    df.to_json(f'./dataset/{dataset_name}/{dataset_name}_results_seed_{seed}_sample_num_{sample_size}.json', lines=True, orient='records')

# Main execution
if __name__ == "__main__":
    #parse args with start and end
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10)


    start = parser.parse_args().start
    end = parser.parse_args().end

    print(f"Start from {start} to {end}")
    datasets = [
        ("cladder", "./dataset/cladder/data_full_v1.5_default.jsonl", process_cladder, ["yes", "no"]),
        ("copa", "./dataset/copa/copa-test.json", process_copa, ["A", "B"]),
        ("crab", "./dataset/crab/pairwise_causality.jsonl", process_crab, ["A", "B", "C", "D"]),
        ("crass", "./dataset/crass/CRASS_FTM_BIG_Bench_data_set.json", process_crass, ["A", "B", "C", "D"]),
        ("e_care", "./dataset/e_care/e_care_dev_full.jsonl", process_e_care, ["A", "B"]),
        ("moca", "./dataset/moca/causal_dataset_v1.jsonl", process_moca, ["yes", "no"]),
        ("pain", "./dataset/pain/pain.json", process_pain, ["True", "False"]),
        ("tram", "./dataset/tram/tram_merge.jsonl", process_tram, ["A", "B"]),
        ("corr2cause", "./dataset/corr2cause/perturbation_by_paraphrasing_test.json", process_corr2cause, ["neutral", "contradiction", "entailment"]),
    ]
    
    for dataset_name, file_path, process_func, parse_options in datasets:
        evaluate_dataset(dataset_name, file_path, process_func, parse_options,sample_size=500,start=start,end=end)

    print("Evaluation complete for all datasets.")