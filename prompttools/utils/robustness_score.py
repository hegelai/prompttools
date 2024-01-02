import promptbanch as pb 
from promptbench.attack import Attack
from promptbench.models import LLMModel 
import textattack
import tensorflow as tf   
import tensorflow_hub as tf_hub   
import numpy as np
from sklearn.metrics import accuracy_score 
from pb.metrics import bleu_score
from pb.metrics import math_score

LABEL_SET = {
    # 'positive\'', 'negative\'' is used for label constraint due to a bug of TextAttack repo.
    'sst2': ['positive', 'negative', 'positive\'', 'negative\'', '0', '1', '0\'', '1\''],
    'mnli': MNLI_LABEL,
    'mnli_mismatched': MNLI_LABEL,
    'mnli_matched': MNLI_LABEL,
    'qqp': EQ_LABEL,
    'qnli': ENTAIL_LABEL,
    'rte': ENTAIL_LABEL,
    'cola': ['unacceptable', 'acceptable', 'unacceptable\'', 'acceptable\''],
    'mrpc': EQ_LABEL,
    'wnli': ENTAIL_LABEL,
    'mmlu': ['A', 'B', 'C', 'D', 'A\'', 'B\'', 'C\'', 'D\'', 'a', 'b', 'c', 'd', 'a\'', 'b\'', 'c\'', 'd\''],
    # do not change the word 'nothing' in prompts.
    'squad_v2': ['unanswerable', 'unanswerable\''],
    'iwslt': ['translate', 'translate\''],
    'un_multi': ['translate', 'translate\''],
    'math': ['math', 'math\''],
    'bool_logic': ['True', 'False', 'True\'', 'False\'', "bool", "boolean", "bool\'", "boolean\'"],
    'valid_parentheses': ['Valid', 'Invalid', 'Valid\'', 'Invalid\'', 'matched', 'matched\'', 'valid', 'invalid', 'valid\'', 'invalid\''],
}

tasks = ["classification","translation","math"]
def eval_func(prompts, dataset, model, gts, task):
    preds = []
    scores = {}
    for prompt in prompts:
        for d in dataset:
            input_text = pb.InputProcess.basic_format(prompt, d)
            raw_output = model(input_text)
            preds.append(output)
        if task == "classification":
            scores[prompt] = accuracy_score(gts,preds)
        elif task == "translation":
            scores[prompt] = bleu_score(gts,preds)
        elif task == "math":
            scores[prompt] = math_score(dataset,gts,preds)
    return scores

def calculate_robustness_score(task = None,model,prompts,ground_truth_labels,is_attack = False,attack = 'stresstest',dataset, percentage_dataset = 1.0):
    
    if task is None or task not in tasks:
        raise Exception('Please enter one of the following tasks: \
                        (classification,transaltion,math).')
    
    try:
        llm = LLMModel(model)
    except NameError:
        print("Unable to support this model: Please input one of the following models: ",pb.SUPPORTED_MODELS)
    
    try:
        model_dataset = pd.DatasetLoader.load_dataset(dataset)
    except NameError:
        print("Unable to support this dataset: Please input one of the following dataset: ",pb.SUPPORTED_DATASETS)

    if is_attack:
        dataset = dataset[:int(percentage*len(dataset))]
        unmodifiable_words = LABEL_SET[dataset]
        attack = Attack(llm, attack, dataset, prompt, eval_func, unmodifiable_words, verbose = True)

        print(attack.attack())
        return 
        
    return eval_func(prompts, dataset, model, gts, task)



