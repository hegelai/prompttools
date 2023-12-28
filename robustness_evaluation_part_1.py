import promptbench as pb
from promptbench.models import LLMModel
from promptbench.attack import Attack
import numpy as np
import textattack
import tensorflow as tf
import tensorflow_hub as tb_hub

model = LLMModel(name="gpt-3.5-turbo")
dataset = pb.DatasetLoader.load_dataset("sst2")

dataset = dataset[:10]

def proj_func(pred):
    mapping = {
        "positive": 1,
        "negative": 0
    }
    return mapping.get(pred, -1)

def eval_func(prompt, dataset, model):
    preds = []
    labels = []
    for d in dataset:
        input_text = pb.InputProcess.basic_format(prompt, d)
        raw_output = model(input_text)

        output = pb.OutputProcess.cls(raw_output, proj_func)
        preds.append(output)

        labels.append(d["label"])
    return pb.Eval.compute_cls_accuracy(preds,labels)

unmodifiable_words = ["positive\'", "negative\'", "content"]
attack = Attack(model, "stresstest", dataset, prompt, eval_func, unmodifiable_words, verbose = True)
print(attack.attack())
