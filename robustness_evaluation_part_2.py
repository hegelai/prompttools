import promptbench as pb
from promptbench.models import LLMModel
from tqdm import tqdm

model_t5 = LLMModel(name="gpt-3.5-turbo",max_new_tokens=10, temperature=0.0001, device='cuda')
dataset = pb.DatasetLoader.load_dataset("sst2")
prompts = pb.Prompt(["Classify the sentence as positive or negative: {content}",
                     "Determine the emotion of the following sentence as positive or negative: {content}"
                     ])
dataset = dataset[:5]

def proj_func(pred):
    mapping = {
        "positive": 1,
        "negative": 0
    }
    return mapping.get(pred, -1)

def eval_func(prompts, dataset, model):
    for prompt in prompts:
        preds = []
        labels = []
        for data in tqdm(dataset):
            input_text = pb.InputProcess.basic_format(prompt, data)
            label = data['label']
            raw_pred = model(input_text)

            pred = pb.OutputProcess.cls(raw_pred, proj_func)
            preds.append(pred)
            labels.append(label)

    score = pb.Eval.compute_cls_accuracy(preds, labels)        
    return score

print(eval_func(prompts, dataset, model))
