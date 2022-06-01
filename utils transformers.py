from IPython.display import clear_output
import matplotlib.pyplot as plt
from typing import Tuple, List
import torch
from collections import defaultdict
from collections.abc import Iterable


def tokenize(sample, tokenizer, MAX_LEN):
    return tokenizer(
        sample['sentence'], 
        truncation=True, 
        padding='max_length', 
        max_length=MAX_LEN, 
        return_tensors='pt'
    )

def to_cuda(batch: dict)-> dict:
    return {key: value.cuda() for key, value in batch.items()}

class Logger:
    def __init__(self, label):
        self.stats = defaultdict(list)
        self.label = label
        
    def __getitem__(self, name: str) -> List:
        return self.stats[name]
    
    def add_stat(self, name: str) -> None:
        self.stats[name] = []
    
    def log(self, new_stats: dict) -> None:
        for name in new_stats:
            self.stats[name].append(new_stats[name])

def plot_results(
    loggers: Logger,
    pure_markers=True
)-> None:

    clear_output()

    if  not isinstance(loggers, Iterable):
        loggers = [loggers]

    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    
    markers = ['p', 'x', '*', '.', 'P', 'X']
    colors  = ['g', 'r', 'c', 'y', 'b', 'm']

    for i, logger in enumerate(loggers):
        ax[0, 0].plot(
            logger['train_loss'], marker=markers[i % 6] if pure_markers else 'o', 
            color=colors[i % 6], label=logger.label, alpha=0.5
        )
        ax[0, 1].plot(
            logger['train_metric'], marker=markers[i % 6] if pure_markers else 'o', 
            color=colors[i % 6], label=logger.label, alpha=0.5
        )
        ax[1, 0].plot(
            logger['test_loss'], marker=markers[i % 6] if pure_markers else 'o', 
            color=colors[i % 6], label=logger.label, alpha=0.5
        )
        ax[1, 1].plot(
            logger['test_metric'], marker=markers[i % 6] if pure_markers else 'o', 
            color=colors[i % 6], label=logger.label, alpha=0.5
        )
    
    for i in range(2):
        for j in range(2):
            ax[i, j].grid(True)
    
    ax[0, 0].set_xlabel('Log step')
    ax[0, 1].set_xlabel('Log step')
    ax[1, 0].set_xlabel('Validation step')
    ax[1, 1].set_xlabel('Validation step')
    
    ax[0, 0].set_ylabel('Loss')
    ax[0, 1].set_ylabel('Metric')
    ax[1, 0].set_ylabel('Loss')
    ax[1, 1].set_ylabel('Metric')

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    plt.show()
    
class HuggingMetric:
    def __init__(self, metric):
        self.metric = metric
        
    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        predictions = torch.argmax(logits, dim=-1)
        return float(
            self.metric.compute(predictions=predictions, references=targets)['matthews_correlation']
        )

def freeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return model
    