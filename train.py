import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils import *
from tqdm import tqdm
import numpy as np

def train_epoch(
    model: nn.Module, 
    optim: torch.optim, 
    train_loader: DataLoader, 
    valid_loader: DataLoader,
    logger: Logger,
    metric,
    scheduler: torch.optim.lr_scheduler=None,
    log_step: int=None,
    valid_step: int=None,
)-> None:
    
    if log_step is None:
        log_step = train_loader.__len__() // 10
    
    if valid_step is None:
        valid_step = train_loader.__len__() // 4
    
    train_metric, train_loss = [], []
    
    for step, batch in tqdm(enumerate(train_loader)):
        batch = to_cuda(batch)
        out = model(**batch)
        
        loss = out['loss']
        loss.backward()
        
        optim.step()
        scheduler.step()
        
        optim.zero_grad()
        
        train_metric.append(
            metric(
                out['logits'].detach().cpu(),
                batch['labels'].cpu()
            )
        )
        train_loss.append(loss.detach().cpu().numpy())
        
        if step % log_step == 0:
            logger.log({
                'train_loss': np.mean(train_loss),
                'train_metric': np.mean(train_metric)
            })
            
            train_metric, train_loss = [], []
                    
        if step % valid_step == 0:
            with torch.no_grad():
                model.eval()
                metrics, losses = [], []
                for batch in tqdm(valid_loader):
                    batch = to_cuda(batch)
                    out = model(**batch)
                    
                    loss = out['loss']
                    
                    metrics.append(
                        metric(
                            out['logits'].detach().cpu(),
                            batch['labels'].cpu()
                            
                        )
                    )
                    losses.append(loss.detach().cpu().numpy())
                
                logger.log({
                        'test_loss': np.mean(losses),
                        'test_metric': np.mean(metrics)
                })
                model.train()

                
def Distil_loss(
        output: torch.Tensor,
        output_teacher: torch.Tensor,
        T: int,
) -> float:
    '''
    Distil loss = KL(output, output_teacher)
    '''
    return F.kl_div(
               F.log_softmax(output / T, dim=-1),
               F.softmax(output_teacher / T, dim=-1),
               reduction="batchmean"
           ) * (T ** 2)


def train_epoch_distil(
    alpha: float,       # Коэффициент, регулирующий силу дистилляции 
    T: int,             # Температура в KL
    model: nn.Module, 
    teacher_model: nn.Module,
    optim: torch.optim, 
    train_loader: DataLoader, 
    valid_loader: DataLoader,
    logger: Logger,
    metric,
    scheduler: torch.optim.lr_scheduler=None,
    log_step: int=None,
    valid_step: int=None,
) -> None:
    teacher_model.eval()
    
    if log_step is None:
        log_step = train_loader.__len__() // 10
    
    if valid_step is None:
        valid_step = train_loader.__len__() // 4
    
    train_metric, train_loss = [], []
    
    for step, batch in tqdm(enumerate(train_loader)):
        batch = to_cuda(batch)
        out = model(**batch)
        
        with torch.no_grad():
            out_teacher = teacher_model(**batch)
            
        loss = out['loss'] + alpha * Distil_loss(out['logits'], out_teacher['logits'], T)
        loss.backward()
        
        optim.step()
        scheduler.step()
        
        optim.zero_grad()
        
        train_metric.append(
            metric(
                out['logits'].detach().cpu(),
                batch['labels'].cpu()
            )
        )
        train_loss.append(loss.detach().cpu().numpy())
        
        if step % log_step == 0:
            logger.log({
                'train_loss': np.mean(train_loss),
                'train_metric': np.mean(train_metric)
            })
            
            train_metric, train_loss = [], []
                    
        if step % valid_step == 0:
            with torch.no_grad():
                model.eval()
                metrics, losses = [], []
                for batch in tqdm(valid_loader):
                    batch = to_cuda(batch)
                    out = model(**batch)
                    
                    loss = out['loss']
                    
                    metrics.append(
                        metric(
                            out['logits'].detach().cpu(),
                            batch['labels'].cpu()
                            
                        )
                    )
                    losses.append(loss.detach().cpu().numpy())
                
                logger.log({
                        'test_loss': np.mean(losses),
                        'test_metric': np.mean(metrics)
                })
                model.train()