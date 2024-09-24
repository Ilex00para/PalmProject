import torch
from torch import nn

def Train_Step(model: nn.Module,
                  x: torch.Tensor,
                  y: torch.Tensor,
                  loss_function: nn.Module, 
                  optimizer,
                  device = None,
                  lr_scheduler = None) -> float:
        #model to train
        model.train()
   
        x,y  = x.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(x)
        
        try:
            loss = loss_function(output, y)
        except:
            y.view(-1,1)
            loss = loss_function(output, y)

        loss.backward()

        optimizer.step()

        if lr_scheduler is not None:
            if type(lr_scheduler) is torch.optim.lr_scheduler.CyclicLR:
                lr_scheduler.step()

        torch.cuda.empty_cache()

        return loss


class Trainer:
    def __init__(self, model, optimizer, loss_function, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device

    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        x, y = x.to(self.device), y.to(self.device)

        prediction = self.model(x)
        loss = self.loss_function(prediction, y)
        loss.backward()
        self.optimizer.step()

        return loss

class Trainer_LRScheduled(Trainer):
    def __init__(self, model, optimizer, loss_function, device, lr_scheduler):
        super().__init__(model, optimizer, loss_function, device)
        self.lr_scheduler = lr_scheduler

    def train_step(self, x, y):
        loss = super().train_step(x, y)
        self.lr_scheduler.step()
        return loss

