import torch
from torch import nn,optim,Tensor
from model import LinearQNet


class Trainer:

    def __init__(self,
        model : LinearQNet,
        learning_rate : float = 0.001,
        gamma : float = 0.9,
        optimizer : optim.Optimizer = None,
        criterion : nn.Module = None
    ) -> None:
        self.model = model
        self.gamma = gamma
        self.optimizer = optimizer if optimizer is not None else optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = criterion if criterion is not None else nn.MSELoss()

    def train_step(self, state : Tensor, action : Tensor, reward : Tensor, next_state : Tensor, game_over : Tensor) -> Tensor:

        y = self.model(state)
        y_next = self.model(next_state)

        Q_new = torch.where(game_over, reward, reward + self.gamma * torch.max(y_next, dim=-1))
        target = y.clone()  
        mask = torch.eye(self.model.output_size, dtype=torch.bool)[action]
        target[mask] = Q_new
        
        loss = self.criterion(y, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
