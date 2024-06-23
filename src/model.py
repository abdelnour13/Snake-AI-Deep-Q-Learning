import torch
import os
from torch import nn
from torch import Tensor

class LinearQNet(nn.Module):

    def __init__(self, input_size: int, output_size: int, hidden_size : int = 256) -> None:

        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x : Tensor) -> Tensor:
            
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
    
        return x
    
    def predict(self, x : Tensor) -> Tensor:
        return self.forward(x).argmax(dim=-1)
    
    def save(self, name : str) -> None:
        file_path = os.path.join('.','checkpoints', name)
        return torch.save(self.state_dict(), file_path)