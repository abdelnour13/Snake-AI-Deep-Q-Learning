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
    
    def load(self, name : str | None = None) -> None:

        if name is None:
            name = next(iter(sorted(os.listdir(os.path.join('.','checkpoints')), reverse=True, key=lambda x : int(x.split('_')[1].split('.')[0]))))
            
        file_path = os.path.join('.','checkpoints', name)
        return self.load_state_dict(torch.load(file_path))
    
class ConvQNet(nn.Module):

    def __init__(self, output_size: int, hidden_size : int = 256) -> None:

        super().__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size

        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=6, stride=3, padding=1)
        self.relu_1 = nn.ReLU()

        self.linear_1 = nn.Linear(1024, hidden_size)
        self.relu_3 = nn.ReLU()

        self.linear_2 = nn.Linear(256, output_size)

    def forward(self, x : Tensor) -> Tensor:
                
        x = self.conv_1(x)
        x = self.relu_1(x)
        
        x = torch.flatten(x, start_dim=1)
    
        x = self.linear_1(x)
        x = self.relu_3(x)
    
        x = self.linear_2(x)
        
        return x
    
    def predict(self, x : Tensor) -> Tensor:
        return self.forward(x).argmax(dim=-1)
    
    def save(self, name : str) -> None:
        file_path = os.path.join('.','checkpoints', name)
        return torch.save(self.state_dict(), file_path)
    
    def load(self, name : str | None = None) -> None:

        if name is None:
            name = next(iter(sorted(os.listdir(os.path.join('.','checkpoints')), reverse=True, key=lambda x : int(x.split('_')[1].split('.')[0]))))
            
        file_path = os.path.join('.','checkpoints', name)
        return self.load_state_dict(torch.load(file_path))