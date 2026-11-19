import torch
from torch import nn

class MarioNet(nn.Module):
    """
    Mini CNN structure:
    input [batch, 4, 84, 84] -> 
    Conv2d(4, 32, k=8, s=4) + ReLU ->
    Conv2d(32, 64, k=4, s=2) + ReLU ->
    Conv2d(64, 64, k=3, s=1) + ReLU ->
    Flatten ->
    Linear(3136, 512) + ReLU ->
    Linear(512, output_dim)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        if h != 84 or w != 84:
            raise ValueError(f"Esperando entrada de resolución 84x84, recibido: ({h}, {w})")

        self.online = self.__build_cnn(c, output_dim)
        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Congelar gradientes de Q_target (se sincroniza periódicamente)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model="online"):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
        else:
            raise ValueError(f"Modelo desconocido: {model}. Debe ser 'online' o 'target'")

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
