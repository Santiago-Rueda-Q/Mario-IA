import random
import numpy as np
import torch

class FastReplayBuffer:
    """
    Buffer de experiencia circular de alto rendimiento.
    Guarda transiciones (state, next_state, action, reward, done)
    y permite muestreo uniforme para DDQN.
    """
    def __init__(self, capacity=100000, device="cpu"):
        self.capacity = int(capacity)
        self.device = device
        self.buffer = []
        self.position = 0

    def add(self, state, next_state, action, reward, done):
        """
        Almacena una tupla de transicion en el buffer.
        """
        # Convertir a numpy o arrays ligeros si es tensor/lazyframe
        if hasattr(state, "__array__"):
            state = state.__array__()
        elif isinstance(state, torch.Tensor):
            state = state.cpu().numpy()

        if hasattr(next_state, "__array__"):
            next_state = next_state.__array__()
        elif isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()

        transition = (state, next_state, action, reward, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Muestrea un batch aleatorio y lo devuelve como tensores PyTorch en el dispositivo destino.
        """
        batch = random.sample(self.buffer, batch_size)
        states, next_states, actions, rewards, dones = zip(*batch)

        states_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        return states_t, next_states_t, actions_t, rewards_t, dones_t

    def __len__(self):
        return len(self.buffer)
