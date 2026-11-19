import numpy as np
import torch
from pathlib import Path
from mario_net import MarioNet
from replay_buffer import FastReplayBuffer

class Mario:
    """
    Agente Mario entrenado mediante Double Deep Q-Networks (DDQN).
    """
    def __init__(self, state_dim, action_dim, save_dir, checkpoint_path=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Red neuronal dual (Online y Target)
        self.net = MarioNet(self.state_dim, self.action_dim).float().to(device=self.device)

        # Hiperparametros de exploracion
        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1

        # Pasos y guardado
        self.curr_step = 0
        self.save_every = 5e4  # guardar checkpoint cada 50,000 pasos

        # Replay Buffer y tamano de batch
        self.memory = FastReplayBuffer(capacity=100000, device=self.device)
        self.batch_size = 32

        # Descuento y actualizacion DDQN
        self.gamma = 0.9
        self.burnin = 1e4        # 10,000 pasos iniciales antes de entrenar
        self.learn_every = 3     # entrenar cada 3 pasos
        self.sync_every = 1e4    # sincronizar Q_target cada 10,000 pasos

        # Optimizador y Loss
        self.optimizer = torch.optim.Adam(self.net.online.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        if checkpoint_path and Path(checkpoint_path).exists():
            self.load(checkpoint_path)

    def act(self, state):
        """
        Elige una accion con politica epsilon-greedy.
        """
        # Explorar (accion aleatoria)
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        # Explotar (mejor accion predicha por Q_online)
        else:
            if hasattr(state, "__array__"):
                state_arr = state.__array__()
            elif isinstance(state, tuple):
                state_arr = state[0].__array__() if hasattr(state[0], "__array__") else np.array(state[0])
            else:
                state_arr = np.array(state)

            state_tensor = torch.as_tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action_values = self.net(state_tensor, model="online")
            action_idx = torch.argmax(action_values, dim=1).item()

        # Decaimiento del factor de exploracion
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1

        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Guarda la transicion en el buffer de memoria.
        """
        self.memory.add(state, next_state, action, reward, done)

    def recall(self):
        """
        Recupera un lote aleatorio del buffer de memoria.
        """
        return self.memory.sample(self.batch_size)

    def td_estimate(self, state, action):
        """
        Prediccion actual: Q_online(s, a)
        """
        current_Q = self.net(state, model="online")[
            torch.arange(0, self.batch_size, device=self.device), action
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """
        Objetivo DDQN:
        a' = argmax_a Q_online(s', a)
        target = r + gamma * (1 - done) * Q_target(s', a')
        """
        next_state_Q_online = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q_online, dim=1)

        next_Q_target = self.net(next_state, model="target")[
            torch.arange(0, self.batch_size, device=self.device), best_action
        ]
        return (reward + (1.0 - done) * self.gamma * next_Q_target).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate,
                curr_step=self.curr_step
            ),
            save_path
        )
        print(f"--> Checkpoint guardado en {save_path} (paso {self.curr_step})")

    def load(self, load_path):
        checkpoint = torch.load(load_path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model"])
        self.exploration_rate = checkpoint.get("exploration_rate", self.exploration_rate)
        self.curr_step = checkpoint.get("curr_step", self.curr_step)
        print(f"--> Modelo cargado desde {load_path} (paso {self.curr_step}, epsilon {self.exploration_rate:.4f})")

    def learn(self):
        """
        Ejecuta un ciclo de aprendizaje DDQN si se cumplen las condiciones de paso.
        """
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0 and self.curr_step > 0:
            self.save()

        # Esperar a llenar el buffer inicial
        if len(self.memory) < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_est, td_tgt)
        return td_est.mean().item(), loss
