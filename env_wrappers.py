from collections import deque
import numpy as np
import torch
from torchvision import transforms as T
import gym
from gym.spaces import Box

class SkipFrame(gym.Wrapper):
    """
    Repite la misma accion durante `skip` frames consecutivos y acumula los rewards.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        obs = res[0] if isinstance(res, tuple) else res
        info = res[1] if isinstance(res, tuple) and len(res) > 1 else {}
        return obs, info

    def step(self, action):
        total_reward = 0.0
        step_result = self.env.step(action)

        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
            terminated, truncated = done, False

        total_reward += reward

        for _ in range(self._skip - 1):
            if terminated or truncated:
                break
            step_result = self.env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
                terminated, truncated = done, False
            total_reward += reward

        return obs, total_reward, terminated, truncated, info

class GrayScaleObservation(gym.Wrapper):
    """
    Convierte la observacion RGB [H, W, 3] en escala de grises [H, W].
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def _transform_obs(self, observation):
        # Transponer [H, W, C] -> [C, H, W]
        obs = np.transpose(observation, (2, 0, 1))
        obs = torch.tensor(obs.copy(), dtype=torch.float)
        transform = T.Grayscale()
        obs = transform(obs).squeeze(0)  # [H, W]
        return obs.numpy()

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        obs = res[0] if isinstance(res, tuple) else res
        info = res[1] if isinstance(res, tuple) and len(res) > 1 else {}
        return self._transform_obs(obs), info

    def step(self, action):
        res = self.env.step(action)
        if len(res) == 5:
            obs, reward, terminated, truncated, info = res
        else:
            obs, reward, done, info = res
            terminated, truncated = done, False
        return self._transform_obs(obs), reward, terminated, truncated, info

class ResizeObservation(gym.Wrapper):
    """
    Redimensiona la observacion a [shape, shape] (84, 84).
    """
    def __init__(self, env, shape=84):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        self.observation_space = Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

    def _transform_obs(self, observation):
        obs = torch.tensor(observation.copy(), dtype=torch.float).unsqueeze(0)
        transforms = T.Compose([
            T.Resize(self.shape, antialias=True),
            T.Normalize(0, 255)
        ])
        obs = transforms(obs).squeeze(0)
        return obs.numpy()

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        obs = res[0] if isinstance(res, tuple) else res
        info = res[1] if isinstance(res, tuple) and len(res) > 1 else {}
        return self._transform_obs(obs), info

    def step(self, action):
        res = self.env.step(action)
        if len(res) == 5:
            obs, reward, terminated, truncated, info = res
        else:
            obs, reward, done, info = res
            terminated, truncated = done, False
        return self._transform_obs(obs), reward, terminated, truncated, info

class CustomFrameStack(gym.Wrapper):
    """
    Apila `num_stack` frames consecutivos produciendo un tensor/array de shape [num_stack, H, W].
    """
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        obs = res[0] if isinstance(res, tuple) else res
        info = res[1] if isinstance(res, tuple) and len(res) > 1 else {}
        for _ in range(self.num_stack):
            self.frames.append(obs)
        stacked = np.stack(self.frames, axis=0)
        return stacked, info

    def step(self, action):
        res = self.env.step(action)
        if len(res) == 5:
            obs, reward, terminated, truncated, info = res
        else:
            obs, reward, done, info = res
            terminated, truncated = done, False

        self.frames.append(obs)
        stacked = np.stack(self.frames, axis=0)
        return stacked, reward, terminated, truncated, info
