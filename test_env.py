import warnings
warnings.filterwarnings("ignore")

import sys
import torch
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from env_wrappers import SkipFrame, GrayScaleObservation, ResizeObservation, CustomFrameStack

def main():
    print("=" * 60)
    print(" VERIFICACION DEL ENTORNO MARIO RL")
    print("=" * 60)

    # 1. Comprobar CUDA y PyTorch
    print(f"PyTorch version: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA disponible: {cuda_ok}")
    if cuda_ok:
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # 2. Inicializar Super Mario Bros 1-1
    print("\nInicializando SuperMarioBros-1-1-v0...")
    raw_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True)
    print("Ambiente base creado con exito.")

    # 3. Aplicar JoypadSpace
    env = JoypadSpace(raw_env, [["right"], ["right", "A"]])
    print(f"Espacio de acciones reducido a: {env.action_space.n} (0: right, 1: right+A)")

    # 4. Aplicar Wrappers de observacion
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = CustomFrameStack(env, num_stack=4)

    # 5. Ejecutar un reset
    obs, info = env.reset()
    state = obs[0] if isinstance(obs, tuple) else obs
    print(f"Estado inicial recibido. Shape: {np_shape(state)}")
    assert tuple(np_shape(state)) == (4, 84, 84), f"Esperado shape (4, 84, 84), pero se obtuvo {np_shape(state)}"

    # 6. Ejecutar 50 pasos aleatorios
    print("\nEjecutando 50 pasos con acciones aleatorias...")
    total_reward = 0
    for step in range(50):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        if done:
            print(f"Episodio terminado en el paso {step}")
            env.reset()

    print(f"Prueba completada exitosamente.")
    print(f"Recompensa acumulada en 50 pasos: {total_reward}")
    print(f"Info final de la ROM: {info}")
    print("=" * 60)
    print(" TODO EL STACK ESTA OPERATIVO Y LISTO PARA ENTRENAR!")
    print("=" * 60)
    env.close()

def np_shape(x):
    if hasattr(x, "shape"):
        return list(x.shape)
    if hasattr(x, "__array__"):
        return list(x.__array__().shape)
    import numpy as np
    return list(np.array(x).shape)

if __name__ == "__main__":
    main()
