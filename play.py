import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path
import time
import torch
import numpy as np

import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from env_wrappers import SkipFrame, GrayScaleObservation, ResizeObservation, CustomFrameStack
from mario import Mario

def play(checkpoint_path, episodes=5, render=True):
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"No se encontro el archivo de checkpoint: {checkpoint_path}")

    # En Gym 0.26 con compatibilidad se debe especificar render_mode='human'
    render_mode = "human" if render else "rgb_array"
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode=render_mode, apply_api_compatibility=True)
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = CustomFrameStack(env, num_stack=4)

    mario = Mario(
        state_dim=(4, 84, 84),
        action_dim=env.action_space.n,
        save_dir=Path("eval"),
        checkpoint_path=checkpoint_path
    )
    # En evaluacion forzamos explotacion pura (sin aleatoriedad)
    mario.exploration_rate = 0.0

    print("=" * 60)
    print(f" EVALUANDO MARIO CON CHECKPOINT: {checkpoint_path}")
    print(" Modo: Explotacion pura (epsilon = 0)")
    print("=" * 60)

    for ep in range(episodes):
        state, info = env.reset()
        total_reward = 0
        step_count = 0

        while True:
            if render:
                env.render()
                time.sleep(0.02)  # limitar FPS para visualizacion humana

            action = mario.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            step_count += 1
            state = next_state

            if done or info.get("flag_get", False):
                break

        print(f"Episodio {ep + 1}/{episodes} | Recompensa: {total_reward:.1f} | Pasos: {step_count} | Meta alcanzada: {info.get('flag_get', False)}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar Mario con modelo entrenado")
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta al archivo .chkpt")
    parser.add_argument("--episodes", type=int, default=3, help="Numero de episodios a ejecutar")
    parser.add_argument("--no-render", action="store_true", help="Desactivar renderizado grafico")
    args = parser.parse_args()

    play(checkpoint_path=args.checkpoint, episodes=args.episodes, render=not args.no_render)
