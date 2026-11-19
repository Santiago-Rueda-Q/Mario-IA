import warnings
warnings.filterwarnings("ignore")

import argparse
import datetime
from pathlib import Path
import torch

import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros

from env_wrappers import SkipFrame, GrayScaleObservation, ResizeObservation, CustomFrameStack
from mario import Mario
from logger import MetricLogger

def make_mario_env():
    """
    Crea y envuelve el entorno de Super Mario Bros 1-1.
    Aplica SkipFrame, Grayscale, Resize y CustomFrameStack para producir estados de forma [4, 84, 84].
    """
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", apply_api_compatibility=True)
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = CustomFrameStack(env, num_stack=4)
    return env

def train(episodes=40000, checkpoint=None, save_dir=None):
    use_cuda = torch.cuda.is_available()
    print("=" * 60)
    print(" INICIANDO ENTRENAMIENTO DE MARIO RL AGENT (DDQN)")
    print(f" Usando CUDA: {use_cuda}")
    if use_cuda:
        print(f" Dispositivo: {torch.cuda.get_device_name(0)}")
    print(f" Episodios programados: {episodes}")
    print("=" * 60)

    if save_dir is None:
        save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    env = make_mario_env()

    # Estado: [4, 84, 84]
    mario = Mario(
        state_dim=(4, 84, 84),
        action_dim=env.action_space.n,
        save_dir=save_dir,
        checkpoint_path=checkpoint
    )
    logger = MetricLogger(save_dir)

    for e in range(episodes):
        state, info = env.reset()

        while True:
            action = mario.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Guardar experiencia en replay memory
            mario.cache(state, next_state, action, reward, done)

            # Paso de aprendizaje DDQN
            q, loss = mario.learn()

            # Registro de paso
            logger.log_step(reward, loss, q)

            state = next_state

            if done or info.get("flag_get", False):
                break

        logger.log_episode()

        # Mostrar y guardar registro cada 20 episodios o en el ultimo
        if (e % 20 == 0) or (e == episodes - 1):
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)

    # Guardar modelo final
    mario.save()
    env.close()
    print(f"\nEntrenamiento finalizado. Registros y modelos en: {save_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar agente Mario con DDQN")
    parser.add_argument("--episodes", type=int, default=40000, help="Numero de episodios a entrenar")
    parser.add_argument("--checkpoint", type=str, default=None, help="Ruta de checkpoint para reanudar")
    parser.add_argument("--save-dir", type=str, default=None, help="Directorio donde guardar checkpoints y graficos")
    args = parser.parse_args()

    train(episodes=args.episodes, checkpoint=args.checkpoint, save_dir=args.save_dir)
