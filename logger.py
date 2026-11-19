import time
import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend no interactivo para guardar imagenes en background
import matplotlib.pyplot as plt

class MetricLogger:
    """
    Registra metricas por paso y episodio, escribe logs en texto y grafica
    las curvas de aprendizaje (recompensa, longitud de episodio, perdida y valores Q).
    """
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_log = self.save_dir / "log.txt"

        with open(self.save_log, "w", encoding="utf-8") as f:
            f.write(
                f"{'Episode':>8}{'Step':>10}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>22}\n"
            )

        self.ep_rewards_plot = self.save_dir / "reward_plot.jpg"
        self.ep_lengths_plot = self.save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = self.save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot = self.save_dir / "q_plot.jpg"

        # Historial de metricas por episodio
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []

        # Medias moviles ultimos 100 episodios
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # Metricas del episodio actual
        self.init_episode()

        # Tiempo
        self.record_time = time.time()

    def log_step(self, reward, loss, q):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss is not None:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        """Marca el final del episodio actual y guarda sus estadisticas"""
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)

        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0.0
            ep_avg_q = 0.0
        else:
            ep_avg_loss = float(np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5))
            ep_avg_q = float(np.round(self.curr_ep_q / self.curr_ep_loss_length, 5))

        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = float(np.round(np.mean(self.ep_rewards[-100:]), 3))
        mean_ep_length = float(np.round(np.mean(self.ep_lengths[-100:]), 3))
        mean_ep_loss = float(np.round(np.mean(self.ep_avg_losses[-100:]), 3))
        mean_ep_q = float(np.round(np.mean(self.ep_avg_qs[-100:]), 3))

        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = float(np.round(self.record_time - last_record_time, 3))
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(
            f"Episodio {episode:6d} | "
            f"Paso {step:8d} | "
            f"Epsilon {epsilon:6.4f} | "
            f"Reward {mean_ep_reward:8.2f} | "
            f"Longitud {mean_ep_length:6.1f} | "
            f"Loss {mean_ep_loss:7.4f} | "
            f"Q {mean_ep_q:7.4f} | "
            f"Tiempo {time_since_last_record:6.2f}s | "
            f"{now_str}"
        )

        with open(self.save_log, "a", encoding="utf-8") as f:
            f.write(
                f"{episode:8d}{step:10d}{epsilon:10.4f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{now_str:>22}\n"
            )

        # Generar graficos actualizados
        metrics_dict = {
            "ep_rewards": ("Recompensa media (ultimos 100)", self.moving_avg_ep_rewards, self.ep_rewards_plot, "green"),
            "ep_lengths": ("Longitud media de episodio", self.moving_avg_ep_lengths, self.ep_lengths_plot, "blue"),
            "ep_avg_losses": ("Loss media (Smooth L1)", self.moving_avg_ep_avg_losses, self.ep_avg_losses_plot, "red"),
            "ep_avg_qs": ("Valor Q medio", self.moving_avg_ep_avg_qs, self.ep_avg_qs_plot, "purple"),
        }

        for _, (title, data, path, color) in metrics_dict.items():
            if len(data) > 0:
                plt.figure(figsize=(8, 4))
                plt.plot(data, color=color, linewidth=1.5)
                plt.title(title)
                plt.xlabel("Registros")
                plt.ylabel("Valor")
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                plt.savefig(path, dpi=120)
                plt.close()
