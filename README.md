# 🍄 Mario RL Agent (DDQN) con PyTorch y CUDA

Agente autónomo de Aprendizaje por Refuerzo (Reinforcement Learning) para jugar **Super Mario Bros (NES)** utilizando **Double Deep Q-Networks (DDQN)** con aceleración GPU (NVIDIA CUDA 12.4).

---

## 💻 Especificaciones del Entorno

- **Python**: 3.11.8
- **PyTorch**: 2.6.0+cu124 (CUDA 12.4)
- **Aceleración GPU**: NVIDIA GeForce RTX 3050 6GB Laptop GPU
- **Emulador**: `nes-py` + `gym-super-mario-bros` (Super Mario Bros 1-1)
- **Compilador nativo**: Microsoft Visual Studio 2022 C++ Build Tools (instalado)

---

## 🚀 Inicio Rápido

### 1. Activar el entorno virtual

En PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Verificar el entorno y la GPU

Ejecuta el script de prueba para validar que el emulador, la ROM y PyTorch con CUDA funcionan:
```powershell
python test_env.py
```

---

## 🏋️‍♂️ Entrenamiento del Agente

### Entrenamiento de prueba (ejemplo: 50 episodios):
```powershell
python train.py --episodes 50
```

### Entrenamiento completo / nocturno (40,000 episodios):
```powershell
python train.py --episodes 40000
```

### Reanudar desde un checkpoint:
```powershell
python train.py --episodes 40000 --checkpoint checkpoints/2026-11-18T18-00-00/mario_net_1.chkpt
```

---

## 🎮 Evaluación y Visualización

Para ver jugar a Mario con un modelo guardado en pantalla:
```powershell
python play.py --checkpoint checkpoints/<directorio>/mario_net_<paso>.chkpt --episodes 3
```

Para evaluar en segundo plano sin ventana gráfica:
```powershell
python play.py --checkpoint checkpoints/<directorio>/mario_net_<paso>.chkpt --episodes 5 --no-render
```

---

## 📊 Métricas y Gráficos Generados

Dentro de la carpeta de checkpoints asignada (por ejemplo `checkpoints/<fecha-hora>/`) se generan en tiempo real:
- `log.txt`: Registro detallado de recompensa, longitud de episodio, epsilon, pérdida y valor Q.
- `reward_plot.jpg`: Gráfico de media móvil de recompensas obtenidas.
- `loss_plot.jpg`: Curva de convergencia de pérdida de la red neuronal.
- `length_plot.jpg`: Duración en pasos de cada intento de Mario.
- `q_plot.jpg`: Estimación del valor Q a lo largo del tiempo.
- `mario_net_*.chkpt`: Pesos del modelo y estado del optimizador.

---

## 🧱 Estructura del Código

- [mario.py](file:///c:/React/Mario-IA/mario.py): Agente DDQN (`act`, `cache`, `recall`, `learn`, `save`, `load`).
- [mario_net.py](file:///c:/React/Mario-IA/mario_net.py): Red Convolucional (`Q_online` y `Q_target`).
- [env_wrappers.py](file:///c:/React/Mario-IA/env_wrappers.py): Pipeline de preprocesamiento (`SkipFrame`, `GrayScale`, `Resize`, `CustomFrameStack`).
- [replay_buffer.py](file:///c:/React/Mario-IA/replay_buffer.py): Buffer de repetición de experiencias optimizado.
- [logger.py](file:///c:/React/Mario-IA/logger.py): Sistema de logging y generación de gráficos.
- [train.py](file:///c:/React/Mario-IA/train.py): Bucle principal de entrenamiento.
- [play.py](file:///c:/React/Mario-IA/play.py): Script de inferencia y visualización.
- [test_env.py](file:///c:/React/Mario-IA/test_env.py): Script de diagnóstico rápido.
