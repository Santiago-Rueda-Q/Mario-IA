# 🍄 Guía de Inicio Rápido: Entrenamiento de Mario con IA (DDQN)

Esta guía te muestra paso a paso cómo poner a Mario a aprender desde cero en tu computadora con tu tarjeta gráfica **NVIDIA GeForce RTX 3050 (6GB)**.

---

## ⚡ Paso 1: Activar el entorno según tu terminal

### 👉 Si estás en CMD (Símbolo del sistema - pantalla negra clásica):
Ejecuta:
```cmd
.\venv\Scripts\activate.bat
```
*(Verás que el prompt cambia a `(venv) c:\React\Mario-IA>`)*

> 💡 **Forma directa (sin activar nada, funciona siempre en CMD y PowerShell):**
> Puedes usar directamente el python del entorno:
> ```cmd
> .\venv\Scripts\python.exe test_env.py
> .\venv\Scripts\python.exe train.py --episodes 40000
> ```

---

### 👉 Si estás en PowerShell (pantalla azul o Windows Terminal):
Ejecuta:
```powershell
.\venv\Scripts\Activate.ps1
```

---

## 🧪 Paso 2: Verificación rápida (10 segundos)

Antes de iniciar un entrenamiento largo, comprueba que el juego y la GPU respondan:
```powershell
python test_env.py
```
> Si todo está bien, verás que detecta tu **RTX 3050**, arranca la ROM de Mario 1-1, ejecuta 50 pasos de prueba y muestra `TODO EL STACK ESTA OPERATIVO Y LISTO PARA ENTRENAR`.

---

## 🚀 Paso 3: Poner a Mario a Aprender

Tienes dos opciones dependiendo del tiempo que tengas:

### Opción A: Prueba rápida (50 a 100 episodios)
Para ver cómo funciona el bucle y comprobar que se guardan los archivos:
```powershell
python train.py --episodes 100
```

### Opción B: Entrenamiento Completo / Nocturno (Recomendado)
Para que Mario explore el mundo, descubra cómo superar los obstáculos y llegue a la bandera:
```powershell
python train.py --episodes 40000
```

---

## 📈 ¿Qué ocurre mientras entrena?

En la consola verás una línea como esta cada 20 episodios:
```text
Episodio 20 | Paso 4077 | Epsilon 0.9989 | Reward 684.71 | Longitud 194.1 | Loss 0.0125 | Q 0.4512
```
- **Episodio**: Número de partida actual.
- **Paso**: Cantidad total de acciones realizadas en el juego.
- **Epsilon**: Nivel de exploración (empieza en 1.0 = 100% aleatorio, y va bajando lentamente a 0.1 = 90% decisiones inteligentes del modelo).
- **Reward**: Puntuación que Mario acumuló avanzando a la derecha.
- **Loss / Q**: Pérdida del entrenamiento y calidad estimada de las acciones.

### Gráficos automáticos en tiempo real
Se creará una carpeta en `checkpoints/` (con la fecha y hora de inicio). Dentro encontrarás:
- 📊 `reward_plot.jpg`: Curva de recompensa (irá subiendo a medida que aprende).
- 📉 `loss_plot.jpg`: Pérdida de la red neuronal.
- ⏱️ `length_plot.jpg`: Cuántos segundos/pasos sobrevive Mario en cada episodio.
- 💾 `mario_net_*.chkpt`: Los checkpoints guardados automáticamente cada 50,000 pasos.

---

## ⏸️ Reanudar el entrenamiento si lo detienes

Si necesitas cerrar la computadora o pausar con `Ctrl + C`, puedes continuar desde el último checkpoint guardado:
```powershell
python train.py --episodes 40000 --checkpoint checkpoints/<nombre_carpeta>/mario_net_<numero>.chkpt
```

---

## 🎮 Paso 4: Ver a Mario jugando en pantalla

Cuando quieras ver cómo juega tu modelo sin acciones aleatorias (modo experto $\epsilon = 0$):

```powershell
python play.py --checkpoint checkpoints/<nombre_carpeta>/mario_net_<numero>.chkpt --episodes 3
```
*Se abrirá la ventana del emulador y verás a la IA controlando a Mario.*

Si solo quieres evaluar su puntuación sin abrir la ventana:
```powershell
python play.py --checkpoint checkpoints/<nombre_carpeta>/mario_net_<numero>.chkpt --episodes 5 --no-render
```
