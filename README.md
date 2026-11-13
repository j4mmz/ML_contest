# GDG UAM ML Contest: clasificador rápido de series temporales

Entrega para el **1.er Machine Learning Contest de GDG on Campus UAM**: clasificar series temporales de 4 canales × 256 pasos en 8 clases, **una muestra por llamada, en 1 hilo de CPU**, puntuando cuántas muestras se aciertan en 60 s de `predict`.

| | |
|---|---|
| **Puntuación oficial local** (harness oficial, 3 runs, val-sim de 750 000 muestras) | **118.7** |
| **Latencia media por muestra** (harness, 1 hilo) | **0.170 ms** (~353 000 muestras en 60 s) |
| Acierto medio por clase: val-sim / demo real (400) | 0.855 / 0.847 |
| Puntuación esperada con el acierto del demo | 114.7 |
| Baseline oficial, medido igual | 0.3 puntos estimados (0.395 de acierto en el demo) |

## Qué hay en el repositorio

| Fichero | Para qué |
|---|---|
| `model.py` | **El modelo** (`class Model` con `predict(x) -> int`). Solo usa numpy. Tiene la ruta rápida y la de referencia. |
| `procesado.py` | Limpieza y rasgos: **el mismo código** en el entrenamiento y en `predict`. |
| `modelo.npz` | Pesos de la CNN (~64 KB, sin pickle). |
| `requirements.txt` | Dependencias del modelo: `numpy==2.2.6`. |
| `final.ipynb` | **Notebook completo** (en español, ejecutable de principio a fin): entender los datos → limpiar → modelo → evaluar. Entrena el modelo entregado. |
| `requirements-notebook.txt` | Dependencias para re-ejecutar el notebook (torch, scikit-learn, pandas…). |
| `aprendizaje.html` | Guía para aprender: el recorrido completo, los porqués y ejercicios con solución. |
| `proyecto.html` | El proyecto y los datos, con gráficos por clase y **cómo probarlo en Google Colab**. |
| `resultados.json` | Los números de la última ejecución del notebook (los usan este README y los HTML). |

## Cómo probarlo

```bash
git clone <este-repo> submission-check
git clone https://github.com/GDG-UAM/ml-contest-materials materiales
pip install -r submission-check/requirements.txt
cd materiales
# los datos: https://github.com/GDG-UAM/ml-contest-materials/releases (train_100k.npz y test_demo.npz en materiales/data/)
python harness.py evaluate --model ../submission-check/model.py --source file:data/test_demo.npz
```

Con `--source file:data/test_demo.npz` el run se acaba en 400 muestras, así que la puntuación que imprime no es comparable con la oficial. Lo útil es el **acierto por clase** y la **latencia**. Sin `--source`, el harness evalúa sobre el train, con un acierto inflado. Los pasos para **Google Colab** están en `proyecto.html`.

## Cómo funciona 

1. **Limpieza** (`procesado.py`): interpolación lineal de los huecos, filtro de **Hampel** (mediana móvil de 5 puntos, umbral 5× la escala robusta del canal) contra los picos, y resta de la media de cada canal.
2. **Normalización global**: se divide la muestra por su amplitud (quita la ganancia de `group` y conserva las amplitudes relativas entre canales).
3. **CNN 1D pequeña** (4 capas convolucionales, 15 600 parámetros) + media y máximo globales + `sample_rate` → 8 clases, con **sesgos por clase** ajustados para maximizar la métrica oficial.
4. **Nunca se abstiene**: con 8 clases la probabilidad máxima es ≥ 12.5 %, por encima del umbral a partir del cual compensa pasar (≈ 10–12 %).

**Las claves del resultado:**

* **Trampas del train detectadas y evitadas.** `level` y el nivel medio delatan las clases 3 y 5 solo en el train, así que no se usan. La mitad de las clases 3 y 6 llevan **canales intercambiados** solo en el train, así que se deshacen antes de entrenar.
* **El test es más sucio.** Tiene el doble de huecos, picos en el 21 % de las muestras y el doble de ruido. Un **simulador calibrado contra el demo** lleva esos defectos al train. La CNN ve en **cada época** un train simulado nuevo.
* **Validación que imita el test (val-sim).** Predice el acierto del demo a menos de 2 puntos; la validación sin simular se pasaba en 14 puntos.
* **La métrica premia la velocidad.** La red se ejecuta en numpy con un producto de matrices por capa, sin PyTorch al predecir. `predict` usa una **ruta rápida** que hace las mismas cuentas que el código directo en 2.2 veces menos tiempo (0.33 → 0.15 ms): 0 predicciones distintas en 20 400 muestras.

## Reproducir el entrenamiento

```bash
pip install -r requirements-notebook.txt
jupyter nbconvert --to notebook --execute --inplace final.ipynb   # ~30 min con 6 núcleos (Ryzen 5 2600)
```

El notebook descarga los datos y el harness oficial si no los encuentra en `data/` o en la carpeta superior. Todo usa semillas fijas.
