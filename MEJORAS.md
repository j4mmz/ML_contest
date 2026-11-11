# Mejoras sobre el modelo del equipo

Parto del modelo de `prueba_ml` (CNN 1D en numpy, 88 puntos en local) y de todo su análisis: el simulador del test, el des-intercambio de canales, val-sim y las decisiones D1 a D17 siguen valiendo y no las repito aquí. Sigo la numeración del diario.

La idea de fondo: la puntuación va con la raíz del número de muestras contestadas y el modelo anterior gastaba casi todo su tiempo en la sobrecarga de numpy, no en la red. Atacando eso sale mucho más que afinando el acierto.

## Resultado

Mismo fichero val-sim (240 000 muestras, harness oficial sin tocar) y misma máquina para los dos:

| | Modelo anterior | Modelo nuevo |
|---|---|---|
| `predict` | numpy | compilado con numba |
| Latencia media por muestra | 76.6 µs | **4.9 µs** |
| Acierto medio por clase en val-sim | 0.857 | 0.849 |
| Acierto en el demo (400) | 0.848 | 0.855 |
| Parte de la puntuación que depende del acierto (G) | 0.571 | 0.551 |
| Excepciones / respuestas inválidas | 0 | 0 |
| Puntuación estimada si Colab va 3.9 veces más lento que esta máquina | ~91 | **~345** |

El 3.9 sale de comparar la latencia del modelo anterior aquí (76.6 µs) con la que midió el equipo en su equipo (309 µs). En Colab habrá que medirlo. Se pierde algo menos de un punto de acierto y se gana un factor 15 en velocidad.

Ojo con una cosa: el harness corta el run a los 30 minutos reales o a los 10 millones de muestras. Si en Colab el generador de muestras no da abasto, el número de muestras lo limita eso y no nuestra velocidad. Con un tope de 1.5 millones de muestras por run la estimación baja a ~239 (el modelo anterior seguiría en ~91).

## D18. ¿Dónde se iba el tiempo de `predict`?

- **Evidencia:** la red anterior tiene 582 000 multiplicaciones por muestra, que a la velocidad de un núcleo son unos 10 µs. Pero `predict` tardaba 76.6 µs: el resto eran las decenas de llamadas a numpy sobre arrays pequeños (cada una cuesta 1 a 5 µs aunque el array tenga 1000 números).
- **Decisión:** escribir todo `predict` (limpieza + red) como bucles compilados con numba, en `motor.py`, con la memoria reservada una sola vez. Las bases dejan usar cualquier librería del `requirements.txt` y dicen que en el warm-up se puede compilar.
- **Comprobado:** con la red anterior sin cambiar, 76.6 µs pasan a 11.5 µs y las respuestas son las mismas (diferencia máxima de 5e-5 en las puntuaciones frente a PyTorch, misma clase en el 100 % de 3 400 muestras).
- **Detalles que importan:**
  - Los tamaños de cada capa quedan como constantes dentro del código compilado (`crear_capa`). Pasándolos como argumentos salía un 50 % más lento, porque el compilador no desenrolla los bucles.
  - Se calculan 8 filtros a la vez y la serie va en orden (tiempo, canal), para que la ventana de cada filtro sea memoria seguida.
  - No se usa el `fastmath` completo, solo permitir reordenar sumas: con el completo el compilador puede asumir que no hay NaN y la limpieza dejaría de verlos.
- **Descartadas:** onnxruntime o PyTorch en `predict` (la llamada sola ya cuesta decenas de µs); compilar C con gcc en el arranque (sería más rápido, pero depende de que haya compilador en la máquina de evaluación).

## D19. La escala robusta del filtro de Hampel

- **Problema:** el rango intercuartílico obliga a ordenar cada canal. En código compilado eso costaba más que toda la limpieza junta.
- **Decisión:** desviación absoluta media recortada. Se calcula la desviación media del canal, se descartan los puntos a más de 3 veces esa desviación y se repite. Solo usa sumas. El umbral pasa de 5 a 7 para que equivalga al anterior (para una señal gaussiana la desviación media es 0.8 veces la típica).
- **Evidencia:** misma red, 30 épocas, 2 semillas: 0.8585 con la escala nueva frente a 0.8577 con el IQR. La limpieza entera cuesta 0.9 µs.
- `limpieza.py` es la versión numpy (para entrenar y como respaldo) y da lo mismo que la compilada (diferencia máxima 2e-6).

## D20. ¿Qué tamaño de red?

Con numpy no compensaba achicar la red porque el coste fijo mandaba. Con el motor compilado el coste fijo son unos 2 µs, así que el tamaño vuelve a importar. Para comparar uso G / raíz(latencia), que es la puntuación salvo una constante.

val-sim, 60 épocas, media de 2 semillas. Latencia de `predict` completo en esta máquina:

| Red (filtros por capa) | Operaciones | Latencia | val-sim | G | G / raíz(latencia) |
|---|---|---|---|---|---|
| 8-8-16-16 (paso 4 al principio) | 51 k | 2.4 µs | 0.791 | 0.410 | 0.265 |
| 8-16-16-24 (paso 4 al principio) | 78 k | 3.1 µs | 0.818 | 0.474 | 0.269 |
| 8-8-16-16-24 | 109 k | 3.8 µs | 0.836 | 0.513 | 0.262 |
| 8-8-16-24-32 | 137 k | 4.6 µs | 0.842 | 0.531 | 0.247 |
| **8-8-16-24-32-48 (elegida)** | 168 k | **4.6 µs** | 0.849 | 0.549 | **0.257** |
| 8-16-24-32-48 | 266 k | 6.1 µs | 0.855 | 0.565 | 0.228 |
| 8-16-16-24-32-48 | 315 k | 7.0 µs | 0.863 | 0.589 | 0.222 |
| 16-32-32-64 (la anterior) | 582 k | 11.5 µs | 0.860 | 0.580 | 0.171 |
| 16-32-32-64-64 | 745 k | 14.4 µs | 0.869 | 0.604 | 0.159 |

- **Lectura:** entre 50 k y 170 k operaciones la puntuación estimada es casi plana (0.26 a 0.27). Las redes profundas y estrechas rinden mucho más por operación que la anterior: con 6 capas y la mitad de anchura se iguala su acierto con menos de la mitad de coste.
- **Decisión:** de la zona plana cojo la más grande, 8-8-16-24-32-48. Si hay tope de muestras en Colab, lo único que cuenta es el acierto y conviene no haber apurado el tamaño. Las dos últimas capas trabajan sobre 8 posiciones y casi no cuestan.
- **Entrenamiento:** el mismo del equipo (simulador nuevo en cada época, AdamW, one-cycle, pesos de clase equilibrados). 120 épocas en vez de 30: +0.5 puntos; con 200 ya no mejora.

## D21. Cosas probadas que no han funcionado

| Idea | Resultado en val-sim |
|---|---|
| Primera capa propia para `sample_rate` = 2 (ventana y paso dobles) | −2.9 puntos |
| Una red distinta para cada `sample_rate` | −0.1 a −0.7 |
| Destilación desde 3 redes grandes (0.915 en val-sim) | −0.1 a −0.7 |
| Destilación con maestros cruzados (cada muestra la etiqueta un maestro que no la vio) | −0.5 |
| Dar a la capa final la amplitud de la muestra y el grupo | ±0.3, dentro del ruido entre semillas |

Lo de la amplitud tiene además trampa: la desviación de log(amplitud / ganancia) es 0.39 en el demo y 0.26 en el train simulado. El rango de amplitudes del test es más ancho, así que se deja fuera, como ya decía D8.

Los retardos entre canales no escalan con `sample_rate` (ch2 va 5 pasos por detrás de ch0 tanto con 1 como con 2), y por eso tratar las muestras de `sample_rate` = 2 a media resolución pierde información.

## D22. Las clases 6 y 7

Siguen siendo el punto débil (0.70 y 0.65). Lo que he podido medir:

- Una red enorme (ResNet de 8.7 millones de parámetros, inservible por velocidad) llega a 0.83 y 0.82. El techo con estos datos está ahí.
- La información para separarlas está solo en ch0 y ch1, y cada uno la lleva por separado (68 % con un solo canal, 77 % con los dos, 50 % con ch2 y ch3).
- No está en el espectro: aleatorizando las fases de la serie (mismo espectro) se cae al 50 %.
- Es asimétrica: si a la red grande se le da la serie con el tiempo invertido, con el signo cambiado o con trozos de 16 pasos barajados, las muestras de la clase 6 pasan a clasificarse como 7 (del 83 % al 24-37 %), pero las de la 7 se quedan igual. La clase 6 es "la 7 más una estructura": la amplitud de la oscilación crece poco a poco durante varios ciclos y cae de golpe (asimetría de los incrementos de altura de los picos: −0.14 en la clase 6 y 0.00 en la 7).

Hace falta ver muchos ciclos seguidos para notarlo, y por eso las redes pequeñas se quedan cortas. No he encontrado un rasgo barato que lo capture (el mejor por separado da AUC 0.60).

## D23. Robustez

- Si numba no se puede importar o falla al compilar, `model.py` usa la misma red en numpy. Comprobado quitando numba: mismas respuestas en las 400 del demo.
- Después de compilar se llama a `disable_compile()`: si llegara una entrada de otro tipo, numba lanza un error (y se contesta con numpy) en vez de ponerse a compilar en mitad de la evaluación, que sería un timeout.
- Entradas rotas (todo NaN, inf, 1e30, otra forma, `meta` vacía o con texto, listas, float64): siempre devuelve un entero entre 0 y 7.
- Probado con numpy 2.5 + numba 0.68 (Python 3.12) y con numpy 1.26 + numba 0.59 (Python 3.11). Por eso `requirements.txt` no fija versiones exactas: así no se desinstala el numpy que ya tenga Colab.
- `Model()` tarda unos 3 s (compilar). El límite es 120 s.
- Copia limpia de `model.py`, `motor.py`, `limpieza.py`, `pesos.npz` y `requirements.txt` con el harness del repositorio oficial: mismo resultado.

## Ficheros

| Fichero | Para qué |
|---|---|
| `model.py` | `class Model`. Prepara la memoria, compila y contesta. |
| `motor.py` | Limpieza y convoluciones compiladas con numba. |
| `limpieza.py` | La misma limpieza en numpy (entrenamiento y respaldo). |
| `pesos.npz` | Pesos de la red (54 KB, sin pickle). |
| `entrenar.py` | Entrena la red y escribe `pesos.npz` (unos 9 minutos). |
| `requirements.txt` | `numpy` y `numba`. |

## Cómo probarlo

```bash
pip install -r requirements.txt
python harness.py evaluate --model model.py --source file:data/test_demo.npz   # acierto por clase y latencia
python entrenar.py                                                             # volver a entrenar (necesita torch y scikit-learn)
```

Los pesos entregados son los de `python entrenar.py --semilla 2` (de tres semillas, la mejor en val-sim empatada con la 0 y mejor en el demo).

## Pendiente de medir en Colab

1. La latencia real. Aquí son 4.9 µs; con una máquina 4 veces más lenta serían unos 20 µs y 3 millones de muestras por run.
2. Si se llega al tope de 30 minutos del harness. Si se llega, conviene cambiar a una red más grande de la tabla de D20 (solo hay que cambiar `CAPAS` en `entrenar.py` y volver a entrenar): con el número de muestras fijo gana la que más acierta.
