"""Modelo para el 1.er Machine Learning Contest de GDG on Campus UAM.

Una CNN 1D pequena que clasifica cada serie (4 canales x 256 pasos) en 8 clases.

`predict(x)` hace, para cada muestra:

1. Limpieza: interpola los huecos (NaN), quita los picos con un filtro de Hampel, resta la media de
   cada canal y divide la muestra por su amplitud global.
2. CNN 1D: convoluciones con ReLU (la BatchNorm del entrenamiento ya va plegada en los pesos),
   media y maximo de cada filtro a lo largo del tiempo, y una capa lineal con `sample_rate`.
3. Devuelve la clase con mayor puntuacion.

Todo eso esta compilado con numba (motor.py), que es lo que lo hace rapido. Si numba no se puede
importar o falla al compilar, se usa la misma red escrita en numpy (mas lenta, mismas respuestas).

No se usa `meta["level"]` (en el train delata la clase y en el test no) y nunca se devuelve -1:
con 8 clases, contestar siempre vale mas que pasar. Si algo falla, se devuelve una clase de respaldo
en vez de lanzar una excepcion.
"""
import gc
from pathlib import Path

import numpy as np

import limpieza

AQUI = Path(__file__).resolve().parent
PESOS = AQUI / "pesos.npz"

L, C, CLASES = 256, 4, 8


class Model:
    def __init__(self):
        with np.load(PESOS, allow_pickle=False) as z:
            self.capas = [tuple(int(v) for v in fila) for fila in z["capas"]]      # (filtros, k, paso) de cada capa
            self.W = [z[f"W{i}"].astype(np.float32) for i in range(len(self.capas))]   # (filtros, k * canales_entrada)
            self.b = [z[f"b{i}"].astype(np.float32) for i in range(len(self.capas))]
            self.Wc = z["Wc"].astype(np.float32)                                    # (8, 2 * filtros + 1)
            self.bc = (z["bc"] + z["sesgo"]).astype(np.float32)                     # sesgo por clase ya sumado
            self.umbral = np.float32(z["umbral"])
            self.respaldo = int(z["respaldo"])
        self._preparar_numpy()
        # Por defecto, la version numpy. Si numba funciona, se cambia por la compilada.
        self.compilado = False
        self._predecir, self._P, self._T = self._predecir_numpy, None, None
        try:
            self._preparar_numba()
            self.compilado = True
        except Exception:
            pass
        self._calentar()
        # Los objetos creados hasta aqui no se van a borrar: se apartan para que el recolector de basura
        # no los recorra en mitad de un predict.
        gc.collect()
        gc.freeze()

    # ------------------------------------------------------------------ version compilada

    def _preparar_numba(self):
        import motor

        # 1) dimensiones de cada capa
        plan, largo, canales = [], L, C
        for filtros, k, paso in self.capas:
            relleno = k // 2
            posiciones = (largo + 2 * relleno - k) // paso + 1
            ventana = k * canales
            ventana16 = -(-ventana // 16) * 16          # redondeada a multiplo de 16: los pesos de sobra valen 0
            plan.append(dict(filtros=filtros, paso=paso, relleno=relleno, posiciones=posiciones, ventana=ventana,
                             ventana16=ventana16, largo=largo, canales=canales))
            largo, canales = posiciones, filtros

        # 2) espacio de trabajo T: un bufer por capa con su relleno de ceros a los lados (nunca se pisa)
        pos = motor.T_LIBRE
        for d in plan:
            d["bufer"] = pos
            pos += (d["largo"] + 2 * d["relleno"]) * d["canales"] + 32
        salida_red = pos
        pos += plan[-1]["posiciones"] * plan[-1]["filtros"] + 32
        n_rasgos = 2 * plan[-1]["filtros"] + 1
        n_rasgos8 = -(-n_rasgos // 8) * 8
        rasgos = pos
        pos += n_rasgos8 + 8
        salida = pos
        T = np.zeros(pos + 64, np.float32)

        # 3) vector de pesos P y una funcion compilada por capa, encadenadas
        trozos, off, red = [], 0, None
        for i, d in enumerate(plan):
            M = np.zeros((d["filtros"], d["ventana16"]), np.float32)
            M[:, :d["ventana"]] = self.W[i]
            trozos += [M.ravel(), self.b[i]]
            if i + 1 < len(plan):                       # cada capa escribe en el bufer de la siguiente, tras su relleno
                sal = plan[i + 1]["bufer"] + plan[i + 1]["relleno"] * d["filtros"]
            else:
                sal = salida_red
            capa = motor.crear_capa(d["ventana16"], d["paso"] * d["canales"], d["filtros"], d["posiciones"],
                                    off, off + M.size, d["bufer"], sal)
            red = capa if red is None else motor.encadenar(red, capa)
            off += M.size + d["filtros"]
        Mc = np.zeros((CLASES, n_rasgos8), np.float32)
        Mc[:, :n_rasgos] = self.Wc
        trozos += [Mc.ravel(), self.bc]
        P = np.concatenate(trozos).astype(np.float32)
        entrada = plan[0]["bufer"] + plan[0]["relleno"] * C
        puntuaciones, predecir = motor.crear_predecir(red, entrada, salida_red, plan[-1]["posiciones"], plan[-1]["filtros"],
                                                      rasgos, n_rasgos8, off, off + Mc.size, salida, self.umbral)

        # Primera llamada: aqui es donde numba compila. Se compila para un array de solo lectura, que es
        # lo que entrega el harness, y para uno normal.
        x = np.zeros((C, L), np.float32)
        for solo_lectura in (True, False):
            x = x.copy()
            x.setflags(write=not solo_lectura)
            if predecir(x, 1.0, P, T) < 0:
                raise RuntimeError("el motor compilado no responde")
            puntuaciones(x, 1.0, P, T)
        # A partir de aqui no se compila nada mas: con una entrada de otro tipo numba lanza TypeError
        # (y se usa numpy) en vez de ponerse a compilar en mitad de la evaluacion.
        predecir.disable_compile()
        puntuaciones.disable_compile()
        self._predecir, self._P, self._T = predecir, P, T
        self._puntuaciones, self._salida = puntuaciones, salida

    # ------------------------------------------------------------------ version numpy (respaldo)

    def _preparar_numpy(self):
        """Indices de las ventanas de cada capa (im2col): cada convolucion es un indexado y un producto."""
        self._np, largo, canales = [], L, C
        for (filtros, k, paso), W, b in zip(self.capas, self.W, self.b):
            relleno = k // 2
            posiciones = (largo + 2 * relleno - k) // paso + 1
            idx = np.arange(posiciones)[:, None] * paso + np.arange(k)[None, :]
            self._np.append((np.ascontiguousarray(W.T), b, relleno, idx, posiciones, k * canales, largo, canales))
            largo, canales = posiciones, filtros

    def _puntuaciones_numpy(self, serie, sr):
        with np.errstate(all="ignore"):
            h = limpieza.limpiar(serie).T                # (tiempo, canal)
            for Wt, b, relleno, idx, posiciones, ventana, largo, canales in self._np:
                bufer = np.zeros((largo + 2 * relleno, canales), np.float32)
                bufer[relleno:relleno + largo] = h
                h = bufer[idx].reshape(posiciones, ventana) @ Wt + b
                np.maximum(h, 0, out=h)
            rasgos = np.concatenate([h.mean(0), h.max(0), np.array([(sr - 1.5) * 2.0], np.float32)])
            return self.Wc @ rasgos + self.bc

    def _predecir_numpy(self, serie, sr, P=None, T=None):
        return int(np.argmax(self._puntuaciones_numpy(np.asarray(serie, dtype=np.float32), sr)))

    # ------------------------------------------------------------------ predict

    def puntuaciones(self, serie, sr):
        """Las 8 puntuaciones de una muestra (para analisis; predict no pasa por aqui)."""
        if self.compilado:
            x = np.ascontiguousarray(serie, dtype=np.float32)
            self._puntuaciones(x, float(sr), self._P, self._T)
            return self._T[self._salida:self._salida + CLASES].copy()
        return self._puntuaciones_numpy(np.asarray(serie, dtype=np.float32), float(sr))

    def _calentar(self):
        """Unas cuantas llamadas de prueba para que todo este cargado antes de la primera muestra de verdad."""
        rng = np.random.default_rng(0)
        for i in range(200):
            x = rng.standard_normal((C, L)).astype(np.float32)
            if i % 2:
                x[rng.integers(0, C), rng.integers(0, L - 8):][:8] = np.nan
            x.setflags(write=False)
            self._predecir(x, 1.0 + (i % 2), self._P, self._T)

    def predict(self, x) -> int:
        try:
            clase = self._predecir(x.series, float(x.meta["sample_rate"]), self._P, self._T)
            if clase >= 0:
                return clase
        except Exception:
            pass
        return self._predict_lento(x)

    def _predict_lento(self, x):
        """Camino de emergencia: entrada con forma o tipo raros, o sin `sample_rate`."""
        try:
            sr = 1.0
            if isinstance(x.meta, dict):
                try:
                    sr = float(x.meta.get("sample_rate", 1.0))
                except (TypeError, ValueError):
                    sr = 1.0
            return self._predecir_numpy(x.series, sr)
        except Exception:
            return self.respaldo
