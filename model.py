"""Modelo para el 1.er Machine Learning Contest de GDG on Campus UAM.

Una CNN 1D pequeña ejecutada en numpy puro (sin PyTorch ni scikit-learn al predecir).

``predict(x)`` hace, para cada muestra:

1. **Limpieza** (la cadena de ``procesado.py``, la misma del entrenamiento): interpola los huecos
   (NaN), quita los picos con un filtro de Hampel y resta la media de cada canal.
2. **Normalización global**: divide la muestra por su amplitud (quita la ganancia de ``group``).
3. **CNN 1D** de convoluciones con ReLU (pesos en ``modelo.npz``, exportados desde PyTorch con la
   BatchNorm plegada en cada convolución) + media y máximo globales + capa lineal.
4. **Sesgo por clase** ajustado con la métrica oficial y ``argmax``.

No usa ``meta["level"]`` (en el train delata la clase y en el test no) y nunca devuelve -1:
con 8 clases, adivinar siempre vale más que pasar.
Si algo falla, devuelve una clase de respaldo en vez de lanzar una excepción.
"""
from __future__ import annotations

import json
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import as_strided

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

import procesado as pp  # noqa: E402

try:    # np.interp sin su envoltorio de Python (unos 4 µs menos por llamada); mismo cálculo
    from numpy._core.multiarray import interp as _interp
    assert float(_interp(np.array([1]), np.array([0, 2]), np.array([0.0, 4.0], np.float32))[0]) == 2.0
except Exception:   # otra versión de numpy: se usa la función pública
    _interp = np.interp

# con ±inf en la entrada, numpy avisa antes de que la ruta rápida lo detecte y pase a la de referencia
warnings.filterwarnings("ignore", category=RuntimeWarning, module=re.escape(__name__))

PESOS = _DIR / "modelo.npz"
L, C0 = pp.L_SERIE, 4
FORMA = (C0, L)
_INF = float("inf")
# ventana de 5 puntos alrededor de cada posición de la serie aplanada (4 canales seguidos), con los
# bordes de cada canal repetidos: fila i = índices de [t-2, t-1, t, t+1, t+2] dentro de su canal
_VENT5 = (np.clip(np.arange(L)[:, None] + np.arange(-2, 3)[None, :], 0, L - 1)[None, :, :]
          + (np.arange(C0) * L)[:, None, None]).reshape(C0 * L, 5)


class Model:
    def __init__(self):
        with np.load(PESOS, allow_pickle=False) as z:
            cfg = json.loads(str(z["cfg"]))
            self.capas_ref = [(np.ascontiguousarray(z[f"W{i}"]), z[f"b{i}"].copy(), int(k), int(st))
                              for i, (k, st) in enumerate(zip(cfg["ks"], cfg["strides"]))]
            self.Wc_ref = np.ascontiguousarray(z["Wc"])
            self.bc_ref = (z["bc"] + z["sesgo"]).astype(np.float32)
            self.respaldo = int(z["respaldo"])
        self._preparar_ruta_rapida()

    # ================================================================== ruta de referencia
    def logits_referencia(self, serie: np.ndarray, sample_rate: float) -> np.ndarray:
        """El cálculo escrito de la forma más directa (y la única ruta para formas distintas de 4×256)."""
        h = pp.normalizar(pp.limpiar(serie)).T                 # (L, C)
        for W, b, k, st in self.capas_ref:
            n, C = h.shape
            p = k // 2
            hp = np.zeros((n + 2 * p, C), np.float32)           # relleno de ceros
            hp[p:p + n] = h
            n_sal = (n + 2 * p - k) // st + 1
            idx = np.arange(n_sal)[:, None] * st + np.arange(k)[None, :]
            h = hp[idx].reshape(n_sal, k * C) @ W + b           # ventanas (im2col) × pesos
            np.maximum(h, 0, out=h)                             # ReLU
        f = np.concatenate([np.add.reduce(h, 0) * (1.0 / h.shape[0]), np.maximum.reduce(h, 0),
                            np.array([(sample_rate - 1.5) * 2], np.float32)])
        return f @ self.Wc_ref + self.bc_ref

    # ================================================================== ruta rápida
    def _preparar_ruta_rapida(self):
        """Reserva una sola vez todos los búferes y las vistas que ``logits`` reutiliza en cada llamada."""
        # --- limpieza
        self.A = np.empty(FORMA, np.float32)                    # la muestra, que se va limpiando en el sitio
        self.a = self.A.reshape(-1)                             # la misma memoria, aplanada (4 canales seguidos)
        self.At = self.A.T
        self.m = np.empty(FORMA, bool)                          # máscara de huecos
        self.mf = self.m.reshape(-1)
        self.okf = np.empty(C0 * L, bool)
        self.S = np.empty(FORMA, np.float32)                    # cada canal ordenado (para los cuartiles)
        self.S_q1, self.S_q3 = self.S[:, L // 4:L // 4 + 1], self.S[:, 3 * L // 4:3 * L // 4 + 1]
        self.umb = np.empty((C0, 1), np.float32)                # umbral de Hampel de cada canal
        self.umbf = self.umb.reshape(-1)
        self.D = np.zeros(FORMA, np.float32)                    # |x[t+1] - x[t]|
        self.Df = self.D.reshape(-1)
        self.d = self.Df[:-1]
        self.D_fin = self.D[:, -1]
        self.a_izq, self.a_der = self.a[:-1], self.a[1:]
        self.cand = np.empty(FORMA, bool)
        self.candf = self.cand.reshape(-1)
        self.mu = np.empty((C0, 1), np.float32)
        # --- red
        # Truco del sesgo: relu(x + b) = max(x, -b) + b. Cada capa guarda y' = max(x, -b) (una sola
        # operación, sin sumar el sesgo) y el "+ b" pendiente se suma, ya multiplicado por los pesos,
        # al sesgo de la capa siguiente. El relleno de los bordes vale -b, que equivale al 0 real.
        geo, n, C = [], L, C0
        for W, b, k, st in self.capas_ref:
            p = k // 2
            n_sal = (n + 2 * p - k) // st + 1
            geo.append((W, b.astype(np.float64), k, st, p, n, C, n_sal))
            n, C = n_sal, W.shape[1]
        bufs = [np.zeros((g[5] + 2 * g[4], g[6]), np.float32) for g in geo]     # entrada de cada capa, con relleno
        self.H = np.empty((n, C), np.float32)                   # salida de la última capa
        self.capas, pendiente = [], np.zeros(C0)
        for i, (W, b, k, st, p, n_ent, C_ent, n_sal) in enumerate(geo):
            b_ef = b + np.tile(pendiente, k) @ W.astype(np.float64)
            # vista de las ventanas sobre el búfer de entrada: fila t = k pasos × C canales seguidos en memoria
            vista = as_strided(bufs[i], shape=(n_sal, k * C_ent), strides=(st * C_ent * 4, 4), writeable=False)
            if i + 1 < len(geo):
                bufs[i + 1][:] = (-b_ef).astype(np.float32)
                p_sig = geo[i + 1][4]
                salida = bufs[i + 1][p_sig:p_sig + n_sal]       # se escribe directamente en la entrada de la capa siguiente
            else:
                salida = self.H
            suelo = np.ascontiguousarray(np.broadcast_to((-b_ef).astype(np.float32), (n_sal, W.shape[1])))
            self.capas.append((vista, np.empty((n_sal, k * C_ent), np.float32), W, suelo, salida))
            pendiente = b_ef
        self.entrada = bufs[0][geo[0][4]:geo[0][4] + L]         # (L, 4), contigua
        self._bufs = bufs                                       # las vistas dependen de estos búferes
        # --- cabeza: [media, máximo, sample_rate, 1] @ Wc (la media y el sesgo van dentro de la matriz)
        self.f = np.zeros(2 * C + 2, np.float32)
        self.f[-1] = 1.0
        self.f_media, self.f_max = self.f[:C], self.f[C:2 * C]
        Wc = self.Wc_ref.astype(np.float64)
        bc = self.bc_ref.astype(np.float64) + pendiente @ Wc[:C] + pendiente @ Wc[C:2 * C]
        self.Wc = np.ascontiguousarray(np.vstack([Wc[:C] / n, Wc[C:], bc[None, :]]).astype(np.float32))

    def _limpiar(self, X: np.ndarray) -> bool:
        """Deja en ``self.entrada`` la muestra limpia y normalizada: lo mismo que
        ``pp.normalizar(pp.limpiar(X)).T``. Devuelve False si hay que usar la ruta de referencia (±inf)."""
        A, a, mf = self.A, self.a, self.mf
        np.copyto(A, X)
        # 1) huecos: una sola interpolación lineal sobre los 4 canales seguidos
        np.not_equal(A, A, out=self.m)                          # NaN es lo único distinto de sí mismo
        i_mal = mf.nonzero()[0]
        if i_mal.size:
            if mf[0] or mf[255] or mf[256] or mf[511] or mf[512] or mf[767] or mf[768] or mf[1023]:
                m = self.m
                for c in range(C0):                             # hueco en un borde: copiar el valor válido más cercano
                    if m[c, 0] or m[c, -1]:
                        ok = (~m[c]).nonzero()[0]
                        if ok.size == 0:                        # canal vacío
                            A[c] = 0.0
                            m[c] = False
                            continue
                        A[c, :ok[0]] = A[c, ok[0]]
                        A[c, ok[-1] + 1:] = A[c, ok[-1]]
                        m[c, :ok[0]] = False
                        m[c, ok[-1] + 1:] = False
                i_mal = mf.nonzero()[0]
            if i_mal.size:
                np.logical_not(mf, out=self.okf)
                i_ok = self.okf.nonzero()[0]
                a[i_mal] = _interp(i_mal, i_ok, a[i_ok])
        # 2) picos (Hampel): umbral = 5 × IQR/1.349 de cada canal, con UNA ordenación
        S, umb = self.S, self.umb
        np.copyto(S, A)
        S.sort(axis=-1)
        np.subtract(self.S_q3, self.S_q1, out=umb)
        np.multiply(umb, 1.0 / 1.349, out=umb)
        np.add(umb, 1e-6, out=umb)
        np.multiply(umb, 5.0, out=umb)
        #    La mediana de 5 solo hace falta donde puede haber un pico: si |x[t] − mediana5| > umbral,
        #    alguno de sus dos vecinos dista de x[t] más que el umbral. Se buscan esos saltos.
        np.subtract(self.a_der, self.a_izq, out=self.d)
        np.abs(self.D, out=self.D)
        self.D_fin[:] = 0.0                                     # el salto entre un canal y el siguiente no cuenta
        np.greater(self.D, umb, out=self.cand)
        i_c = self.candf.nonzero()[0]
        if i_c.size:
            pos = np.concatenate([i_c, i_c + 1])                # los dos extremos de cada salto
            x = a[pos]
            if not np.isfinite(x).all():
                return False                                    # hay ±inf: ruta de referencia
            med = np.sort(a[_VENT5[pos]], axis=1)[:, 2]         # mediana de 5 (bordes repetidos)
            pico = np.abs(x - med) > self.umbf[pos >> 8]
            if pico.any():
                a[pos[pico]] = med[pico]
        # 3) centrar cada canal y 4) dividir por la amplitud global
        mu = self.mu
        np.add.reduce(A, axis=-1, keepdims=True, out=mu)
        np.multiply(mu, 1.0 / L, out=mu)
        np.subtract(A, mu, out=A)
        s = float(np.dot(a, a))
        if not s < _INF:                                        # desbordamiento, inf o NaN
            return False
        np.multiply(A, 1.0 / (math.sqrt(s / a.size) + 1e-6), out=A)
        np.copyto(self.entrada, self.At)                        # (4, L) -> (L, 4)
        return True

    def logits(self, serie: np.ndarray, sample_rate: float) -> np.ndarray:
        if type(serie) is not np.ndarray or serie.shape != FORMA or not self._limpiar(serie):
            return self.logits_referencia(serie, sample_rate)
        for vista, cols, W, suelo, salida in self.capas:
            np.copyto(cols, vista)                              # ventanas -> matriz contigua (así numpy usa BLAS)
            np.matmul(cols, W, out=salida)                      # la convolución entera es un producto de matrices
            np.maximum(salida, suelo, out=salida)               # ReLU con el sesgo dentro
        f = self.f
        np.add.reduce(self.H, 0, out=self.f_media)
        np.maximum.reduce(self.H, 0, out=self.f_max)
        f[-2] = (sample_rate - 1.5) * 2
        return f @ self.Wc

    def predict(self, x) -> int:
        try:
            meta = x.meta
            sr = float(meta.get("sample_rate", 1.0)) if isinstance(meta, dict) else 1.0
        except Exception:
            sr = 1.0
        try:
            return int(self.logits(x.series, sr).argmax())
        except Exception:
            try:
                return int(self.logits_referencia(x.series, sr).argmax())
            except Exception:
                return self.respaldo
