"""Preprocesado y rasgos comunes a entrenamiento y predicción.

Es el MISMO código en los dos sitios (notebook de entrenamiento y ``model.py``), para que el
modelo vea en ``predict`` exactamente lo que vio al entrenar. Solo depende de numpy.

Cadena decidida en las Fases 1 y 2 del notebook (``final.ipynb``):

    rellenar_nan  ->  quitar_picos  ->  centrar  ->  normalizar   (entrada de la CNN)
                                                 ->  rasgos       (entrada del GBDT)

Todas las funciones aceptan una muestra ``(C, L)`` o un lote ``(n, C, L)``.
"""
from __future__ import annotations

import numpy as np

L_SERIE = 256
_T = np.arange(L_SERIE, dtype=np.float32)


# ----------------------------------------------------------------------------- limpieza

def rellenar_nan(X: np.ndarray) -> np.ndarray:
    """Interpola linealmente los huecos (NaN o ±inf) de cada canal.

    Los huecos de los bordes copian el valor válido más cercano; un canal sin ningún valor válido
    queda a 0. Devuelve un array float32 nuevo (la entrada de ``predict`` es de solo lectura).
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:                        # vía rápida para una sola muestra (la de predict)
        A = X.copy()
        malos = ~np.isfinite(A)
        if not malos.any():
            return A
        if malos[:, 0].any() or malos[:, -1].any():          # bordes: copiar el valor válido más cercano
            for c in np.flatnonzero(malos[:, 0] | malos[:, -1]):
                ok = np.flatnonzero(~malos[c])
                if ok.size == 0:
                    A[c] = 0.0
                    malos[c] = False
                    continue
                A[c, :ok[0]] = A[c, ok[0]]
                A[c, ok[-1] + 1:] = A[c, ok[-1]]
                malos[c, :ok[0]] = False
                malos[c, ok[-1] + 1:] = False
        # con los extremos de cada canal ya válidos, ningún hueco cruza de un canal a otro:
        # basta UNA interpolación sobre la serie aplanada (4 canales seguidos)
        a, m = A.ravel(), malos.ravel()
        i_mal = np.flatnonzero(m)
        if i_mal.size:
            i_ok = np.flatnonzero(~m)
            a[i_mal] = np.interp(i_mal, i_ok, a[i_ok])
        return A
    forma, L = X.shape, X.shape[-1]        # lote: versión vectorizada (mismo resultado)
    A = X.reshape(-1, L).copy()
    A[~np.isfinite(A)] = np.nan
    ok = ~np.isnan(A)
    if ok.all():
        return A.reshape(forma)
    t = np.arange(L)
    prev = np.maximum.accumulate(np.where(ok, t, -1), axis=1)                 # último índice válido <= t
    nxt = np.minimum.accumulate(np.where(ok, t, L)[:, ::-1], axis=1)[:, ::-1]  # primer índice válido >= t
    hay_prev, hay_next = prev >= 0, nxt < L
    vp = np.take_along_axis(A, np.clip(prev, 0, L - 1), axis=1)
    vn = np.take_along_axis(A, np.clip(nxt, 0, L - 1), axis=1)
    ambos = hay_prev & hay_next
    w = np.where(ambos, (t - prev) / np.maximum(nxt - prev, 1), 0.0)
    relleno = np.where(ambos, vp + w * (vn - vp), np.where(hay_prev, vp, vn))
    out = np.where(ok, A, relleno)
    out[~ok.any(axis=1)] = 0.0
    return out.astype(np.float32).reshape(forma)


def _med3(x, y, z):
    return np.maximum(np.minimum(x, y), np.minimum(np.maximum(x, y), z))


def mediana_movil5(X: np.ndarray) -> np.ndarray:
    """Mediana móvil de 5 puntos (bordes repetidos) con una red de comparaciones: 10 operaciones
    min/max sobre el array entero, unas 4 veces más rápida que ``np.median`` sobre ventanas."""
    L = X.shape[-1]
    Xp = np.concatenate([X[..., :1], X[..., :1], X, X[..., -1:], X[..., -1:]], axis=-1)
    a, b, d, e = Xp[..., 0:L], Xp[..., 1:L + 1], Xp[..., 3:L + 3], Xp[..., 4:L + 4]
    return _med3(e, np.maximum(np.minimum(a, b), np.minimum(X, d)), np.minimum(np.maximum(a, b), np.maximum(X, d)))


def escala_robusta(X: np.ndarray) -> np.ndarray:
    """Rango intercuartílico / 1.349 de cada canal (keepdims): estima la desviación típica sin que la
    muevan los picos. Para ruido gaussiano y para una onda seno coincide con 1.4826 · MAD, pero
    necesita una sola ``np.partition`` (la mitad de tiempo)."""
    L = X.shape[-1]
    q1, q3 = L // 4, (3 * L) // 4
    P = np.partition(X, (q1, q3), axis=-1)
    return (P[..., q3:q3 + 1] - P[..., q1:q1 + 1]) * (1.0 / 1.349) + 1e-6


def _mediana(X):
    """Mediana de cada canal (keepdims) con ``np.partition``: igual que ``np.median``, más rápida."""
    L = X.shape[-1]
    k0, k1 = (L - 1) // 2, L // 2
    P = np.partition(X, (k0, k1), axis=-1)
    return (P[..., k0:k0 + 1] + P[..., k1:k1 + 1]) * 0.5


def _media(X):
    """Media de cada canal (keepdims); ``np.add.reduce`` evita la sobrecarga de ``ndarray.mean``."""
    return np.add.reduce(X, axis=-1, keepdims=True) * (1.0 / X.shape[-1])


def quitar_picos(X: np.ndarray, umbral: float = 5.0) -> np.ndarray:
    """Filtro de Hampel: sustituye por la mediana móvil de 5 puntos los valores que se alejan de
    ella más de ``umbral`` veces la escala robusta DEL CANAL.

    La escala es la del canal entero, no la de la ventana: con 5 puntos la MAD local casi vale 0 y
    todo parecería un pico (probado en la Fase 1: marcaba el 4 % de los puntos del train).
    """
    X = np.asarray(X, dtype=np.float32)
    m5 = mediana_movil5(X)
    return np.where(np.abs(X - m5) > umbral * escala_robusta(X), m5, X)


def centrar(X: np.ndarray) -> np.ndarray:
    """Resta la media de cada canal (quita el nivel/DC, que en train delata la clase y en test no)."""
    return X - _media(X)


def limpiar(X: np.ndarray, picos: bool = True, umbral: float = 5.0) -> np.ndarray:
    """Cadena completa: rellenar NaN -> (quitar picos) -> centrar."""
    X = rellenar_nan(X)
    if picos:
        X = quitar_picos(X, umbral)
    return centrar(X).astype(np.float32, copy=False)


def limpiar_lotes(X: np.ndarray, lote: int = 8192, **kw) -> np.ndarray:
    """``limpiar`` por trozos, para no disparar la memoria con 100 000 muestras."""
    return np.concatenate([limpiar(X[i:i + lote], **kw) for i in range(0, len(X), lote)])


def normalizar(X: np.ndarray) -> np.ndarray:
    """Divide la muestra entera por su amplitud global (RMS de los 4 canales juntos).

    Quita la ganancia de ``group`` (×1/×2/×4) pero conserva las amplitudes RELATIVAS entre canales,
    que sí son información de clase.
    """
    if X.ndim == 2:
        v = X.ravel()
        return (X * np.float32(1.0 / (np.sqrt(np.dot(v, v) / v.size) + 1e-6))).astype(np.float32, copy=False)
    s = np.sqrt((X * X).mean(axis=(-2, -1), keepdims=True)) + 1e-6
    return (X / s).astype(np.float32)


# ----------------------------------------------------------------------------- rasgos

VENTANA = np.hanning(L_SERIE).astype(np.float32)
_H = np.zeros(L_SERIE, dtype=np.float32)            # filtro de Hilbert (señal analítica)
_H[0] = 1.0
_H[1:L_SERIE // 2] = 2.0
_H[L_SERIE // 2] = 1.0
_K = np.arange(L_SERIE // 2 + 1, dtype=np.float32)
_PARES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
_PA = np.array([a for a, _ in _PARES])
_PB = np.array([b for _, b in _PARES])
_LAGS = np.arange(-10, 11)
_ACF = (1, 2, 4, 8)


def nombres_rasgos() -> list[str]:
    n = [f"lsd_rel_{c}" for c in range(4)]
    n += [f"acf{l}_{c}" for l in _ACF for c in range(4)]
    for nom in ("asimdif", "rugosidad", "asim", "curt", "envCV", "logfpico", "logfcent", "arm2", "arm3", "anchura"):
        n += [f"{nom}_{c}" for c in range(4)]
    for nom in ("xc0", "xc+4", "xc-4", "xcmax", "xclag", "phcos", "phsin", "amp_pico"):
        n += [f"{nom}_{a}{b}" for a, b in _PARES]
    return n + ["sample_rate"]


def rasgos(X: np.ndarray, sr) -> np.ndarray:
    """Rasgos de "mecanismo" (Fase 1, §1.7) de series YA LIMPIAS, invariantes a ganancia y nivel.

    Por canal: amplitud relativa, autocorrelación, asimetría temporal (diente de sierra), rugosidad,
    forma de la distribución, envolvente (Hilbert), frecuencia física del pico y del centroide,
    armónicos relativos al pico y anchura espectral.
    Por pares de canales: correlación cruzada con retardo (quién adelanta a quién), fase y amplitud
    relativas en la frecuencia del pico de ch0.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[None]
    n = X.shape[0]
    sr = np.asarray(sr, dtype=np.float32).reshape(-1)
    if sr.size == 1 and n > 1:
        sr = np.full(n, sr[0], dtype=np.float32)
    sd = X.std(-1) + 1e-6                                    # (n, 4)
    Z = X / sd[..., None]
    lsd = np.log(sd)
    f = [lsd - lsd.mean(-1, keepdims=True)]
    f += [(Z[..., l:] * Z[..., :-l]).mean(-1) for l in _ACF]
    d = np.diff(X, axis=-1)
    dsd = d.std(-1) + 1e-6
    dz = (d - d.mean(-1, keepdims=True)) / dsd[..., None]
    f += [(dz ** 3).mean(-1), np.log(dsd / sd), (Z ** 3).mean(-1), (Z ** 4).mean(-1)]
    env = np.abs(np.fft.ifft(np.fft.fft(X, axis=-1) * _H, axis=-1))[..., 16:-16]
    f.append(env.std(-1) / (env.mean(-1) + 1e-6))
    F = np.fft.rfft(X * VENTANA, axis=-1)                    # (n, 4, 129)
    P = F.real ** 2 + F.imag ** 2
    Pt = P[..., 1:].sum(-1) + 1e-12
    pico = P[..., 3:].argmax(-1) + 3                         # (n, 4), sin tendencia lenta
    cent = (P[..., 1:] * _K[1:]).sum(-1) / Pt
    anch = np.sqrt((P[..., 1:] * (_K[1:] - cent[..., None]) ** 2).sum(-1) / Pt) / (cent + 1e-6)
    srb = sr[:, None]
    Pb = P.copy()                                            # potencia en 3 bins centrados (robusta al bin exacto)
    Pb[..., 1:-1] += P[..., :-2] + P[..., 2:]

    def banda(m):
        idx = np.minimum(m * pico, Pb.shape[-1] - 1)
        return np.take_along_axis(Pb, idx[..., None], -1)[..., 0]

    p1 = banda(1) + 1e-12
    f += [np.log(pico * srb), np.log(cent * srb), np.log(banda(2) / p1 + 1e-6), np.log(banda(3) / p1 + 1e-6), anch]
    G = np.fft.rfft(Z, n=2 * L_SERIE, axis=-1)               # correlación cruzada sin vuelta circular
    R = np.fft.irfft(G[:, _PA] * np.conj(G[:, _PB]), n=2 * L_SERIE, axis=-1) / L_SERIE
    R = R[..., _LAGS]                                        # retardos -10..10: R[lag] = mean z_a(t+lag) z_b(t)
    f += [R[..., 10], R[..., 14], R[..., 6], R.max(-1), (R.argmax(-1) - 10).astype(np.float32)]
    k0 = pico[:, 0][:, None, None]
    Fk = np.take_along_axis(F, np.broadcast_to(k0, (n, 4, 1)), -1)[..., 0]      # (n, 4) complejo
    cruz = Fk[:, _PB] * np.conj(Fk[:, _PA])
    mod = np.abs(cruz) + 1e-12
    f += [cruz.real / mod, cruz.imag / mod, np.log(np.abs(Fk[:, _PB]) / (np.abs(Fk[:, _PA]) + 1e-9) + 1e-9)]
    f.append(sr[:, None])
    return np.concatenate(f, axis=-1).astype(np.float32)
