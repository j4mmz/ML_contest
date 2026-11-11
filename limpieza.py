"""Limpieza de las series en numpy.

Hace lo mismo que la version compilada de motor.py, pero con numpy. Se usa para entrenar (por lotes)
y como respaldo en model.py si numba no esta disponible.

Cadena:   rellenar huecos  ->  quitar picos  ->  centrar cada canal  ->  dividir por la amplitud global

Todas las funciones aceptan una muestra (4, 256) o un lote (n, 4, 256).
"""
import numpy as np

UMBRAL_PICOS = 7.0     # un punto es un pico si se aleja de la mediana movil mas de 7 escalas del canal
RECORTE = 3.0          # la escala del canal se calcula sin los puntos a mas de 3 desviaciones medias


def rellenar_huecos(X):
    """Interpola linealmente los huecos (NaN o inf) de cada canal.

    En los bordes se copia el valor valido mas cercano. Un canal sin ningun valor valido queda a 0.
    """
    X = np.asarray(X, dtype=np.float32)
    forma = X.shape
    L = forma[-1]
    A = X.reshape(-1, L).copy()
    A[~np.isfinite(A)] = np.nan
    ok = ~np.isnan(A)
    if ok.all():
        return A.reshape(forma)
    t = np.arange(L)
    # indice del ultimo valor valido a la izquierda y del primero a la derecha de cada punto
    izq = np.maximum.accumulate(np.where(ok, t, -1), axis=1)
    der = np.minimum.accumulate(np.where(ok, t, L)[:, ::-1], axis=1)[:, ::-1]
    hay_izq = izq >= 0
    hay_der = der < L
    A0 = np.where(ok, A, np.float32(0.0))
    v_izq = np.take_along_axis(A0, np.clip(izq, 0, L - 1), axis=1)
    v_der = np.take_along_axis(A0, np.clip(der, 0, L - 1), axis=1)
    dentro = hay_izq & hay_der
    peso = np.where(dentro, (t - izq) / np.maximum(der - izq, 1), 0.0).astype(np.float32)
    relleno = np.where(dentro, v_izq + peso * (v_der - v_izq), np.where(hay_izq, v_izq, v_der))
    sal = np.where(ok, A0, relleno).astype(np.float32)
    sal[~ok.any(axis=1)] = 0.0
    return sal.reshape(forma)


def mediana_movil5(X):
    """Mediana movil de 5 puntos (bordes repetidos) con una red de minimos y maximos."""
    L = X.shape[-1]
    Xp = np.concatenate([X[..., :1], X[..., :1], X, X[..., -1:], X[..., -1:]], axis=-1)
    a, b, d, e = Xp[..., 0:L], Xp[..., 1:L + 1], Xp[..., 3:L + 3], Xp[..., 4:L + 4]
    g = np.maximum(np.minimum(a, b), np.minimum(X, d))
    h = np.minimum(np.maximum(a, b), np.maximum(X, d))
    return np.maximum(np.minimum(e, g), np.minimum(np.maximum(e, g), h))


def escala_canal(X):
    """Escala robusta de cada canal: desviacion absoluta media, sin contar los puntos muy alejados.

    Un pico grande dispara la desviacion tipica. Aqui se calcula primero la desviacion media con todos
    los puntos y despues se repite solo con los que quedan a menos de RECORTE veces esa desviacion.
    Solo usa sumas, asi que es mucho mas barata que ordenar el canal para sacar cuartiles.
    """
    m0 = X.mean(-1, keepdims=True)
    d0 = np.abs(X - m0)
    dentro = (d0 <= RECORTE * d0.mean(-1, keepdims=True)).astype(np.float32)
    n = dentro.sum(-1, keepdims=True)
    m1 = (X * dentro).sum(-1, keepdims=True) / n
    return (np.abs(X - m1) * dentro).sum(-1, keepdims=True) / n


def quitar_picos(X, umbral=UMBRAL_PICOS):
    """Filtro de Hampel: los puntos que se alejan de la mediana movil mas de `umbral` escalas del canal
    se sustituyen por esa mediana."""
    m5 = mediana_movil5(X)
    return np.where(np.abs(X - m5) > umbral * escala_canal(X) + 1e-6, m5, X)


def limpiar(X):
    """Cadena completa. Devuelve float32 con cada canal centrado y la muestra con RMS global 1."""
    X = quitar_picos(rellenar_huecos(X))
    X = X - X.mean(-1, keepdims=True)
    rms = np.sqrt((X * X).mean(axis=(-2, -1), keepdims=True)) + 1e-6
    return (X / rms).astype(np.float32)


def limpiar_lotes(X, lote=8192):
    """`limpiar` por trozos, para no llenar la memoria con el train entero."""
    return np.concatenate([limpiar(X[i:i + lote]) for i in range(0, len(X), lote)])
