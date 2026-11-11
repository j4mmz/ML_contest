"""Version compilada (numba) de todo lo que hace `predict`: limpieza de la serie + red convolucional.

Por que existe: con numpy, cada operacion sobre una muestra tan pequena (4 x 256) paga unos microsegundos
de sobrecarga de Python, y entre la limpieza y la red hay decenas de operaciones. Aqui todo son bucles
compilados a codigo maquina, sin reservar memoria en cada llamada, y `predict` tarda mucho menos.

Toda la memoria se prepara una sola vez en model.py:

    P   vector con todos los pesos (solo se lee)
    T   espacio de trabajo: los buferes de entrada de cada capa, ya con su relleno de ceros

Las convoluciones recorren la serie en orden (tiempo, canal): asi la ventana que ve cada filtro es un
trozo seguido de memoria y el producto con los pesos se vectoriza bien.

Las funciones `crear_...` devuelven funciones compiladas a medida: los tamanos de cada capa y las
posiciones de memoria quedan como constantes dentro del codigo compilado. Con los tamanos fijos el
compilador desenrolla y vectoriza los bucles; pasandolos como argumentos sale un 50 % mas lento.
"""
import numpy as np
from numba import njit

L = 256                 # pasos de tiempo
C = 4                   # canales
CLASES = 8

# Permite al compilador reordenar las sumas (necesario para vectorizarlas) y usar FMA. No se le dice que
# "no hay NaN" (nnan/ninf), porque la limpieza tiene que detectarlos.
RAPIDO = {"reassoc", "contract", "arcp", "nsz"}

# zonas fijas al principio del espacio de trabajo T
T_A = 0                 # canal con los huecos ya rellenos (256)
T_M = 264               # su mediana movil (256)
T_Y = 528               # los 4 canales limpios y centrados, uno detras de otro (1024)
T_LIBRE = 1560          # a partir de aqui, los buferes de las capas


@njit(fastmath=RAPIDO, error_model="numpy")
def limpiar(x, T, umbral):
    """Deja en T[T_Y:] los 4 canales con los huecos interpolados, sin picos y con media 0.

    Devuelve 1 / RMS global, el factor por el que hay que multiplicar para normalizar la muestra.
    Es la misma cadena que limpieza.py.
    """
    cero = np.float32(0.0)
    uno = np.float32(1.0)
    suma_cuadrados = cero
    for c in range(C):
        # --- 1) huecos (NaN o inf). v - v vale 0 solo si v es un numero normal.
        malos = 0
        for t in range(L):
            v = x[c, t]
            T[T_A + t] = v
            if not (v - v == cero):
                malos += 1
        if malos > 0:
            ant = -1                     # posicion del ultimo valor valido
            v_ant = cero
            for t in range(L):
                v = x[c, t]
                if v - v == cero:
                    if ant < t - 1:
                        if ant < 0:      # hueco al principio: se copia el primer valor valido
                            for j in range(t):
                                T[T_A + j] = v
                        else:            # hueco en medio: recta entre los dos vecinos validos
                            dif = v - v_ant
                            inv = uno / np.float32(t - ant)
                            for j in range(ant + 1, t):
                                T[T_A + j] = v_ant + (np.float32(j - ant) * inv) * dif
                    ant = t
                    v_ant = v
            if ant < 0:                  # canal sin ningun valor valido
                for t in range(L):
                    T[T_A + t] = cero
            else:                        # hueco al final: se copia el ultimo valor valido
                for t in range(ant + 1, L):
                    T[T_A + t] = v_ant

        # --- 2) escala robusta del canal: desviacion absoluta media, sin los puntos muy alejados
        s = cero
        for t in range(L):
            s += T[T_A + t]
        m0 = s / np.float32(L)
        s = cero
        for t in range(L):
            d = T[T_A + t] - m0
            s += max(d, -d)
        limite = np.float32(3.0) * s / np.float32(L)
        s = cero
        n = cero
        for t in range(L):
            v = T[T_A + t]
            d = v - m0
            k = uno if max(d, -d) <= limite else cero
            s += v * k
            n += k
        m1 = s / n
        s = cero
        for t in range(L):
            v = T[T_A + t]
            d = v - m0
            e = v - m1
            k = uno if max(d, -d) <= limite else cero
            s += max(e, -e) * k
        umbral_canal = umbral * (s / n) + np.float32(1e-6)

        # --- 3) mediana movil de 5 puntos con una red de minimos y maximos (bordes repetidos)
        for b in range(4):               # los 2 primeros y los 2 ultimos puntos
            t = b if b < 2 else L - 4 + b
            p = T[T_A + max(t - 2, 0)]
            q = T[T_A + max(t - 1, 0)]
            r = T[T_A + t]
            u = T[T_A + min(t + 1, L - 1)]
            w = T[T_A + min(t + 2, L - 1)]
            g = max(min(p, q), min(r, u))
            h = min(max(p, q), max(r, u))
            T[T_M + t] = max(min(w, g), min(max(w, g), h))
        for t in range(2, L - 2):
            p = T[T_A + t - 2]
            q = T[T_A + t - 1]
            r = T[T_A + t]
            u = T[T_A + t + 1]
            w = T[T_A + t + 2]
            g = max(min(p, q), min(r, u))
            h = min(max(p, q), max(r, u))
            T[T_M + t] = max(min(w, g), min(max(w, g), h))

        # --- 4) filtro de Hampel (el pico se cambia por la mediana) y centrado del canal
        s = cero
        for t in range(L):
            v = T[T_A + t]
            m = T[T_M + t]
            d = v - m
            if max(d, -d) > umbral_canal:
                v = m
            T[T_Y + c * L + t] = v
            s += v
        media = s / np.float32(L)
        for t in range(L):
            v = T[T_Y + c * L + t] - media
            T[T_Y + c * L + t] = v
            suma_cuadrados += v * v
    return uno / (np.sqrt(suma_cuadrados / np.float32(C * L)) + np.float32(1e-6))


def crear_capa(ventana, paso, filtros, posiciones, pesos, sesgo, ent, sal):
    """Compila una capa convolucional con ReLU para estos tamanos.

    Para cada posicion t la entrada es el trozo T[ent + t*paso : ent + t*paso + ventana] y cada filtro es
    una fila de `ventana` pesos (P[pesos + f*ventana : ...]). Se calculan 8 filtros a la vez: asi cada
    dato de la entrada se lee una sola vez para los 8 y el bucle interno se convierte en instrucciones
    vectoriales. `filtros` tiene que ser multiplo de 8.
    """
    @njit(fastmath=RAPIDO, error_model="numpy")
    def capa(T, P):
        cero = np.float32(0.0)
        for t in range(posiciones):
            base = ent + t * paso
            for f in range(0, filtros, 8):
                w = pesos + f * ventana
                a0 = cero
                a1 = cero
                a2 = cero
                a3 = cero
                a4 = cero
                a5 = cero
                a6 = cero
                a7 = cero
                for i in range(ventana):
                    v = T[base + i]
                    a0 += P[w + i] * v
                    a1 += P[w + ventana + i] * v
                    a2 += P[w + 2 * ventana + i] * v
                    a3 += P[w + 3 * ventana + i] * v
                    a4 += P[w + 4 * ventana + i] * v
                    a5 += P[w + 5 * ventana + i] * v
                    a6 += P[w + 6 * ventana + i] * v
                    a7 += P[w + 7 * ventana + i] * v
                o = sal + t * filtros + f
                T[o] = max(a0 + P[sesgo + f], cero)
                T[o + 1] = max(a1 + P[sesgo + f + 1], cero)
                T[o + 2] = max(a2 + P[sesgo + f + 2], cero)
                T[o + 3] = max(a3 + P[sesgo + f + 3], cero)
                T[o + 4] = max(a4 + P[sesgo + f + 4], cero)
                T[o + 5] = max(a5 + P[sesgo + f + 5], cero)
                T[o + 6] = max(a6 + P[sesgo + f + 6], cero)
                T[o + 7] = max(a7 + P[sesgo + f + 7], cero)
    return capa


def encadenar(primera, segunda):
    """Funcion compilada que ejecuta `primera` y despues `segunda`."""
    @njit(fastmath=RAPIDO, error_model="numpy")
    def las_dos(T, P):
        primera(T, P)
        segunda(T, P)
    return las_dos


def crear_predecir(red, entrada, ultima, posiciones, filtros, rasgos, n_rasgos, pesos_cab, sesgo_cab, salida, umbral):
    """Compila `puntuaciones` y `predecir` para una red ya encadenada.

    entrada      donde se escribe la serie limpia para la primera capa, en orden (tiempo, canal)
    ultima       donde deja su salida la ultima capa: `posiciones` x `filtros`
    rasgos       donde se guarda el resumen global (media y maximo de cada filtro, y sample_rate)
    pesos_cab    capa lineal final: 8 filas de `n_rasgos` pesos; sesgo_cab: sus 8 sesgos
    salida       donde quedan las 8 puntuaciones
    """
    @njit(fastmath=RAPIDO, error_model="numpy")
    def puntuaciones(x, sr, P, T):
        escala = limpiar(x, T, umbral)
        for t in range(L):
            for c in range(C):
                T[entrada + t * C + c] = T[T_Y + c * L + t] * escala
        red(T, P)
        # resumen global: media y maximo de cada filtro a lo largo del tiempo
        for f in range(filtros):
            T[rasgos + f] = np.float32(0.0)
            T[rasgos + filtros + f] = np.float32(0.0)
        for t in range(posiciones):
            for f in range(filtros):
                v = T[ultima + t * filtros + f]
                T[rasgos + f] += v
                T[rasgos + filtros + f] = max(T[rasgos + filtros + f], v)
        for f in range(filtros):
            T[rasgos + f] *= np.float32(1.0 / posiciones)
        T[rasgos + 2 * filtros] = np.float32((sr - 1.5) * 2.0)
        # capa lineal final
        for k in range(CLASES):
            s = P[sesgo_cab + k]
            for i in range(n_rasgos):
                s += T[rasgos + i] * P[pesos_cab + k * n_rasgos + i]
            T[salida + k] = s

    @njit(fastmath=RAPIDO, error_model="numpy")
    def predecir(x, sr, P, T):
        """Clase con mayor puntuacion (0..7), o -2 si la muestra no tiene la forma esperada."""
        if x.shape[0] != C or x.shape[1] != L:
            return -2
        puntuaciones(x, sr, P, T)
        mejor = 0
        for k in range(1, CLASES):
            if T[salida + k] > T[salida + mejor]:
                mejor = k
        return mejor

    return puntuaciones, predecir
