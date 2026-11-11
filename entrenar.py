"""Entrena la red que usa model.py y guarda sus pesos en pesos.npz.

    python entrenar.py                  # unos minutos con GPU; tambien funciona en CPU, mas lento

Necesita torch y scikit-learn ademas de numpy (model.py no los necesita), y los datos del concurso en
data/ o en ../data/ (train_100k.npz y test_demo.npz).

Pasos:
1. Cargar el train y el demo.
2. Deshacer los canales intercambiados de las clases 3 y 6, que solo existen en el train.
3. Separar un 20 % para validar, estratificado.
4. En cada epoca, llevar el train a las condiciones del test con el simulador (ruido, picos y huecos
   nuevos cada vez), limpiarlo con limpieza.py y entrenar la red.
5. Ajustar un sesgo por clase con la formula de la puntuacion.
6. Guardar los pesos con la BatchNorm plegada en cada convolucion.

El simulador y la regla de los intercambios son los del notebook del equipo (final.ipynb, Fases 1 y 2).
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

import limpieza

AQUI = Path(__file__).resolve().parent
CLASES = 8

# Red: (filtros, tamano del filtro, paso) de cada capa. Los filtros van de 8 en 8 por como esta
# escrita la convolucion compilada de motor.py.
CAPAS = [(8, 7, 2), (8, 5, 2), (16, 5, 2), (24, 5, 2), (32, 5, 2), (48, 3, 1)]

# Simulador del test, calibrado contra el demo en el notebook del equipo (Fase 2)
RUIDO_M = (4.0, 4.0, 2.0, 6.0)          # cuanto puede multiplicarse el suelo de ruido de cada canal
P_PICO, LAMBDA_PICOS = 0.25, 2.5        # una de cada 4 muestras lleva 1 + Poisson(2.5) picos
BORDES_PICO = np.array([3, 4, 6, 10, 16, 32, 100, 316, 1000], dtype=np.float64)        # tamano del pico, en escalas del canal
PESOS_PICO = np.array([0.3493, 0.2298, 0.1744, 0.0903, 0.0875, 0.0406, 0.025, 0.0031])  # reparto entre esos tramos


# ----------------------------------------------------------------------------- datos

def buscar_datos():
    for base in (AQUI / "data", AQUI.parent / "data"):
        if (base / "train_100k.npz").exists() and (base / "test_demo.npz").exists():
            return base
    raise FileNotFoundError("faltan data/train_100k.npz y data/test_demo.npz (ver el README de los materiales)")


def cargar(ruta):
    z = np.load(ruta, allow_pickle=False)
    return dict(X=z["series"], y=z["labels"].astype(np.int64), g=z["meta__group"].astype(str),
                sr=z["meta__sample_rate"].astype(np.float32))


def deshacer_intercambios(X_limpio, X_crudo, y):
    """En el train, la mitad de la clase 3 lleva ch2 y ch3 cambiados y la mitad de la 6, ch0 y ch2.

    En el test no pasa, asi que se devuelven a su sitio. Usa la etiqueta: solo vale para entrenar.
    """
    Z = X_limpio / (X_limpio.std(-1, keepdims=True) + 1e-9)
    acf4 = (Z[..., 4:] * Z[..., :-4]).mean(-1)                 # autocorrelacion a 4 pasos de cada canal
    cambiar3 = (y == 3) & (acf4[:, 2] > acf4[:, 3])
    cambiar6 = (y == 6) & ((Z[:, 1] * Z[:, 2]).mean(-1) > (Z[:, 0] * Z[:, 1]).mean(-1))
    X = X_crudo.copy()
    X[cambiar3] = X[cambiar3][:, [0, 1, 3, 2]]
    X[cambiar6] = X[cambiar6][:, [2, 1, 0, 3]]
    return X, int(cambiar3.sum()), int(cambiar6.sum())


def simular(X, escala, suelo, mascaras, rng):
    """Lleva series crudas del train (n, 4, 256) a las condiciones del test.

    escala y suelo son (n, 4): la escala robusta y el suelo de ruido de cada canal de la serie limpia.
    """
    n = len(X)
    Y = np.array(X, dtype=np.float32, copy=True)
    # ruido blanco: el suelo de cada canal se multiplica por M^u, con el mismo u para los 4 canales
    u = rng.uniform(size=(n, 1))
    m = np.exp(u * np.log(np.asarray(RUIDO_M))[None, :])
    Y += rng.standard_normal(Y.shape, dtype=np.float32) * np.sqrt(suelo * (m - 1)).astype(np.float32)[..., None]
    # picos de un punto, en canal y posicion al azar
    k = np.where(rng.uniform(size=n) < P_PICO, 1 + rng.poisson(LAMBDA_PICOS, size=n), 0)
    i = np.repeat(np.arange(n), k)
    canal = rng.integers(0, 4, len(i))
    t = rng.integers(0, X.shape[-1], len(i))
    tramo = rng.choice(len(PESOS_PICO), size=len(i), p=PESOS_PICO / PESOS_PICO.sum())
    tam = np.exp(rng.uniform(np.log(BORDES_PICO[tramo]), np.log(BORDES_PICO[tramo + 1])))
    Y[i, canal, t] += (rng.choice(np.array([-1.0, 1.0]), size=len(i)) * tam * escala[i, canal]).astype(np.float32)
    # huecos: se copia la mascara de NaN de otra muestra del train
    Y[mascaras[rng.integers(0, len(mascaras), n)]] = np.nan
    return Y


# ----------------------------------------------------------------------------- red

class Red(nn.Module):
    """Convoluciones con BatchNorm y ReLU + media y maximo globales + sample_rate -> 8 clases."""

    def __init__(self, capas):
        super().__init__()
        bloques, entrada = [], 4
        for filtros, k, paso in capas:
            bloques.append(nn.Sequential(nn.Conv1d(entrada, filtros, k, stride=paso, padding=k // 2, bias=False),
                                         nn.BatchNorm1d(filtros), nn.ReLU()))
            entrada = filtros
        self.cuerpo = nn.Sequential(*bloques)
        self.cabeza = nn.Linear(2 * entrada + 1, CLASES)

    def forward(self, x, sr):
        h = self.cuerpo(x)
        return self.cabeza(torch.cat([h.mean(-1), h.amax(-1), (sr[:, None] - 1.5) * 2], 1))


def predecir(red, X, sr, disp, lote=4096):
    """Puntuaciones (n, 8) de un conjunto ya limpio."""
    red.eval()
    sal = []
    with torch.no_grad():
        for i in range(0, len(X), lote):
            x = torch.from_numpy(X[i:i + lote]).to(disp)
            s = torch.from_numpy(sr[i:i + lote]).to(disp)
            sal.append(red(x, s).cpu().numpy())
    return np.concatenate(sal)


def acierto_por_clase(y, p):
    return np.array([(p[y == k] == k).mean() for k in range(CLASES)])


def valor_acierto(a):
    """Parte de la puntuacion oficial que depende del acierto a de una clase: a^2 (a - 0.33 (1 - a))."""
    return a ** 2 * (a - 0.33 * (1 - a))


def ajustar_sesgos(puntos, y, pasos=(0.4, 0.2, 0.1, 0.05), rondas=3):
    """Busca 8 sesgos que, sumados a las puntuaciones, maximizan la media geometrica de la formula oficial."""
    def objetivo(b):
        a = acierto_por_clase(y, (puntos + b).argmax(1))
        return np.log(np.maximum(1e-4, valor_acierto(a))).mean()

    b = np.zeros(CLASES)
    mejor = objetivo(b)
    for paso in pasos:
        for _ in range(rondas):
            for k in range(CLASES):
                for cambio in (paso, -paso):
                    c = b.copy()
                    c[k] += cambio
                    v = objetivo(c)
                    if v > mejor + 1e-9:
                        mejor, b = v, c
    return b - b.mean()


def guardar(red, capas, sesgo, respaldo, ruta):
    """Pliega cada BatchNorm en su convolucion y guarda los pesos como los lee model.py."""
    datos = {}
    for i, bloque in enumerate(red.cuerpo):
        conv, bn = bloque[0], bloque[1]
        g = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
        W = (conv.weight.detach() * g[:, None, None]).cpu().numpy()            # (filtros, canales, k)
        # cada filtro como una fila, en orden (posicion del filtro, canal): asi lo recorre la convolucion
        datos[f"W{i}"] = np.transpose(W, (0, 2, 1)).reshape(W.shape[0], -1).astype(np.float32)
        datos[f"b{i}"] = (bn.bias - bn.running_mean * g).detach().cpu().numpy().astype(np.float32)
    np.savez(ruta, capas=np.array(capas, np.int64), Wc=red.cabeza.weight.detach().cpu().numpy().astype(np.float32),
             bc=red.cabeza.bias.detach().cpu().numpy().astype(np.float32), sesgo=sesgo.astype(np.float32),
             umbral=np.float32(limpieza.UMBRAL_PICOS), respaldo=np.int64(respaldo), **datos)


# ----------------------------------------------------------------------------- principal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=120)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--salida", default=str(AQUI / "pesos.npz"))
    args = ap.parse_args()
    t0 = time.time()
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.semilla)

    # 1) datos
    datos = buscar_datos()
    TR, DE = cargar(datos / "train_100k.npz"), cargar(datos / "test_demo.npz")
    X_tr = np.asarray(TR["X"])
    print(f"train {X_tr.shape} | demo {DE['X'].shape} | {disp}")

    # 2) intercambios y lo que necesita el simulador (escala y suelo de ruido de la serie limpia)
    limpio = limpieza.limpiar_lotes(X_tr)
    crudo, n3, n6 = deshacer_intercambios(limpio, X_tr, TR["y"])
    print(f"canales devueltos a su sitio: {n3} muestras de la clase 3 y {n6} de la 6")
    sin_picos = limpieza.quitar_picos(limpieza.rellenar_huecos(crudo))
    sin_picos = sin_picos - sin_picos.mean(-1, keepdims=True)
    q1, q3 = np.percentile(sin_picos, [25, 75], axis=-1)
    escala = ((q3 - q1) / 1.349 + 1e-6).astype(np.float32)
    ventana = np.hanning(256).astype(np.float32)
    espectro = np.abs(np.fft.rfft(sin_picos * ventana, axis=-1)) ** 2 / (ventana ** 2).sum()
    suelo = espectro[..., 100:].mean(-1).astype(np.float32)        # potencia media en las frecuencias altas
    del limpio, sin_picos, espectro

    # 3) particion 80 / 20 estratificada por clase, grupo y sample_rate
    estrato = TR["y"] * 100 + np.searchsorted(["A", "B", "C"], TR["g"]) * 10 + TR["sr"].astype(int)
    i_tr, i_va = train_test_split(np.arange(len(TR["y"])), test_size=0.2, stratify=estrato, random_state=0)
    i_tr, i_va = np.sort(i_tr), np.sort(i_va)
    y_tr, y_va = TR["y"][i_tr], TR["y"][i_va]
    mascaras = np.isnan(X_tr[np.random.default_rng(5).choice(i_tr, 20000, replace=False)])

    def lote_simulado(idx, semilla):
        rng = np.random.default_rng(semilla)
        return limpieza.limpiar_lotes(simular(crudo[idx], escala[idx], suelo[idx], mascaras, rng))

    val_X = lote_simulado(i_va, 777)                # validacion simulada fija ("val-sim")
    dem_X = limpieza.limpiar(np.asarray(DE["X"]))

    # 4) entrenamiento: en cada epoca un train simulado nuevo, que se prepara en otro hilo mientras se entrena
    red = Red(CAPAS).to(disp)
    pesos_clase = torch.tensor(len(y_tr) / (CLASES * np.bincount(y_tr, minlength=CLASES)), dtype=torch.float32, device=disp)
    criterio = nn.CrossEntropyLoss(weight=pesos_clase, label_smoothing=0.05)
    opt = torch.optim.AdamW(red.parameters(), lr=3e-3, weight_decay=1e-3)
    pasos = args.epocas * int(np.ceil(len(i_tr) / 256))
    plan = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=pasos, pct_start=0.15)
    sr_tr = torch.from_numpy(TR["sr"][i_tr]).to(disp)
    yt = torch.from_numpy(y_tr).to(disp)
    hilo = ThreadPoolExecutor(1)
    futuro = hilo.submit(lote_simulado, i_tr, 1000 * (args.semilla + 1))
    for epoca in range(args.epocas):
        X = torch.from_numpy(futuro.result()).to(disp)
        if epoca + 1 < args.epocas:
            futuro = hilo.submit(lote_simulado, i_tr, 1000 * (args.semilla + 1) + epoca + 1)
        red.train()
        orden = torch.randperm(len(yt), device=disp)
        for i in range(0, len(yt), 256):
            j = orden[i:i + 256]
            perdida = criterio(red(X[j], sr_tr[j]), yt[j])
            opt.zero_grad()
            perdida.backward()
            opt.step()
            plan.step()
        if (epoca + 1) % 10 == 0 or epoca + 1 == args.epocas:
            a = acierto_por_clase(y_va, predecir(red, val_X, TR["sr"][i_va], disp).argmax(1))
            print(f"  epoca {epoca + 1:3d} | val-sim {a.mean():.4f} (peor clase {a.min():.3f}) | {time.time() - t0:.0f} s", flush=True)
    hilo.shutdown()

    # 5) sesgos por clase con la formula oficial, sobre val-sim
    pv = predecir(red, val_X, TR["sr"][i_va], disp)
    pd_ = predecir(red, dem_X, DE["sr"], disp)
    sesgo = ajustar_sesgos(pv, y_va)
    for nombre, b in (("sin sesgos", 0.0), ("con sesgos", sesgo)):
        av = acierto_por_clase(y_va, (pv + b).argmax(1))
        ad = acierto_por_clase(DE["y"], (pd_ + b).argmax(1))
        print(f"{nombre}: val-sim {av.mean():.4f} {np.round(av, 3)} | demo {ad.mean():.4f}")
    print("sesgos:", np.round(sesgo, 3))

    # 6) guardar
    respaldo = int(np.argmax(acierto_por_clase(y_va, (pv + sesgo).argmax(1))))
    guardar(red, CAPAS, sesgo, respaldo, args.salida)
    print(f"guardado {args.salida} | {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
