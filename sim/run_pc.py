"""
run_pc.py
=========
Entrena el modelo de referencia en la PC y guarda todo lo necesario para
compararlo despues contra el microcontrolador.

Genera:
  resultados_pc.json        -> corrida principal (float64)
  resultados_pc_f32.json    -> misma corrida en float32 (efecto de precision)
  resultados_escala.json    -> comparativa ASCII_SCALE = 1.0 vs 30.0
  pesos_iniciales.h         -> pesos de arranque para pegar en el firmware

Uso:
    python run_pc.py
"""

import time
import numpy as np

import mlp_core as core
from mlp_core import MLP, split_dataset, epoch_orders, save_json, gradient_check

# ---------------- configuracion del experimento ----------------
EPOCHS = 30          # 3 epocas dan solo 3 puntos: insuficiente para graficar
LR = 0.3
SEED_WEIGHTS = 1
SEED_DATA = 0
SEED_ORDER = 7
NOISE_STD = 0.5
REPS = 15


def train(net, Xtr, Ytr, Xva, Yva, orders, lr):
    """Entrena registrando metricas por epoca y la prediccion de cada muestra.

    El registro por muestra (pred_log) es lo que permite comparar PC contra
    MCU paso a paso, no solo al final de cada epoca.
    """
    hist = {"loss_train": [], "loss_val": [], "acc_train": [], "acc_val": [],
            "pred_log": [], "loss_log": [],
            "grad_l1": [], "grad_l2": [], "grad_l3": [], "z1_absmax": []}

    t0 = time.perf_counter()
    for ep, order in enumerate(orders):
        acc_loss = 0.0
        g1 = g2 = g3 = 0.0
        z1max = 0.0
        for idx in order:
            # se hace forward/backward explicito para poder observar los delta
            a3, cache = net.forward([Xtr[idx]])
            loss, (d1, d2, d3) = net.backward(cache, [Ytr[idx]], lr, update=True)
            acc_loss += loss
            g1 += float(np.mean(np.abs(d1)))
            g2 += float(np.mean(np.abs(d2)))
            g3 += float(np.mean(np.abs(d3)))
            z1max = max(z1max, float(np.max(np.abs(cache[1]))))
            hist["pred_log"].append(float(a3[0]))
            hist["loss_log"].append(loss)
        n = len(order)
        hist["loss_train"].append(acc_loss / n)
        # Magnitud media del error local por capa: si la capa 1 esta saturada,
        # grad_l1 colapsa varios ordenes de magnitud frente a grad_l3.
        hist["grad_l1"].append(g1 / n)
        hist["grad_l2"].append(g2 / n)
        hist["grad_l3"].append(g3 / n)
        # |z1| maximo: si supera ~177, expf(-c*z) desborda en float32 y
        # NeuronDer() del firmware devuelve NaN.
        hist["z1_absmax"].append(z1max)

        lv, av = net.evaluate(Xva, Yva)
        lt, at = net.evaluate(Xtr, Ytr)
        hist["loss_val"].append(lv)
        hist["acc_val"].append(av)
        hist["acc_train"].append(at)
        print(f"  epoca {ep:3d}  loss_tr={hist['loss_train'][-1]:.6f}  "
              f"loss_val={lv:.6f}  acc_val={av:5.1f}%")
    hist["wall_time_s"] = time.perf_counter() - t0
    hist["time_per_sample_us"] = 1e6 * hist["wall_time_s"] / (EPOCHS * len(Xtr))
    return hist


def run(dtype, tag, scale=None):
    if scale is not None:
        core.ASCII_SCALE = scale

    Xtr, Ytr, Xva, Yva = split_dataset(REPS, NOISE_STD, SEED_DATA)
    orders = epoch_orders(len(Xtr), EPOCHS, SEED_ORDER)
    net = MLP(seed=SEED_WEIGHTS, dtype=dtype)
    w0 = net.get_weights()

    print(f"\n[{tag}]  dtype={np.dtype(dtype).name}  "
          f"ASCII_SCALE={core.ASCII_SCALE}  "
          f"{len(Xtr)} train / {len(Xva)} val")
    hist = train(net, Xtr, Ytr, Xva, Yva, orders, LR)

    # Curva de salida de la red sobre todo el rango de entrada:
    # con una sola entrada, esto ES la frontera de decision aprendida.
    xs = np.linspace(-1.2, 1.2, 400)
    hist["curve_x"] = xs.tolist()
    hist["curve_y"] = [net.predict(x) for x in xs]

    hist.update({
        "config": {"epochs": EPOCHS, "lr": LR, "ascii_scale": core.ASCII_SCALE,
                   "ascii_center": core.ASCII_CENTER, "dtype": np.dtype(dtype).name,
                   "seed_weights": SEED_WEIGHTS, "seed_data": SEED_DATA,
                   "seed_order": SEED_ORDER, "noise_std": NOISE_STD,
                   "n_train": len(Xtr), "n_val": len(Xva),
                   "topology": [core.N_IN, core.N_L1, core.N_L2, core.N_L3]},
        "weights_init": w0,
        "weights_final": net.get_weights(),
        "orders": orders,
        "X_train": Xtr.tolist(), "y_train": Ytr.tolist(),
        "X_val": Xva.tolist(), "y_val": Yva.tolist(),
    })
    return hist


def main():
    print("Gradient check (analitico vs diferencias centradas)")
    print(f"  error relativo maximo = {gradient_check():.3e}\n")

    # --- corrida principal en doble precision (referencia) ---
    hist64 = run(np.float64, "PC float64", scale=30.0)
    save_json("resultados_pc.json", hist64)

    # --- misma corrida en precision simple: aisla el efecto de float32 ---
    hist32 = run(np.float32, "PC float32", scale=30.0)
    save_json("resultados_pc_f32.json", hist32)

    # --- demostracion del problema de normalizacion ---
    # Con ASCII_SCALE = 1.0 la entrada llega en [-28.5, 28.5]; la capa 1
    # satura y su gradiente se anula. Esta comparativa es la evidencia
    # para la seccion "problemas encontrados" del informe.
    bad = run(np.float64, "PC sin normalizar (scale=1.0)", scale=1.0)
    core.ASCII_SCALE = 30.0
    save_json("resultados_escala.json",
              {"scale_1": {k: bad[k] for k in
                           ("loss_train", "loss_val", "acc_val", "weights_init",
                            "weights_final", "curve_x", "curve_y",
                            "grad_l1", "grad_l2", "grad_l3", "z1_absmax")},
               "scale_30": {k: hist64[k] for k in
                            ("loss_train", "loss_val", "acc_val", "weights_init",
                             "weights_final", "curve_x", "curve_y",
                             "grad_l1", "grad_l2", "grad_l3", "z1_absmax")}})

    # --- pesos iniciales para el firmware ---
    with open("pesos_iniciales.h", "w", encoding="utf-8") as f:
        f.write(MLP(seed=SEED_WEIGHTS).export_c())
    print("\nArchivos generados: resultados_pc.json, resultados_pc_f32.json,")
    print("resultados_escala.json, pesos_iniciales.h")


if __name__ == "__main__":
    main()