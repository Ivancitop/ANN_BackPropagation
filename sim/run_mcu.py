"""
run_mcu.py
==========
Entrena la red EN EL MICROCONTROLADOR usando exactamente el mismo dataset,
el mismo orden de muestras y (si cargas pesos_iniciales.h en el firmware)
los mismos pesos iniciales que run_pc.py. Solo asi la comparacion de curvas
prueba algo sobre la implementacion en C.

Genera resultados_mcu.json con:
  - perdida y prediccion de CADA muestra, tal como las reporta el MCU
  - metricas por epoca
  - tiempo de ida y vuelta por muestra (mide el enlace serie, no el computo)
  - ciclos de CPU por etapa, si el firmware los reporta (ver parches_firmware.c)

Uso:
    python run_mcu.py                  # hardware real
    python run_mcu.py --simular        # emula el MCU en float32, sin hardware
                                       # (util para validar el pipeline y las
                                       #  graficas antes de ir al laboratorio)
"""

import argparse
import sys
import time

import numpy as np

import mlp_core as core
from mlp_core import MLP, split_dataset, epoch_orders, save_json

# ---------------- configuracion del enlace ----------------
PORT = "COM6"              # Linux/Mac: "/dev/ttyUSB0" o "/dev/ttyACM0"
BAUDRATE = 9600            # debe coincidir con Lpuart_Uart_Ip_xHwConfigPB_6
READ_TIMEOUT_S = 2.0
RESET_DELAY_S = 2.0

# ---------------- configuracion del experimento ----------------
# Deben ser IDENTICAS a las de run_pc.py
EPOCHS = 30
LR = 0.3
SEED_WEIGHTS = 1
SEED_DATA = 0
SEED_ORDER = 7
NOISE_STD = 0.5
REPS = 15


# =====================================================================
# Transporte
# =====================================================================
class SerialLink:
    """Enlace real con el MCU. Protocolo de linea: '<modo>,<x>,<y>\\n'."""

    def __init__(self, port, baud, timeout):
        import serial  # se importa aqui para que --simular no lo exija
        self.ser = serial.Serial(port, baud, timeout=timeout)
        if RESET_DELAY_S > 0:
            time.sleep(RESET_DELAY_S)
        self.ser.reset_input_buffer()

    def send(self, mode, x, y):
        line = f"{mode},{x:.6f},{y:.1f}\n"
        t0 = time.perf_counter()
        self.ser.write(line.encode("ascii"))
        resp = self.ser.readline().decode("ascii", errors="replace").strip()
        rtt = time.perf_counter() - t0

        if not resp:
            raise TimeoutError(f"Sin respuesta del MCU para {line!r}")

        parts = resp.split(",")
        if len(parts) < 2:
            raise ValueError(f"Respuesta inesperada del MCU: {resp!r}")

        pred, loss = float(parts[0]), float(parts[1])
        # Campos opcionales: ciclos de CPU medidos con DWT->CYCCNT en el
        # firmware. Formato "pred,loss,cicl_fwd,cicl_bwd\n".
        cycles = None
        if len(parts) >= 4:
            cycles = (int(parts[2]), int(parts[3]))
        elif len(parts) == 3:
            cycles = (int(parts[2]), 0)
        return pred, loss, rtt, cycles

    def close(self):
        self.ser.close()


class SimLink:
    """Emula al MCU en float32. No sustituye la medicion real: sirve para
    depurar el pipeline y comprobar que las graficas salen bien."""

    def __init__(self):
        self.net = MLP(seed=SEED_WEIGHTS, dtype=np.float32)

    def send(self, mode, x, y):
        t0 = time.perf_counter()
        update = mode in ("T", "t")
        a3, cache = self.net.forward([np.float32(x)])
        loss, _ = self.net.backward(cache, [np.float32(y)], LR, update=update)
        # se simula el retardo del enlace: 8N1 -> 10 bits por byte
        n_bytes = len(f"{mode},{x:.6f},{y:.1f}\n") + len(f"{a3[0]:.6f},{loss:.6f}\n")
        rtt = n_bytes * 10.0 / BAUDRATE
        return float(a3[0]), float(loss), max(rtt, time.perf_counter() - t0), None

    def close(self):
        pass


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simular", action="store_true",
                    help="emula el MCU en float32, sin hardware")
    ap.add_argument("--puerto", default=PORT)
    ap.add_argument("--baud", type=int, default=BAUDRATE)
    ap.add_argument("--epocas", type=int, default=EPOCHS)
    args = ap.parse_args()

    Xtr, Ytr, Xva, Yva = split_dataset(REPS, NOISE_STD, SEED_DATA)
    orders = epoch_orders(len(Xtr), args.epocas, SEED_ORDER)

    print(f"Dataset: {len(Xtr) + len(Xva)} muestras "
          f"({len(Xtr)} train / {len(Xva)} val), "
          f"ASCII_SCALE={core.ASCII_SCALE}")

    # Estimacion del costo del enlace antes de empezar: con tramas de texto
    # y 9600 baud, el tiempo de transporte domina por completo al computo.
    bytes_tx = len("T,-0.383000,0.0\n")
    bytes_rx = len("0.123456,0.007890\n")
    t_link = (bytes_tx + bytes_rx) * 10.0 / args.baud
    total_est = t_link * args.epocas * len(Xtr)
    print(f"Enlace: {args.baud} baud -> {1e3 * t_link:.1f} ms por muestra "
          f"(~{total_est / 60:.1f} min de entrenamiento solo en transporte)\n")

    link = SimLink() if args.simular else SerialLink(args.puerto, args.baud,
                                                     READ_TIMEOUT_S)

    hist = {"loss_train": [], "acc_val": [], "loss_val": [],
            "pred_log": [], "loss_log": [], "rtt_log": [], "cycles_log": []}

    try:
        t0 = time.perf_counter()
        for ep, order in enumerate(orders):
            acc_loss = 0.0
            for idx in order:
                pred, loss, rtt, cyc = link.send("T", Xtr[idx], Ytr[idx])
                acc_loss += loss
                hist["pred_log"].append(pred)
                hist["loss_log"].append(loss)
                hist["rtt_log"].append(rtt)
                if cyc is not None:
                    hist["cycles_log"].append(cyc)
            hist["loss_train"].append(acc_loss / len(order))

            # Validacion en modo 'V': el firmware NO actualiza pesos
            vloss, correct = 0.0, 0
            for xi, yi in zip(Xva, Yva):
                pred, loss, _, _ = link.send("V", xi, yi)
                vloss += loss
                correct += int((pred > 0.5) == bool(yi))
            hist["loss_val"].append(vloss / len(Xva))
            hist["acc_val"].append(100.0 * correct / len(Xva))
            print(f"  epoca {ep:3d}  loss_tr={hist['loss_train'][-1]:.6f}  "
                  f"loss_val={hist['loss_val'][-1]:.6f}  "
                  f"acc_val={hist['acc_val'][-1]:5.1f}%")

        # Curva de salida sobre todo el rango: forward-only, no altera pesos
        xs = np.linspace(-1.2, 1.2, 200)
        hist["curve_x"] = xs.tolist()
        hist["curve_y"] = [link.send("V", x, 0.0)[0] for x in xs]
        hist["wall_time_s"] = time.perf_counter() - t0
    finally:
        link.close()

    rtt = np.array(hist["rtt_log"])
    hist["rtt_mean_ms"] = float(rtt.mean() * 1e3)
    hist["rtt_std_ms"] = float(rtt.std() * 1e3)
    if hist["cycles_log"]:
        cyc = np.array(hist["cycles_log"])          # columnas: forward, backward
        hist["cycles_fwd_mean"] = float(cyc[:, 0].mean())
        hist["cycles_bwd_mean"] = float(cyc[:, 1].mean())
        hist["cycles_mean"] = float(cyc.sum(axis=1).mean())
        # nucleo a 120 MHz -> 1 ciclo = 8.333 ns
        hist["fwd_mean_us"] = hist["cycles_fwd_mean"] / 120.0
        hist["bwd_mean_us"] = hist["cycles_bwd_mean"] / 120.0
        hist["compute_mean_us"] = hist["cycles_mean"] / 120.0
    hist["config"] = {"epochs": args.epocas, "lr": LR, "baud": args.baud,
                      "core_mhz": 120, "uart_clk_mhz": 30,
                      "ascii_scale": core.ASCII_SCALE,
                      "simulado": args.simular,
                      "seed_weights": SEED_WEIGHTS, "seed_data": SEED_DATA,
                      "seed_order": SEED_ORDER}

    save_json("resultados_mcu.json", hist)
    print(f"\nTiempo total: {hist['wall_time_s']:.1f} s")
    print(f"Ida y vuelta por muestra: {hist['rtt_mean_ms']:.2f} "
          f"+/- {hist['rtt_std_ms']:.2f} ms")
    if "compute_mean_us" in hist:
        print(f"Computo en el MCU: {hist['compute_mean_us']:.2f} us por muestra "
              f"({100 * hist['compute_mean_us'] / (hist['rtt_mean_ms'] * 1e3):.4f}% "
              f"del tiempo total)")
    print("Archivo generado: resultados_mcu.json")


if __name__ == "__main__":
    sys.exit(main())