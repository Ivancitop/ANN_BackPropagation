"""
graficas.py
===========
Genera todas las figuras del informe a partir de resultados_pc.json,
resultados_pc_f32.json, resultados_escala.json y resultados_mcu.json.

Salida en PDF vectorial dimensionado a una columna IEEE (3.4 in), asi las
figuras no se reescalan al insertarlas y el texto conserva su tamano.

Uso:
    python run_pc.py
    python run_mcu.py            (o --simular)
    python graficas.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mlp_core as core
from mlp_core import load_json, act, act_der_a, ACT1, ACT2, ACT3

OUT = "figuras"
COL = 3.4          # ancho de columna IEEE en pulgadas
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "lines.linewidth": 1.2, "legend.framealpha": 0.9,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=200)
    plt.close(fig)
    print(f"  {path}.pdf")


# =====================================================================
def fig_perdida(pc, mcu):
    """Curva de perdida. Es la figura central del informe: con los mismos
    pesos iniciales y el mismo orden, las trazas de PC y MCU deben quedar
    superpuestas. Si divergen, el backward en C tiene un error."""
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ep = np.arange(1, len(pc["loss_train"]) + 1)

    ax.semilogy(ep, pc["loss_train"], "-", color="C0", label="PC · entrenamiento")
    ax.semilogy(ep, pc["loss_val"], "--", color="C0", label="PC · validación")
    if mcu:
        em = np.arange(1, len(mcu["loss_train"]) + 1)
        ax.semilogy(em, mcu["loss_train"], "o", color="C3", ms=2.5,
                    mfc="none", label="MCU · entrenamiento")
        ax.semilogy(em, mcu["loss_val"], "s", color="C1", ms=2.5,
                    mfc="none", label="MCU · validación")

    ax.set_xlabel("Época")
    ax.set_ylabel(r"Pérdida MSE  $\frac{1}{2}(a^{(3)}-y)^2$")
    ax.legend(loc="upper right")
    save(fig, "fig_perdida")


def fig_exactitud(pc, mcu):
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    ep = np.arange(1, len(pc["acc_val"]) + 1)
    ax.plot(ep, pc["acc_train"], "-", color="C0", label="PC · entrenamiento")
    ax.plot(ep, pc["acc_val"], "--", color="C2", label="PC · validación")
    if mcu:
        ax.plot(np.arange(1, len(mcu["acc_val"]) + 1), mcu["acc_val"], "s",
                color="C3", ms=2.5, mfc="none", label="MCU · validación")
    ax.set_xlabel("Época")
    ax.set_ylabel("Exactitud [%]")
    ax.set_ylim(None, 101)
    ax.legend(loc="lower right")
    save(fig, "fig_exactitud")


def fig_equivalencia(pc, mcu, pc32):
    """Diferencia muestra a muestra entre la prediccion de la PC y la del MCU.

    Es la evidencia mas fuerte de que el forward/backward en C es correcto:
    si la diferencia se queda en el orden del epsilon de float32 (~1e-7
    relativo) y no crece, la unica discrepancia es el redondeo, no la
    formulacion.
    """
    if not mcu:
        return
    n = min(len(pc["pred_log"]), len(mcu["pred_log"]))
    d_mcu = np.abs(np.array(pc["pred_log"][:n]) - np.array(mcu["pred_log"][:n]))
    d_f32 = np.abs(np.array(pc["pred_log"][:n]) - np.array(pc32["pred_log"][:n]))
    d_mcu = np.maximum(d_mcu, 1e-16)
    d_f32 = np.maximum(d_f32, 1e-16)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * COL, 2.1),
                                   gridspec_kw={"width_ratios": [2.2, 1]})
    k = np.arange(n)
    ax1.semilogy(k, d_mcu, ".", ms=1, color="C3", alpha=0.5,
                 label="|PC$_{64}$ − MCU|")
    ax1.semilogy(k, d_f32, ".", ms=1, color="C0", alpha=0.35,
                 label="|PC$_{64}$ − PC$_{32}$| (solo redondeo)")
    ax1.axhline(np.finfo(np.float32).eps, color="k", ls=":", lw=0.8,
                label=r"$\epsilon_{float32}$")
    ax1.set_xlabel("Muestra de entrenamiento (acumulada)")
    ax1.set_ylabel("Diferencia absoluta en $a^{(3)}$")
    ax1.legend(loc="upper left", markerscale=6)

    ax2.hist(np.log10(d_mcu), bins=40, color="C3", alpha=0.8)
    ax2.set_xlabel(r"$\log_{10}$ |PC − MCU|")
    ax2.set_ylabel("Frecuencia")
    fig.tight_layout()
    save(fig, "fig_equivalencia")

    print(f"     dif. max PC-MCU  = {d_mcu.max():.3e}")
    print(f"     dif. media PC-MCU = {d_mcu.mean():.3e}")


def fig_frontera(pc, mcu):
    """Con una sola entrada, la salida de la red sobre el rango completo ES
    la frontera de decision aprendida. Muestra donde quedo el umbral y con
    que pendiente, algo que la curva de perdida no revela."""
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    x = np.array(pc["curve_x"])
    ax.plot(x, pc["curve_y"], "-", color="C0", label="PC")
    if mcu:
        ax.plot(mcu["curve_x"], mcu["curve_y"], "--", color="C3", label="MCU")

    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.axvline(0.0, color="k", ls=":", lw=0.8)

    Xv, Yv = np.array(pc["X_val"]), np.array(pc["y_val"])
    ax.plot(Xv[Yv == 0], np.zeros((Yv == 0).sum()) + 0.02, "|", color="C2",
            ms=5, alpha=0.5, label="mayúsculas (y=0)")
    ax.plot(Xv[Yv == 1], np.ones((Yv == 1).sum()) - 0.02, "|", color="C4",
            ms=5, alpha=0.5, label="minúsculas (y=1)")

    sc, ce = pc["config"]["ascii_scale"], pc["config"]["ascii_center"]
    sec = ax.secondary_xaxis("top", functions=(lambda v: v * sc + ce,
                                               lambda v: (v - ce) / sc))
    sec.set_xlabel("Código ASCII")
    ax.set_xlabel(r"Entrada normalizada $x=(\mathrm{ASCII}-93.5)/30$")
    ax.set_ylabel(r"Salida $a^{(3)}$")
    ax.legend(loc="center left", fontsize=6)
    save(fig, "fig_frontera")


def fig_activaciones():
    """Las 6 sigmoides parametricas y sus derivadas. Justifica en el informe
    por que cada neurona responde distinto pese a compartir la formula."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * COL, 2.1))
    z = np.linspace(-8, 8, 600)
    todas = [("$N_1$ (capa 1)", ACT1[0]), ("$N_2$ (capa 1)", ACT1[1]),
             ("$N_3$ (capa 2)", ACT2[0]), ("$N_4$ (capa 2)", ACT2[1]),
             ("$N_5$ (capa 2)", ACT2[2]), ("$N_6$ (salida)", ACT3[0])]
    for i, (name, p) in enumerate(todas):
        a = act(z, *p)
        ax1.plot(z, a, color=f"C{i}", label=f"{name}  b={p[1]}, c={p[2]}, d={p[3]}")
        ax2.plot(z, act_der_a(a, *p), color=f"C{i}")
    ax1.set_xlabel("$z$"); ax1.set_ylabel(r"$f(z)=\frac{a}{1+be^{-cz}}+d$")
    ax2.set_xlabel("$z$"); ax2.set_ylabel("$f'(z)=a\\,c\\,u(1-u)$")
    ax1.legend(loc="upper left", fontsize=5.5)
    fig.tight_layout()
    save(fig, "fig_activaciones")


def fig_gradientes(esc):
    """Magnitud media del error local por capa.

    Con ASCII_SCALE=1 la entrada llega en [-28.5, 28.5]: la capa 1 opera en
    la zona plana de la sigmoide y su gradiente colapsa varios ordenes de
    magnitud, de modo que deja de aprender. Con ASCII_SCALE=30 las tres
    capas mantienen gradientes del mismo orden.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * COL, 2.1), sharey=True)
    for ax, key, tit in ((ax1, "scale_1", "Sin normalizar (scale = 1)"),
                         (ax2, "scale_30", "Normalizada (scale = 30)")):
        s = esc[key]
        ep = np.arange(1, len(s["grad_l1"]) + 1)
        ax.semilogy(ep, s["grad_l1"], color="C3", label=r"capa 1  $|\delta^{(1)}|$")
        ax.semilogy(ep, s["grad_l2"], color="C1", label=r"capa 2  $|\delta^{(2)}|$")
        ax.semilogy(ep, s["grad_l3"], color="C0", label=r"capa 3  $|\delta^{(3)}|$")
        ax.set_title(tit)
        ax.set_xlabel("Época")
    ax1.set_ylabel("Magnitud media del gradiente")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig_gradientes")

    r1 = esc["scale_1"]["grad_l3"][-1] / esc["scale_1"]["grad_l1"][-1]
    r3 = esc["scale_30"]["grad_l3"][-1] / esc["scale_30"]["grad_l1"][-1]
    print(f"     razón |δ3|/|δ1| final — scale 1: {r1:.1f}x   scale 30: {r3:.1f}x")
    print(f"     |z1| máx — scale 1: {max(esc['scale_1']['z1_absmax']):.1f}   "
          f"scale 30: {max(esc['scale_30']['z1_absmax']):.2f}  "
          f"(expf desborda en float32 si |c·z| > 88.7)")


def fig_tiempos(pc, mcu):
    """Reparto del tiempo. El punto que interesa para el informe: a 9600
    baud el transporte serie domina por varios ordenes de magnitud sobre el
    computo, asi que el cuello de botella NO es la red neuronal."""
    if not mcu:
        return
    t_link = mcu["rtt_mean_ms"] * 1e3                     # us
    t_mcu = mcu.get("compute_mean_us")                    # us, requiere DWT
    t_pc = pc["time_per_sample_us"]                       # us

    etiquetas, valores, colores = [], [], []
    etiquetas.append("Ida y vuelta\nserie @ 9600 bd"); valores.append(t_link); colores.append("C3")
    if mcu.get("fwd_mean_us"):
        etiquetas.append("Forward MCU\n(DWT, 120 MHz)")
        valores.append(mcu["fwd_mean_us"]); colores.append("C0")
        etiquetas.append("Backward MCU\n(DWT, 120 MHz)")
        valores.append(mcu["bwd_mean_us"]); colores.append("C1")
    elif t_mcu:
        etiquetas.append("Cómputo MCU\n(DWT, 120 MHz)"); valores.append(t_mcu); colores.append("C0")
    etiquetas.append("Cómputo PC\n(NumPy)"); valores.append(t_pc); colores.append("C2")

    fig, ax = plt.subplots(figsize=(COL, 2.2))
    b = ax.bar(etiquetas, valores, color=colores, width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel(r"Tiempo por muestra [$\mu$s]")
    for rect, v in zip(b, valores):
        ax.text(rect.get_x() + rect.get_width() / 2, v * 1.25,
                f"{v:,.1f}", ha="center", fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6)
    ax.set_ylim(top=max(valores) * 6)
    save(fig, "fig_tiempos")

    if t_mcu:
        print(f"     el cómputo es el {100 * t_mcu / t_link:.4f}% del tiempo total")


def fig_memoria(text=74324, data=4, bss=6184,
                flash_region=1024 * 1024, ram_region=96 * 1024):
    """Huella de memoria a partir de arm-none-eabi-size.

    text/data/bss vienen del build real. flash_region y ram_region deben
    tomarse del LINKER SCRIPT del proyecto (int_pflash e int_sram en el .ld),
    no de una hoja de datos: el .ld es lo que el enlazador realmente usa y
    es la fuente citable en el informe.
    """
    flash = text + data
    ram = data + bss

    # RAM atribuible a la aplicacion (lo demas es RTD + newlib)
    app_bss = {"au8Buffer[256]": 256, "banderas de la ISR": 2}
    # OJO: hoy estos viven en la PILA porque son locales de main()/TrainStep.
    # No aparecen en bss. Tras el parche [7] pasan a .bss y si apareceran.
    en_pila = {"Layer1..3": 120, "Act1..3 (a Flash tras el parche)": 96,
               "ForwardCache x2": 96, "txBuf[32]": 32, "delta1..3": 24}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * COL, 2.2),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # --- panel izquierdo: ocupacion frente al recurso disponible ---
    for i, (nom, usado, total) in enumerate(
            (("Flash", flash, flash_region), ("RAM", ram, ram_region))):
        ax1.barh(i, 100 * usado / total, color="C3", height=0.45)
        ax1.barh(i, 100, color="0.88", height=0.45, zorder=0)
        ax1.text(100 * usado / total + 2, i,
                 f"{usado / 1024:.1f} KiB  ({100 * usado / total:.1f} %)",
                 va="center", fontsize=6.5)
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(["Flash", "RAM"])
    ax1.set_xlim(0, 100); ax1.set_xlabel("Ocupación del recurso [%]")
    ax1.grid(axis="y", visible=False)

    # --- panel derecho: a que se va la RAM estatica ---
    resto = ram - sum(app_bss.values())
    piezas = list(app_bss.items()) + [("RTD + newlib (resto de .bss)", resto)]
    base = 0.0
    for i, (k, v) in enumerate(piezas):
        ax2.barh(0, v, left=base, color=f"C{i}", height=0.42,
                 label=f"{k}: {v} B")
        base += v
    base = 0.0
    for i, (k, v) in enumerate(en_pila.items()):
        ax2.barh(1, v, left=base, color=f"C{i}", height=0.42, alpha=0.6)
        base += v
    ax2.text(base + 60, 1, f"{sum(en_pila.values())} B en pila", va="center",
             fontsize=6.5)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels([".bss + .data", "pila (main)"], fontsize=7)
    ax2.set_xlabel("Bytes")
    ax2.grid(axis="y", visible=False)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), fontsize=5.5)

    fig.tight_layout()
    save(fig, "fig_memoria")

    print(f"     Flash {flash:,} B ({100 * flash / flash_region:.2f} %)   "
          f"RAM {ram:,} B ({100 * ram / ram_region:.2f} %, sin pila)")
    print(f"     parámetros entrenables: 68 B = "
          f"{100 * 68 / ram:.2f} % de la RAM estática")
    print(f"     reservado por MAX_W/MAX_OUT: 108 B para 68 B reales "
          f"(40 B sin usar)")
    print(f"     RAM atribuible a tu código: {sum(app_bss.values())} B; "
          f"el resto ({resto} B) es RTD + newlib")


# =====================================================================
def main():
    pc = load_json("resultados_pc.json")
    pc32 = load_json("resultados_pc_f32.json")
    esc = load_json("resultados_escala.json")
    mcu = load_json("resultados_mcu.json") if os.path.exists("resultados_mcu.json") else None

    if mcu and mcu["config"].get("simulado"):
        print("AVISO: resultados_mcu.json viene de --simular, no de hardware real.\n")

    print("Generando figuras en ./figuras/")
    fig_perdida(pc, mcu)
    fig_exactitud(pc, mcu)
    fig_equivalencia(pc, mcu, pc32)
    fig_frontera(pc, mcu)
    fig_activaciones()
    fig_gradientes(esc)
    fig_tiempos(pc, mcu)
    fig_memoria()

    # ---- resumen numerico para pegar en las tablas del informe ----
    print("\nPara las tablas del informe:")
    print(f"  perdida final PC   : {pc['loss_train'][-1]:.6e}")
    print(f"  exactitud val PC   : {pc['acc_val'][-1]:.1f} %")
    if mcu:
        print(f"  perdida final MCU  : {mcu['loss_train'][-1]:.6e}")
        print(f"  exactitud val MCU  : {mcu['acc_val'][-1]:.1f} %")
        print(f"  ida y vuelta serie : {mcu['rtt_mean_ms']:.2f} ms/muestra")
        print(f"  tiempo total MCU   : {mcu['wall_time_s'] / 60:.1f} min")
    d = np.abs(np.array(pc["loss_train"]) - np.array(pc32["loss_train"])).max()
    print(f"  dif. maxima float64 vs float32 en la perdida por epoca: {d:.3e}")


if __name__ == "__main__":
    main()