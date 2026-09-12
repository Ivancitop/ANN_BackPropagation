"""
mlp_core.py
===========
Modelo de referencia en PC del perceptron multicapa 1-2-3-1 implementado en
el S32K312. Este modulo NO se ejecuta solo: lo importan run_pc.py, run_mcu.py
y graficas.py para garantizar que ambas implementaciones (PC y bare metal)
compartan exactamente:

  - el mismo dataset y la misma normalizacion,
  - los mismos pesos iniciales,
  - el mismo orden de presentacion de las muestras en cada epoca.

Sin esas tres condiciones, comparar la curva de perdida de la PC contra la
del microcontrolador no demuestra nada: cualquier diferencia podria venir
del punto de partida o del barajado, no de la implementacion.
"""

import json
import numpy as np

# =====================================================================
# 1. Normalizacion de la entrada
# =====================================================================
# Mayusculas: ASCII 65..90   -> etiqueta 0
# Minusculas: ASCII 97..122  -> etiqueta 1
# 93.5 es el punto medio entre ambos rangos, asi la frontera de decision
# del problema cae exactamente en x = 0.
ASCII_CENTER = 93.5

# CORREGIDO: en clase1.py y MLP.py estaba en 1.0, pero el firmware define
# ASCII_SCALE 30.0f. Con escala 1.0 la entrada llega al MCU en [-28.5, 28.5]
# y satura la capa 1 (ver README del reporte). Debe valer 30.0 en los tres
# archivos.
ASCII_SCALE = 30.0

# =====================================================================
# 2. Sigmoide parametrica  f(z) = a / (1 + b*exp(-c*z)) + d
# =====================================================================
# Mismos (a, b, c, d) que zigmoid NeuronP1..NeuronP6 en main.c.
ACT1 = [(1.0, 1.0, 0.5, -1.0),
        (1.0, 1.0, 0.5,  1.0)]
ACT2 = [(1.0, 0.5, 1.0, 0.0),
        (1.0, 1.0, 1.0, 0.0),
        (1.0, 1.5, 1.0, 0.0)]
ACT3 = [(1.0, 1.0, 1.5, 0.0)]

N_IN, N_L1, N_L2, N_L3 = 1, 2, 3, 1

# Limite para el argumento de exp(): evita overflow en float32 (el MCU
# trabaja en precision simple, donde expf(88) ya desborda a +inf).
Z_CLAMP = 80.0


def act(z, a, b, c, d):
    """f(z) = a / (1 + b*exp(-c*z)) + d"""
    arg = np.clip(-c * z, -Z_CLAMP, Z_CLAMP)
    return a / (1.0 + b * np.exp(arg)) + d


def act_der_z(z, a, b, c, d):
    """Derivada calculada desde la preactivacion z (version original).
    Se conserva solo como referencia para verificar act_der_a()."""
    arg = np.clip(-c * z, -Z_CLAMP, Z_CLAMP)
    e = np.exp(arg)
    denom = 1.0 + b * e
    return (a * b * c * e) / (denom ** 2)


def act_der_a(y, a, b, c, d):
    """Derivada calculada desde la ACTIVACION ya conocida, sin exp().

    Con u = (y - d)/a se tiene y = a*u + d y f'(z) = a*c*u*(1-u).
    Es identica a act_der_z() pero elimina una llamada a expf() por neurona
    en el backward. En el MCU son 6 expf() menos por muestra: esta es la
    optimizacion principal del firmware.
    """
    u = (y - d) / a
    return a * c * u * (1.0 - u)


# =====================================================================
# 3. Dataset sintetico
# =====================================================================
def build_dataset(reps=15, noise_std=0.5, seed=0):
    """780 muestras: 26 letras x 15 repeticiones x 2 clases.

    La variabilidad se introduce con ruido gaussiano sobre el codigo ASCII,
    simulando una lectura ruidosa. noise_std=0.5 mantiene el ruido muy por
    debajo del hueco ASCII entre 'Z'(90) y 'a'(97), asi que las clases
    siguen siendo separables.
    """
    r = np.random.default_rng(seed)
    codes, labels = [], []
    for code in range(65, 91):          # A-Z -> 0 (mayuscula)
        for _ in range(reps):
            codes.append(code + r.normal(0, noise_std))
            labels.append(0.0)
    for code in range(97, 123):         # a-z -> 1 (minuscula)
        for _ in range(reps):
            codes.append(code + r.normal(0, noise_std))
            labels.append(1.0)
    codes = np.asarray(codes, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    idx = r.permutation(len(codes))
    return codes[idx], labels[idx]


def encode_ascii(code):
    return (code - ASCII_CENTER) / ASCII_SCALE


def split_dataset(reps=15, noise_std=0.5, seed=0, train_frac=0.8):
    codes, labels = build_dataset(reps, noise_std, seed)
    X = encode_ascii(codes)
    split = int(train_frac * len(X))
    return (X[:split], labels[:split], X[split:], labels[split:])


def epoch_orders(n_train, epochs, seed=7):
    """Permutaciones fijas de cada epoca.

    Se generan una sola vez y se usan IGUALES en PC y en MCU. Con descenso
    de gradiente estocastico (una muestra por actualizacion) el orden altera
    la trayectoria, asi que compartirlo es obligatorio para comparar.
    """
    r = np.random.default_rng(seed)
    return [r.permutation(n_train).tolist() for _ in range(epochs)]


# =====================================================================
# 4. Red
# =====================================================================
class MLP:
    """Perceptron 1-2-3-1 con sigmoide parametrica por neurona.

    dtype=np.float32 reproduce la aritmetica del MCU; dtype=np.float64 da
    la referencia de alta precision. La comparacion entre ambos cuantifica
    el efecto de trabajar en precision simple.
    """

    def __init__(self, seed=1, dtype=np.float64, use_fast_der=True):
        self.dtype = dtype
        self.use_fast_der = use_fast_der
        rng = np.random.default_rng(seed)

        def init_layer(n_in, n_out):
            # Xavier/Glorot para sigmoide: std = sqrt(1/n_in)
            W = rng.normal(0.0, np.sqrt(1.0 / n_in), size=(n_out, n_in))
            b = np.zeros(n_out)
            return W.astype(dtype), b.astype(dtype)

        self.W1, self.b1 = init_layer(N_IN, N_L1)
        self.W2, self.b2 = init_layer(N_L1, N_L2)
        self.W3, self.b3 = init_layer(N_L2, N_L3)

    # ---------------- persistencia ----------------
    def get_weights(self):
        return {k: getattr(self, k).tolist()
                for k in ("W1", "b1", "W2", "b2", "W3", "b3")}

    def set_weights(self, d):
        for k, v in d.items():
            setattr(self, k, np.asarray(v, dtype=self.dtype))

    def export_c(self):
        """Emite los pesos como inicializadores C, para fijar en el firmware
        exactamente el mismo punto de partida que en la PC."""
        def arr(v):
            return ", ".join(f"{x:.9e}f" for x in np.ravel(v))
        return (
            "/* Pesos iniciales exportados desde mlp_core.MLP(seed=1).\n"
            " * Pegar en main.c en lugar de initLayer()/random_gauss() para\n"
            " * que PC y MCU arranquen del mismo punto. */\n"
            f"static const float32 W1_INIT[{self.W1.size}] = {{ {arr(self.W1)} }};\n"
            f"static const float32 B1_INIT[{self.b1.size}] = {{ {arr(self.b1)} }};\n"
            f"static const float32 W2_INIT[{self.W2.size}] = {{ {arr(self.W2)} }};\n"
            f"static const float32 B2_INIT[{self.b2.size}] = {{ {arr(self.b2)} }};\n"
            f"static const float32 W3_INIT[{self.W3.size}] = {{ {arr(self.W3)} }};\n"
            f"static const float32 B3_INIT[{self.b3.size}] = {{ {arr(self.b3)} }};\n"
        )

    # ---------------- forward ----------------
    def forward(self, x):
        """x: vector de 1 elemento. Devuelve (a3, cache)."""
        x = np.asarray(x, dtype=self.dtype)

        z1 = self.W1 @ x + self.b1
        a1 = np.array([act(z1[j], *ACT1[j]) for j in range(N_L1)], dtype=self.dtype)

        z2 = self.W2 @ a1 + self.b2
        a2 = np.array([act(z2[j], *ACT2[j]) for j in range(N_L2)], dtype=self.dtype)

        z3 = self.W3 @ a2 + self.b3
        a3 = np.array([act(z3[j], *ACT3[j]) for j in range(N_L3)], dtype=self.dtype)

        return a3, (x, z1, a1, z2, a2, z3, a3)

    def _der(self, z, a, params):
        return act_der_a(a, *params) if self.use_fast_der else act_der_z(z, *params)

    # ---------------- backward ----------------
    def backward(self, cache, target, lr, update=True):
        """Retropropagacion con perdida MSE 0.5*(a3-y)^2 y SGD por muestra.

        Devuelve la perdida ANTES de actualizar los pesos, que es lo mismo
        que reporta el firmware en TrainStep().
        """
        x, z1, a1, z2, a2, z3, a3 = cache
        target = np.asarray(target, dtype=self.dtype)

        error = a3 - target
        loss = 0.5 * float(np.sum(error ** 2))

        d3 = np.array([error[j] * self._der(z3[j], a3[j], ACT3[j])
                       for j in range(N_L3)], dtype=self.dtype)

        d2 = np.array([(self.W3[:, i] @ d3) * self._der(z2[i], a2[i], ACT2[i])
                       for i in range(N_L2)], dtype=self.dtype)

        d1 = np.array([(self.W2[:, i] @ d2) * self._der(z1[i], a1[i], ACT1[i])
                       for i in range(N_L1)], dtype=self.dtype)

        if update:
            self.W3 -= (lr * np.outer(d3, a2)).astype(self.dtype)
            self.b3 -= (lr * d3).astype(self.dtype)
            self.W2 -= (lr * np.outer(d2, a1)).astype(self.dtype)
            self.b2 -= (lr * d2).astype(self.dtype)
            self.W1 -= (lr * np.outer(d1, x)).astype(self.dtype)
            self.b1 -= (lr * d1).astype(self.dtype)

        return loss, (d1, d2, d3)

    # ---------------- utilidades ----------------
    def train_sample(self, x, y, lr):
        a3, cache = self.forward([x])
        loss, _ = self.backward(cache, [y], lr, update=True)
        return float(a3[0]), loss

    def predict(self, x):
        a3, _ = self.forward([x])
        return float(a3[0])

    def evaluate(self, X, Y):
        """Devuelve (perdida media, exactitud %) sin tocar los pesos."""
        loss, correct = 0.0, 0
        for xi, yi in zip(X, Y):
            p = self.predict(xi)
            loss += 0.5 * (p - yi) ** 2
            correct += int((p > 0.5) == bool(yi))
        return loss / len(X), 100.0 * correct / len(X)


# =====================================================================
# 5. Verificacion numerica del gradiente
# =====================================================================
def gradient_check(seed=1, eps=1e-5, n_samples=20):
    """Compara el gradiente analitico contra diferencias centradas.

    Es la evidencia mas directa de que forward y backward estan bien
    derivados; vale la pena reportar el error relativo maximo en el informe.
    """
    Xtr, Ytr, _, _ = split_dataset()
    net = MLP(seed=seed, dtype=np.float64)
    w0 = net.get_weights()
    max_rel = 0.0

    for k in range(n_samples):
        x, y = Xtr[k], Ytr[k]

        net.set_weights(w0)
        _, cache = net.forward([x])
        _, (d1, d2, d3) = net.backward(cache, [y], lr=0.0, update=False)

        # gradientes analiticos respecto de W y b
        analytic = {
            "W1": np.outer(d1, [x]), "b1": d1,
            "W2": np.outer(d2, cache[2]), "b2": d2,
            "W3": np.outer(d3, cache[4]), "b3": d3,
        }

        for name, g_an in analytic.items():
            g_an = np.atleast_2d(np.asarray(g_an))
            flat = np.ravel(g_an)
            for idx in range(flat.size):
                g_num = _numeric_grad(w0, name, idx, x, y, eps, seed)
                denom = max(1e-12, abs(flat[idx]) + abs(g_num))
                max_rel = max(max_rel, abs(flat[idx] - g_num) / denom)
    return max_rel


def _numeric_grad(w0, name, idx, x, y, eps, seed):
    def loss_at(delta):
        net = MLP(seed=seed, dtype=np.float64)
        w = {k: np.array(v, dtype=np.float64) for k, v in w0.items()}
        flat = np.ravel(w[name])
        flat[idx] += delta
        w[name] = flat.reshape(np.shape(w[name]))
        net.set_weights(w)
        a3, _ = net.forward([x])
        return 0.5 * float((a3[0] - y) ** 2)
    return (loss_at(+eps) - loss_at(-eps)) / (2 * eps)


# =====================================================================
def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    print("Verificacion: act_der_a == act_der_z")
    z = np.linspace(-10, 10, 1001)
    for params in ACT1 + ACT2 + ACT3:
        a = act(z, *params)
        err = np.max(np.abs(act_der_a(a, *params) - act_der_z(z, *params)))
        print(f"  (a,b,c,d)={params}  error max = {err:.3e}")

    print("\nGradient check (diferencias centradas, eps=1e-5):")
    print(f"  error relativo maximo = {gradient_check():.3e}")

    print("\nPesos iniciales para el firmware:\n")
    print(MLP(seed=1).export_c())