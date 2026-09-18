import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# 0. CARGA Y PREPARACION DE DATOS
RUTA_CSV = "base de datos museos_5.csv"
df = pd.read_csv(RUTA_CSV, sep=";", encoding="latin-1")

target_col = "NOM_TIPO"
le = LabelEncoder()
y = le.fit_transform(df[target_col]).astype(float)  # 0/1

# Se usan 6 features numericas (conteos de boletos) como entrada de la red
feature_cols = [
    "ADU_PAGANTES", "EST_PAGANTES", "NIN_PAGANTES",
    "ADU_NOPAGANTES", "EST_NOPAGANTES", "NIN_NOPAGANTES",
]
X = df[feature_cols].values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Estandarizar (media 0, desviacion 1) ayuda mucho a que la red converja
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Dataset: {df.shape[0]} filas | Target: '{target_col}' -> {list(le.classes_)}")
print(f"Features de entrada: {feature_cols}")


# ARQUITECTURA DE LA RED (3 capas de pesos: entrada->oculta1->oculta2->salida)
N_ENTRADA = X_train.shape[1]   # 6
N_OCULTA1 = 6
N_OCULTA2 = 4
N_SALIDA = 1

# Numero total de parametros (pesos + bias) de toda la red
N_PESOS = (N_ENTRADA * N_OCULTA1 + N_OCULTA1) + \
          (N_OCULTA1 * N_OCULTA2 + N_OCULTA2) + \
          (N_OCULTA2 * N_SALIDA + N_SALIDA)
print(f"Arquitectura: {N_ENTRADA} -> {N_OCULTA1} -> {N_OCULTA2} -> {N_SALIDA}")
print(f"Total de pesos/bias a optimizar con el enjambre: {N_PESOS}")


# 1. REPRESENTACION DE LA PARTICULA (vector de pesos de la red)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def desempacar(vector):
    #Convierte el vector plano (posicion del lobo) en matrices W1,b1,W2,b2,W3,b3
    i = 0
    W1 = vector[i:i + N_ENTRADA * N_OCULTA1].reshape(N_ENTRADA, N_OCULTA1); i += N_ENTRADA * N_OCULTA1
    b1 = vector[i:i + N_OCULTA1]; i += N_OCULTA1
    W2 = vector[i:i + N_OCULTA1 * N_OCULTA2].reshape(N_OCULTA1, N_OCULTA2); i += N_OCULTA1 * N_OCULTA2
    b2 = vector[i:i + N_OCULTA2]; i += N_OCULTA2
    W3 = vector[i:i + N_OCULTA2 * N_SALIDA].reshape(N_OCULTA2, N_SALIDA); i += N_OCULTA2 * N_SALIDA
    b3 = vector[i:i + N_SALIDA]; i += N_SALIDA
    return W1, b1, W2, b2, W3, b3


def forward(vector, X):
    #Propagacion hacia adelante (NO hay retropropagacion en ningun momento)
    W1, b1, W2, b2, W3, b3 = desempacar(vector)
    a1 = np.tanh(X @ W1 + b1)
    a2 = np.tanh(a1 @ W2 + b2)
    salida = sigmoid(a2 @ W3 + b3).ravel()
    return salida


# 2. FUNCION DE APTITUD (FITNESS = perdida de la red, se MINIMIZA)
def fitness(vector):
    #Entropia cruzada binaria entre la salida de la red y el target real
    y_hat = forward(vector, X_train)
    eps = 1e-9
    y_hat = np.clip(y_hat, eps, 1 - eps)
    perdida = -np.mean(y_train * np.log(y_hat) + (1 - y_train) * np.log(1 - y_hat))
    return perdida


#PARAMETROS DEL ALGORITMO GWO
N_LOBOS = 20
MAX_ITER = 60
LIM_INF, LIM_SUP = -2.0, 2.0   #rango inicial de pesos
rng = np.random.default_rng(42)

# 3. INICIALIZACION DEL ENJAMBRE (manada de lobos)
lobos = rng.uniform(LIM_INF, LIM_SUP, size=(N_LOBOS, N_PESOS))
fit_lobos = np.array([fitness(l) for l in lobos])

# Se identifica a los 3 mejores lobos: alfa (mejor), beta (2do), delta (3ro)
orden = np.argsort(fit_lobos)  #menor perdida = mejor
alfa_pos, beta_pos, delta_pos = lobos[orden[0]].copy(), lobos[orden[1]].copy(), lobos[orden[2]].copy()
alfa_fit, beta_fit, delta_fit = fit_lobos[orden[0]], fit_lobos[orden[1]], fit_lobos[orden[2]]

print("\n--- Manada inicial ---")
acc_inicial = accuracy_score(y_train, (forward(alfa_pos, X_train) > 0.5).astype(int))
print(f"Perdida (alfa) inicial: {alfa_fit:.4f} | accuracy train: {acc_inicial:.4f}")


# 4-5. COMPORTAMIENTO DE LA PARTICULA + EVOLUCION (ciclo principal de GWO)
# En GWO, el coeficiente "a" decrece linealmente de 2 a 0: al inicio favorece la EXPLORACION (movimientos grandes) y al final la EXPLOTACION
for it in range(1, MAX_ITER + 1):
    a = 2 - it * (2 / MAX_ITER)

    for i in range(N_LOBOS):
        nueva_pos = np.zeros(N_PESOS)
        for lider_pos in (alfa_pos, beta_pos, delta_pos):
            r1, r2 = rng.uniform(0, 1, N_PESOS), rng.uniform(0, 1, N_PESOS)
            A = 2 * a * r1 - a          # coeficiente de exploracion/explotacion
            C = 2 * r2                  # coeficiente aleatorio de atraccion
            D = np.abs(C * lider_pos - lobos[i])   # distancia al lider
            nueva_pos += lider_pos - A * D
        lobos[i] = np.clip(nueva_pos / 3.0, LIM_INF - 2, LIM_SUP + 2)

    fit_lobos = np.array([fitness(l) for l in lobos])

    #Re-ordenar la jerarquia: puede haber nuevo alfa/beta/delta este ciclo, incluyendo entre los lideres anteriores para no perder la mejor solucion
    candidatos_pos = np.vstack([lobos, alfa_pos, beta_pos, delta_pos])
    candidatos_fit = np.concatenate([fit_lobos, [alfa_fit, beta_fit, delta_fit]])
    orden = np.argsort(candidatos_fit)
    alfa_pos, beta_pos, delta_pos = (candidatos_pos[orden[0]].copy(),
                                      candidatos_pos[orden[1]].copy(),
                                      candidatos_pos[orden[2]].copy())
    alfa_fit, beta_fit, delta_fit = (candidatos_fit[orden[0]],
                                      candidatos_fit[orden[1]],
                                      candidatos_fit[orden[2]])

    if it % 5 == 0 or it == 1:
        acc_train = accuracy_score(y_train, (forward(alfa_pos, X_train) > 0.5).astype(int))
        print(f"Iteracion {it:02d}/{MAX_ITER} -> perdida (alfa) = {alfa_fit:.4f} "
              f"| accuracy train = {acc_train:.4f} | a = {a:.2f}")

# 6. FINALIZACION: mejor red encontrada (lobo alfa) y evaluacion final
print("\n=================== RESULTADO FINAL (GWO) ===================")
y_pred_train = (forward(alfa_pos, X_train) > 0.5).astype(int)
y_pred_test = (forward(alfa_pos, X_test) > 0.5).astype(int)

print(f"Perdida final (entropia cruzada, train): {alfa_fit:.4f}")
print(f"Accuracy en TRAIN: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Accuracy en TEST:  {accuracy_score(y_test, y_pred_test):.4f}")
print(f"Total de pesos optimizados sin backpropagation: {N_PESOS}")