import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# 0. CARGA Y PREPARACION DE DATOS
RUTA_CSV = "IGP_CatalogoEventosSismicosVolcanMisti_2024-2025_Dataset.csv"
df = pd.read_csv(RUTA_CSV, encoding="latin-1")

# Target binario: VT (evento volcano-tectonico, el mas comun) vs OTRO.
# Se colapsan las clases minoritarias (VD, TO, LH, LP, ST) en una sola
# categoria "OTRO" porque individualmente tienen muy pocos registros para
# validar un modelo de forma confiable.
y = (df["TIPO"] == "VT").astype(int).values  # 1 = VT, 0 = OTRO

# Features numericas de la señal sismica
feature_cols = ["FRECUENCIA_PRINCIPAL", "DURACION", "ENERGIA", "HORA_UTC"]
X = df[feature_cols].copy()
# ENERGIA es muy asimetrica (varios ordenes de magnitud) -> log transform
X["ENERGIA"] = np.log10(X["ENERGIA"])
X = X.values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Dataset: {df.shape[0]} eventos sismicos")
print(f"Clase VT: {y.sum()} | Clase OTRO: {(y==0).sum()}")
print(f"Features usadas: {feature_cols}")

# 1. REPRESENTACION DE LA PARTICULA
# Cada particula es un vector de 4 numeros reales, uno por cada
# hiperparametro del Random Forest. Los valores reales se escalan/redondean
# a su rango valido antes de entrenar el modelo:
#
#   dim 0 -> n_estimators     (numero de arboles)      rango [20, 200]
#   dim 1 -> max_depth        (profundidad maxima)      rango [2, 20]
#   dim 2 -> min_samples_split(min. muestras p/dividir)  rango [2, 20]
#   dim 3 -> min_samples_leaf (min. muestras por hoja)   rango [1, 10]

LIMITES = np.array([
    [20, 200],   # n_estimators
    [2, 20],     # max_depth
    [2, 20],     # min_samples_split
    [1, 10],     # min_samples_leaf
], dtype=float)
N_DIM = LIMITES.shape[0]


def decodificar(posicion):
    """Convierte el vector continuo de la particula en hiperparametros validos (enteros)."""
    pos_clip = np.clip(posicion, LIMITES[:, 0], LIMITES[:, 1])
    n_estimators, max_depth, min_split, min_leaf = pos_clip
    return {
        "n_estimators": int(round(n_estimators)),
        "max_depth": int(round(max_depth)),
        "min_samples_split": int(round(min_split)),
        "min_samples_leaf": int(round(min_leaf)),
    }


# 2. FUNCION DE APTITUD (FITNESS)
# Se entrena un Random Forest con los hiperparametros que codifica la
# particula y se mide su F1-macro promedio por validacion cruzada (3-fold).
# Se usa F1-macro (y no accuracy) porque las clases estan desbalanceadas.
def fitness(posicion):
    params = decodificar(posicion)
    clf = RandomForestClassifier(
        **params,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    scores = cross_val_score(clf, X_train, y_train, cv=3, scoring="f1_macro")
    return scores.mean()


# PARAMETROS DEL ALGORITMO PSO
N_PARTICULAS = 8
MAX_ITER = 12
W = 0.6          # inercia
C1 = 1.5         # componente cognitivo (atraccion al mejor personal)
C2 = 1.5         # componente social (atraccion al mejor global)
rng = np.random.default_rng(42)

# 3. INICIALIZACION DEL ENJAMBRE
rango = LIMITES[:, 1] - LIMITES[:, 0]
posiciones = LIMITES[:, 0] + rng.uniform(0, 1, size=(N_PARTICULAS, N_DIM)) * rango
velocidades = rng.uniform(-1, 1, size=(N_PARTICULAS, N_DIM)) * (rango * 0.1)

pbest_pos = posiciones.copy()
pbest_fit = np.array([fitness(p) for p in posiciones])

gbest_idx = np.argmax(pbest_fit)
gbest_pos = pbest_pos[gbest_idx].copy()
gbest_fit = pbest_fit[gbest_idx]

print("\n--- Enjambre inicial ---")
print(f"Mejor fitness (F1-macro) inicial: {gbest_fit:.4f} -> {decodificar(gbest_pos)}")

# 4-5. COMPORTAMIENTO DE LA PARTICULA + EVOLUCION (ciclo principal de PSO)
for it in range(1, MAX_ITER + 1):
    for i in range(N_PARTICULAS):
        r1 = rng.uniform(0, 1, size=N_DIM)
        r2 = rng.uniform(0, 1, size=N_DIM)

        # Actualizacion de velocidad: inercia + atraccion cognitiva + social
        velocidades[i] = (
            W * velocidades[i]
            + C1 * r1 * (pbest_pos[i] - posiciones[i])
            + C2 * r2 * (gbest_pos - posiciones[i])
        )
        # Actualizacion de posicion, restringida a los limites validos
        posiciones[i] = np.clip(posiciones[i] + velocidades[i], LIMITES[:, 0], LIMITES[:, 1])

        # Evaluar la nueva posicion y actualizar el mejor personal (pbest)
        f_actual = fitness(posiciones[i])
        if f_actual > pbest_fit[i]:
            pbest_fit[i] = f_actual
            pbest_pos[i] = posiciones[i].copy()

    # Actualizar el mejor global (gbest)
    it_mejor_idx = np.argmax(pbest_fit)
    if pbest_fit[it_mejor_idx] > gbest_fit:
        gbest_fit = pbest_fit[it_mejor_idx]
        gbest_pos = pbest_pos[it_mejor_idx].copy()

    print(f"Iteracion {it:02d}/{MAX_ITER} -> mejor F1-macro = {gbest_fit:.4f} "
          f"| hiperparametros = {decodificar(gbest_pos)}")

# 6. FINALIZACION: mejores hiperparametros encontrados y evaluacion final
mejores_params = decodificar(gbest_pos)
print("\n=================== RESULTADO FINAL (PSO) ===================")
print(f"Mejores hiperparametros encontrados: {mejores_params}")
print(f"F1-macro (train, CV) de la mejor solucion: {gbest_fit:.4f}")

clf_final = RandomForestClassifier(**mejores_params, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
clf_final.fit(X_train, y_train)
from sklearn.metrics import f1_score, accuracy_score
y_pred = clf_final.predict(X_test)
print(f"Accuracy en TEST: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-macro en TEST: {f1_score(y_test, y_pred, average='macro'):.4f}")

# Comparacion: Random Forest con hiperparametros por defecto (baseline)
clf_base = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
clf_base.fit(X_train, y_train)
y_pred_base = clf_base.predict(X_test)
print(f"\nBaseline (hiperparametros por defecto de sklearn):")
print(f"Accuracy en TEST: {accuracy_score(y_test, y_pred_base):.4f}")
print(f"F1-macro en TEST: {f1_score(y_test, y_pred_base, average='macro'):.4f}")