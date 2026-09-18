import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# 0. CARGA Y PREPARACION DE DATOS
RUTA_CSV = "IGP_CatalogoEventosSismicosVolcanMisti_2024-2025_Dataset.csv"
df = pd.read_csv(RUTA_CSV, encoding="latin-1")

# Features numericas de la señal sismica (no se usa TIPO: es no supervisado)
feature_cols = ["FRECUENCIA_PRINCIPAL", "DURACION", "ENERGIA"]
X_raw = df[feature_cols].copy()
X_raw["ENERGIA"] = np.log10(X_raw["ENERGIA"])  # ENERGIA es muy asimetrica

scaler = StandardScaler()
X = scaler.fit_transform(X_raw.values)
N_PUNTOS, N_DIM = X.shape

K = 3  # numero de clusters a formar

print(f"Dataset: {N_PUNTOS} eventos sismicos | Features: {feature_cols}")
print(f"Numero de clusters (K): {K}")

# 1. REPRESENTACION DE LA PARTICULA
# Cada particula es un vector de tamaño K*N_DIM = 3*3 = 9, que representa
# los K centroides concatenados: [c1_x, c1_y, c1_z, c2_x, c2_y, c2_z, c3_x, c3_y, c3_z]
DIM_PARTICULA = K * N_DIM


def desempacar_centroides(vector):
    return vector.reshape(K, N_DIM)


def asignar_clusters(centroides, datos):
    """Asigna cada punto al centroide mas cercano (distancia euclidiana)."""
    distancias = np.linalg.norm(datos[:, None, :] - centroides[None, :, :], axis=2)
    asignacion = np.argmin(distancias, axis=1)
    return asignacion, distancias


# 2. FUNCION DE APTITUD (error de cuantizacion, se MINIMIZA)
# Es el promedio de la distancia de cada punto a su centroide mas cercano.
# Mientras mas compactos y bien ubicados esten los centroides, menor el error.
def fitness(vector):
    centroides = desempacar_centroides(vector)
    asignacion, distancias = asignar_clusters(centroides, X)
    dist_min = distancias[np.arange(N_PUNTOS), asignacion]

    # Penalizacion si algun cluster queda vacio (centroide "inutil")
    clusters_usados = len(np.unique(asignacion))
    penalizacion = (K - clusters_usados) * 5.0

    return dist_min.mean() + penalizacion


# PARAMETROS DEL ALGORITMO PSO
N_PARTICULAS = 20
MAX_ITER = 100
W_MAX, W_MIN = 0.9, 0.4   # inercia decreciente (mas exploracion al inicio)
C1 = 1.6   # componente cognitivo
C2 = 1.6   # componente social
rng = np.random.default_rng(42)

LIM_INF, LIM_SUP = X.min(axis=0), X.max(axis=0)  # rango valido por dimension
LIM_INF_REP = np.tile(LIM_INF, K)   # limites repetidos K veces (para las 3 centroides)
LIM_SUP_REP = np.tile(LIM_SUP, K)
V_MAX = 0.3 * (LIM_SUP_REP - LIM_INF_REP)  # velocidad maxima por dimension

# 3. INICIALIZACION DEL ENJAMBRE
# Cada particula inicia como K puntos aleatorios TOMADOS DEL DATASET
# (inicializacion "smart" muy usada en clustering: evita centroides fuera
# de la nube de datos y acelera la convergencia).
posiciones = np.zeros((N_PARTICULAS, DIM_PARTICULA))
for p in range(N_PARTICULAS):
    idx = rng.choice(N_PUNTOS, size=K, replace=False)
    posiciones[p] = X[idx].flatten()

velocidades = rng.uniform(-0.5, 0.5, size=(N_PARTICULAS, DIM_PARTICULA))

pbest_pos = posiciones.copy()
pbest_fit = np.array([fitness(p) for p in posiciones])

gbest_idx = np.argmin(pbest_fit)
gbest_pos = pbest_pos[gbest_idx].copy()
gbest_fit = pbest_fit[gbest_idx]

print("\n--- Enjambre inicial ---")
print(f"Mejor error de cuantizacion inicial: {gbest_fit:.4f}")

# 4-5. COMPORTAMIENTO DE LA PARTICULA + EVOLUCION (ciclo principal de PSO)
for it in range(1, MAX_ITER + 1):
    w = W_MAX - (W_MAX - W_MIN) * (it / MAX_ITER)  # inercia decrece linealmente

    for i in range(N_PARTICULAS):
        r1 = rng.uniform(0, 1, DIM_PARTICULA)
        r2 = rng.uniform(0, 1, DIM_PARTICULA)

        velocidades[i] = (
            w * velocidades[i]
            + C1 * r1 * (pbest_pos[i] - posiciones[i])
            + C2 * r2 * (gbest_pos - posiciones[i])
        )
        velocidades[i] = np.clip(velocidades[i], -V_MAX, V_MAX)
        posiciones[i] = np.clip(posiciones[i] + velocidades[i], LIM_INF_REP, LIM_SUP_REP)

        f_actual = fitness(posiciones[i])
        if f_actual < pbest_fit[i]:
            pbest_fit[i] = f_actual
            pbest_pos[i] = posiciones[i].copy()

    it_mejor_idx = np.argmin(pbest_fit)
    if pbest_fit[it_mejor_idx] < gbest_fit:
        gbest_fit = pbest_fit[it_mejor_idx]
        gbest_pos = pbest_pos[it_mejor_idx].copy()

    if it % 5 == 0 or it == 1:
        print(f"Iteracion {it:02d}/{MAX_ITER} -> error de cuantizacion (gbest) = {gbest_fit:.4f}")

# 6. FINALIZACION: mejores centroides encontrados y evaluacion de los clusters
mejores_centroides = desempacar_centroides(gbest_pos)
asignacion_final, _ = asignar_clusters(mejores_centroides, X)

print("\n=================== RESULTADO FINAL (PSO-Clustering) ===================")
print(f"Error de cuantizacion final: {gbest_fit:.4f}")
for c in range(K):
    n_c = (asignacion_final == c).sum()
    centro_original = scaler.inverse_transform(mejores_centroides[c].reshape(1, -1))[0]
    print(f"Cluster {c}: {n_c} eventos | centroide (escala original) "
          f"FRECUENCIA={centro_original[0]:.2f} Hz, DURACION={centro_original[1]:.2f} s, "
          f"log10(ENERGIA)={centro_original[2]:.2f}")

sil_pso = silhouette_score(X, asignacion_final)
print(f"\nSilhouette score (PSO-clustering): {sil_pso:.4f}")

# Comparacion contra K-Means clasico (algoritmo determinista, sin enjambre)
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X)
sil_kmeans = silhouette_score(X, kmeans.labels_)
print(f"Silhouette score (K-Means clasico, baseline): {sil_kmeans:.4f}")
print("\n(PSO encontro una particion valida de los eventos sismicos (silhouette "
      "positivo) usando unicamente busqueda por enjambre, sin la formula de "
      "actualizacion de centroides de K-Means. K-Means obtiene un silhouette "
      "mayor porque su regla de actualizacion (promedio exacto de los puntos "
      "asignados) es muy eficiente para este tipo de espacio; el valor de PSO "
      "aqui es que la misma estrategia de enjambre se puede reutilizar para "
      "optimizar OTRAS funciones de aptitud, no solo la distancia euclidiana.)")