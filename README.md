# Algoritmos-de-Enjambre-Machine-Learning

Repositorio con ejemplos prácticos de **algoritmos bio-inspirados de optimización por enjambres** aplicados a problemas reales de Machine Learning: selección de características, ajuste de hiperparámetros, entrenamiento de redes neuronales sin backpropagation y clustering no supervisado.

Cada ejemplo incluye una versión en script (`.py`) y una versión lista para ejecutar en **Google Colab** (`.ipynb`).

---

## Estructura del repositorio

| Archivo | Algoritmo | Aplicación |
|---|---|---|
| `Ejemplo01 Seleccion Caracteristicas Abc.py` | **ABC** (Artificial Bee Colony) | Selección de características |
| `Ejemplo02 Ajuste Hiperparametros Pso.py` | **PSO** (Particle Swarm Optimization) | Ajuste de hiperparámetros |
| `Ejemplo03 Entrenamiento Red Neuronal sin Backpropagation.py` | **GWO** (Grey Wolf Optimizer) | Entrenamiento de red neuronal |
| `Ejemplo04 Agrupamiento Clustering.py` | **PSO** (Particle Swarm Optimization) | Clustering no supervisado |

Cada script tiene su equivalente `.ipynb` ("Subida para abrir en Colab") para ejecutarlo directamente en el navegador sin instalar nada.

### Datasets utilizados

- **`base de datos museos_5.csv`** — Registros de visitantes a museos (boletos pagantes/no pagantes, nacionales/extranjeros). Usado en los Ejemplos 01 y 03.
- **`IGP_CatalogoEventosSismicosVolcanMisti_2024-2025_Dataset.csv`** — Catálogo de eventos sísmicos del volcán Misti (IGP), con frecuencia, duración, energía y tipo de evento. Usado en los Ejemplos 02 y 04.

---

## Descripción de cada ejemplo

### 1️. Selección de características con ABC (Colonia Artificial de Abejas)

**Objetivo:** encontrar el subconjunto óptimo de columnas del dataset de museos que mejor predice si un visitante es **Nacional** o **Extranjero**, usando la menor cantidad posible de features.

- **Representación:** cada fuente de alimento es un vector continuo que se binariza (umbral 0.5) para indicar qué features se seleccionan.
- **Fitness:** accuracy promedio (validación cruzada 3-fold) de un `DecisionTreeClassifier`, penalizado levemente por la cantidad de features usadas.
- **Fases del algoritmo:** abejas empleadas → abejas observadoras (selección por ruleta según fitness) → abejas exploradoras (reinician fuentes agotadas tras `LIMIT` intentos sin mejora).
- **Resultado:** compara el modelo con las features seleccionadas por ABC contra un baseline entrenado con todas las features.

### 2️. Ajuste de hiperparámetros con PSO (Optimización por Enjambre de Partículas)

**Objetivo:** encontrar la mejor combinación de hiperparámetros de un `RandomForestClassifier` para clasificar eventos sísmicos como **VT (volcano-tectónico)** vs **OTRO**.

- **Representación:** cada partícula es un vector de 4 dimensiones (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`), decodificado a valores enteros válidos.
- **Fitness:** F1-macro promedio (validación cruzada 3-fold), elegido por el desbalance entre clases.
- **Dinámica:** actualización clásica de velocidad/posición con componente de inercia (`W`), cognitivo (`C1`) y social (`C2`).
- **Resultado:** compara el Random Forest optimizado contra uno con hiperparámetros por defecto.

### 3️. Entrenamiento de red neuronal sin backpropagation con GWO (Optimizador de Lobo Gris)

**Objetivo:** entrenar una red neuronal (arquitectura 6 → 6 → 4 → 1) para clasificar visitantes de museos, **optimizando todos sus pesos y bias directamente con un algoritmo de enjambre**, sin usar gradientes ni retropropagación.

- **Representación:** cada lobo es un vector plano con todos los pesos/bias de la red (`W1, b1, W2, b2, W3, b3`).
- **Fitness:** entropía cruzada binaria entre la salida de la red (forward pass con `tanh` y `sigmoid`) y las etiquetas reales, minimizada.
- **Dinámica:** se identifican los 3 mejores lobos (**alfa, beta, delta**) y el resto de la manada actualiza su posición guiándose por ellos; el coeficiente `a` decrece linealmente para pasar de exploración a explotación.
- **Resultado:** accuracy en train y test de la red entrenada íntegramente por enjambre.

### 4. Clustering no supervisado con PSO

**Objetivo:** agrupar eventos sísmicos del volcán Misti en `K=3` clusters según frecuencia, duración y energía, **optimizando la posición de los centroides con PSO** en lugar del algoritmo clásico de K-Means.

- **Representación:** cada partícula codifica los `K` centroides concatenados en un vector de tamaño `K × N_DIM`.
- **Fitness:** error de cuantización (distancia promedio de cada punto a su centroide más cercano), con penalización si algún cluster queda vacío.
- **Inicialización inteligente:** los centroides iniciales se toman de puntos reales del dataset.
- **Dinámica:** PSO con inercia decreciente (`W_MAX → W_MIN`) y límite de velocidad (`V_MAX`).
- **Resultado:** se compara el *silhouette score* obtenido por PSO contra el de K-Means clásico como referencia.

---

## Requisitos

```bash
pip install numpy pandas scikit-learn
```

- Python 3.9+
- numpy
- pandas
- scikit-learn

---

## Cómo ejecutar

**Localmente:**
```bash
python "Ejemplo01 Seleccion Caracteristicas Abc.py"
python "Ejemplo02 Ajuste Hiperparametros Pso.py"
python "Ejemplo03 Entrenamiento Red Neuronal sin Backpropagation.py"
python "Ejemplo04 Agrupamiento Clustering.py"
```

> Asegúrate de que los archivos CSV (`base de datos museos_5.csv` e `IGP_CatalogoEventosSismicosVolcanMisti_2024-2025_Dataset.csv`) estén en el mismo directorio que el script.

**En Google Colab:**
Abre directamente el notebook correspondiente (`.ipynb`) de cada ejemplo y súbele el CSV correspondiente cuando se te solicite.

---

## Resumen de algoritmos usados

| Algoritmo | Sigla | Inspiración biológica | Usado para |
|---|---|---|---|
| Artificial Bee Colony | ABC | Comportamiento de forrajeo de abejas | Selección de características (binario) |
| Particle Swarm Optimization | PSO | Movimiento de bandadas/cardúmenes | Hiperparámetros y clustering (continuo) |
| Grey Wolf Optimizer | GWO | Jerarquía de caza de lobos grises | Entrenamiento de pesos de red neuronal |

Todos comparten la misma estructura conceptual: **(1)** representación de la partícula/agente, **(2)** función de aptitud (fitness), **(3)** inicialización del enjambre, **(4)** comportamiento individual del agente y **(5)** evolución iterativa hacia la mejor solución global.

---

## Datasets

- Datos de museos y eventos sísmicos de fuentes públicas (ver carpeta del repositorio: *"Dataset Publicas usadas"*).
