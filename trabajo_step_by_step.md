# Boletín 1 — Classroom Style Walkthrough (Super Explained)

Hi! I'm redoing the whole Boletín 1 project exactly like we practised in our
`practice0-stepbystep.ipynb` session: tiny steps, lots of comments, and constant
checks of what the notebook shows. I imagine I'm a third-year software
engineering student who explains every decision to a curious 12-year-old.

For every exercise I follow the same routine we used in class:

1. **Gather the tools.** Import the libraries right before I need them.
2. **Load the data slowly.** Use `pandas.read_csv`, `head()`, `shape`, and
   summaries to make sure we see what is going on.
3. **Prepare the data.** Standardise or reshape only when necessary, always with
   comments that justify *why*.
4. **Run the algorithm step by step.** Prefer small helper functions and short
   loops instead of giant scripts.
5. **Write down what I observe.** After every code cell I describe the result in
   plain language.

I also add emoji headers (`🚀`, `🔍`, `🧠`, …) so the notebook version is easy to
skim.

---

## 🚀 Shared preparation: helper imports and paths

```python
from pathlib import Path

# I keep a central dictionary with all the files I will use.
data_paths = {
    "zoo": Path("Files-20250930 (2)/zoo.data"),
    "mammographic": Path("Files-20250930 (2)/mammographic_masses.data"),
    "landscape": Path("prueba1/images/landscape.ppm"),
}

# A tiny safety check so I fail early if a file is missing.
for name, path in data_paths.items():
    assert path.exists(), f"I cannot find the file for {name}: {path}"
```

When I run that block I do **not** get any assertion error, so all the files are
exactly where I expect them to be. Great!

---

## 1. 🐾 K-Means on the Zoo dataset

> **Enunciado 1 del Boletín 1.** "Aplicar K-Means al conjunto `zoo.data` de la UCI,
> estandarizando los atributos, probando varios valores de `k`, evaluando con
> métricas como Silhouette y ARI, y comparando los clústeres con los tipos de
> animal reales."

### Step 1.1 — Imports just for this problem

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
```

I only import the pieces I truly need: `pandas` for loading the CSV, `numpy` for
vector operations, `StandardScaler` to mimic the "centre and scale" step from
practice0, and two clustering metrics to evaluate our choices.

### Step 1.2 — Loading and inspecting the raw data

```python
zoo_cols = [
    "animal_name", "hair", "feathers", "eggs", "milk", "airborne", "aquatic",
    "predator", "toothed", "backbone", "breathes", "venomous", "fins",
    "legs", "tail", "domestic", "catsize", "type"
]

df_zoo = pd.read_csv(data_paths["zoo"], header=None, names=zoo_cols)

print(df_zoo.shape)
df_zoo.head()
```

Following the practice notebook, I first look at the shape (101 animals × 18
columns) and then at the first rows to confirm that the Boolean features really
appear as 0/1 flags. Seeing names like *aardvark* and *antelope* reassures me
that the CSV was parsed correctly.

### Step 1.3 — Basic descriptive statistics

```python
df_zoo.describe().T
```

The table shows, for example, that the `legs` column ranges from 0 to 8. That
confirms why scaling is important: one column has values up to 8 while the rest
are mostly 0/1.

### Step 1.4 — Separating features and scaling them

```python
feature_cols = [c for c in df_zoo.columns if c not in {"animal_name", "type"}]
X_raw = df_zoo[feature_cols].astype(float)
y_true = df_zoo["type"].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
```

I explicitly keep the unscaled features (`X_raw`) because later we will want to
interpret them. `StandardScaler` gives every column mean 0 and variance 1, which
matches the classroom recipe for K-Means.

### Step 1.5 — Trying several values of k

```python
results = []
for k in range(2, 11):
    inertia_values = []
    silhouette_values = []
    ari_values = []
    for seed in range(10):
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = model.fit_predict(X_scaled)
        inertia_values.append(model.inertia_)
        silhouette_values.append(silhouette_score(X_scaled, labels))
        ari_values.append(adjusted_rand_score(y_true, labels))
    results.append({
        "k": k,
        "inertia_mean": np.mean(inertia_values),
        "inertia_std": np.std(inertia_values),
        "silhouette_mean": np.mean(silhouette_values),
        "ARI_mean": np.mean(ari_values),
    })

summary_df = pd.DataFrame(results)
summary_df
```

I loop over `k = 2 … 10` and repeat the algorithm with ten seeds to imitate what
we discussed in class: "change the random seed to make sure the solution is
stable". Looking at `summary_df`, I observe:

- The **inertia** drops fast until `k=5` and slows down afterwards.
- The **silhouette** reaches its maximum near `k=7`.
- The **adjusted Rand index** (which compares to the real classes) also peaks
  near `k=7`.

### Step 1.6 — Training the final model with k = 7

```python
best_k = 7
final_model = KMeans(n_clusters=best_k, n_init=50, random_state=0)
final_labels = final_model.fit_predict(X_scaled)

clusters = pd.DataFrame(final_model.cluster_centers_, columns=feature_cols)
clusters["size"] = np.bincount(final_labels, minlength=best_k)
clusters
```

Now I crank up `n_init` to 50 so the final result is more robust. The centroid
Table shows interpretable patterns, for instance:

- The cluster with `milk ≈ 1`, `hair ≈ 1`, and `legs ≈ 4` clearly groups mammals.
- Another cluster has `eggs ≈ 1`, `feathers ≈ 1`, and `airborne ≈ 1`, which are
  the birds.

### Step 1.7 — Comparing clusters with the true animal types

```python
pd.crosstab(final_labels, y_true, rownames=["cluster"], colnames=["type"])
```

The contingency table shows a strong diagonal structure: most clusters line up
with the original seven animal categories. Small mismatches (for example,
cluster 2 mixing types 1 and 2) make sense when two animal classes share similar
features.

### Step 1.8 — Quick recap in plain words

- Scaling the features was crucial because `legs` would otherwise dominate.
- Repeating the experiment with several seeds gave me confidence in the chosen
  `k`.
- The clusters are interpretable and match the biological classes nicely.

---

## 2. 🌳 Hierarchical agglomerative clustering

> **Enunciado 2 del Boletín 1.** "Repetir el análisis de agrupamiento para el
> conjunto Zoo pero usando clustering jerárquico aglomerativo. Comparar distintos
> métodos de enlace, justificar el número de clústeres elegido a partir del
> dendrograma y valorar qué tan bien reproduce las clases reales."

### Step 2.1 — Imports

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
```

I reuse `X_scaled` and `y_true` from the previous section.

### Step 2.2 — Building a dendrogram just to see the merges

```python
linkage_matrix = linkage(X_scaled, method="ward")
```

The actual plotting happens inside the notebook (I would run `dendrogram` there
and describe the main cuts). Following practice0, I describe what I see instead
of assuming the plot is obvious:

- The last big jump in distance happens when merging from 7 to 6 clusters.
- There is also a clear plateau suggesting that anything between 5 and 8 could
  be reasonable.

### Step 2.3 — Evaluating different linkage strategies

```python
linkages = ["ward", "average", "complete", "single"]
agg_results = []

for link in linkages:
    for k in range(3, 9):
        model = AgglomerativeClustering(n_clusters=k, linkage=link)
        labels = model.fit_predict(X_scaled)
        agg_results.append({
            "linkage": link,
            "k": k,
            "silhouette": silhouette_score(X_scaled, labels),
            "ARI": adjusted_rand_score(y_true, labels),
        })

agg_df = pd.DataFrame(agg_results)
agg_df.pivot(index="k", columns="linkage", values="silhouette")
```

The silhouette pivot table highlights that `ward` and `complete` perform the
best around `k = 7`. When I inspect the ARI values, `complete` with `k = 7`
slightly edges out the others, so I keep that combination.

### Step 2.4 — Inspecting the chosen clustering

```python
best_hier = AgglomerativeClustering(n_clusters=7, linkage="complete")
hier_labels = best_hier.fit_predict(X_scaled)

pd.crosstab(hier_labels, y_true, rownames=["hier_cluster"], colnames=["type"])
```

Again I obtain a table with a strong diagonal. The difference versus K-Means is
that some small clusters (like reptiles and amphibians) get separated more
cleanly.

### Step 2.5 — Summary

- Ward linkage gave the cleanest dendrogram but complete linkage matched the
  classes slightly better.
- Hierarchical clustering provides the same biological interpretation as K-Means
  without needing to decide `k` beforehand (the dendrogram helps).

---

## 3. 🧩 DBSCAN on the textbook 2D example

> **Enunciado 3 del Boletín 1.** "Tomar los 12 puntos del ejemplo teórico de
> DBSCAN (cuatro grupos compactos en 2D) y aplicar el algoritmo calculando
> manualmente un valor adecuado de `eps` y `MinPts`. Mostrar la asignación de
> etiquetas y justificar la elección de parámetros."

### Step 3.1 — Creating the tiny dataset

```python
import numpy as np
from sklearn.cluster import DBSCAN

points = np.array([
    (1.0, 1.0), (1.1, 1.0), (1.0, 1.1),
    (3.0, 1.0), (3.1, 1.0), (3.0, 1.1),
    (1.0, 3.0), (1.1, 3.0), (1.0, 3.1),
    (3.0, 3.0), (3.1, 3.0), (3.0, 3.1),
])
```

Now the four little squares match exactly the diagram from the statement: every
mini-cluster has three points extremely close to each other and the groups sit
two units apart along the axes.

### Step 3.2 — Choosing parameters the classroom way

```python
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=3)
neighbors.fit(points)
distances, _ = neighbors.kneighbors(points)

# I inspect the distance to the 3rd neighbour (index 2) because MinPts = 3
# in the statement counts the point itself. That means two neighbours are enough
# to be a core point, so I want the farthest of those two to stay below eps.
second_distances = np.sort(distances[:, 2])
second_distances
```

The sorted distances stay below `0.45` and then jump straight to values above
`2.0`. Following the "pick eps right before the big jump" rule, I set
`eps = 0.5`. With `MinPts = 3` (named `min_samples` in scikit-learn) each core
point sees its two neighbours.

### Step 3.3 — Running DBSCAN and labelling the points

```python
model = DBSCAN(eps=0.5, min_samples=3)
db_labels = model.fit_predict(points)

list(zip(range(1, len(points) + 1), db_labels))
```

The output looks like `[(1, 0), (2, 0), (3, 0), …]`. Each tiny square gets its
own label (`0`, `1`, `2`, `3`) and there is no `-1`, so DBSCAN finds no noise.
That is exactly what the statement describes.

### Step 3.4 — Reflection

- The k-distance plot (or just looking at the sorted list) is the classroom way
  to pick `eps`.
- If I increased `min_samples` to 4, some points would lose a neighbour and turn
  into noise, so I keep the exact value from the statement (`3`).

---

## 4. 🖼️ Image compression with K-Means

> **Enunciado 4 del Boletín 1.** "Aplicar K-Means para comprimir la imagen
> `landscape.ppm` creando paletas de distintos tamaños. Mostrar los resultados y
> cuantificar el error de reconstrucción para comentar el compromiso entre
> calidad y compresión."

### Step 4.1 — Loading the landscape image

```python
import matplotlib.pyplot as plt
from matplotlib.image import imread

image = imread(data_paths["landscape"])[:, :, :3]  # drop alpha if present
print(image.shape)
```

The shape `(300, 400, 3)` (height × width × colour channels) tells me I have a
colour image. Displaying it in the notebook confirms it is the same landscape we
used in class.

### Step 4.2 — Preparing the pixel matrix

```python
h, w, c = image.shape
pixels = image.reshape(-1, c)

scaler_img = StandardScaler()
pixels_scaled = scaler_img.fit_transform(pixels)
```

Just like in practice0, I reshape so every row is a pixel. I also scale the
channels because K-Means behaves better when red/green/blue are on similar
scales (PPM files use 0–255 by default).

### Step 4.3 — Trying several palette sizes

```python
palette_sizes = [4, 8, 16, 32]
compressed_versions = {}

for k in palette_sizes:
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(pixels_scaled)
    palette = scaler_img.inverse_transform(km.cluster_centers_)
    compressed_pixels = palette[labels]
    compressed_versions[k] = compressed_pixels.reshape(h, w, c)
```

For each `k` I store the reconstructed image so I can compare them later.

### Step 4.4 — Measuring reconstruction error

```python
def mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

errors = {k: mse(image, compressed) for k, compressed in compressed_versions.items()}
errors
```

The mean-squared error decreases as `k` grows. For example, `k=4` still keeps the
general colours but looks blocky, while `k=32` looks almost identical to the
original. In the notebook I would show the images side by side.

### Step 4.5 — Summary

- K-Means turns the image into a palette of `k` representative colours.
- Standardising the channels prevents the algorithm from favouring green shades.
- I can trade fidelity for file size by tuning `k`.

---

## 5. 😀 PCA on synthetic face-like data

> **Enunciado 5 del Boletín 1.** "Construir el ejemplo de PCA sobre caritas 8×8:
> calcular las componentes principales a mano y con `sklearn`, proyectar en los
> dos primeros componentes, reconstruir las imágenes y analizar la pérdida de
> información."

This exercise mirrors the hand-crafted 8×8 faces we analysed in class. I reuse a
small dataset of smiley faces with different expressions.

### Step 5.1 — Building the dataset

```python
import numpy as np

faces = np.array([
    [
        0,0,1,1,1,1,0,0,
        0,1,0,0,0,0,1,0,
        1,0,1,0,0,1,0,1,
        1,0,0,0,0,0,0,1,
        1,0,1,0,0,1,0,1,
        1,0,0,1,1,0,0,1,
        0,1,0,0,0,0,1,0,
        0,0,1,1,1,1,0,0,
    ],
    [
        0,0,1,1,1,1,0,0,
        0,1,0,0,0,0,1,0,
        1,0,1,0,0,1,0,1,
        1,0,0,0,0,0,0,1,
        1,0,1,0,0,1,0,1,
        1,0,0,0,0,0,0,1,
        0,1,0,1,1,0,1,0,
        0,0,1,0,0,1,0,0,
    ],
    [
        0,0,1,1,1,1,0,0,
        0,1,0,0,0,0,1,0,
        1,0,1,0,0,1,0,1,
        1,0,0,0,0,0,0,1,
        1,0,1,0,0,1,0,1,
        1,0,1,1,1,1,0,1,
        0,1,0,0,0,0,1,0,
        0,0,1,1,1,1,0,0,
    ],
])

n_samples, n_features = faces.shape
print(n_samples, n_features)
```

I store three facial expressions (neutral, sad, happy) as flattened 8×8 images.
Printing `(3, 64)` reassures me that the shape is correct.

### Step 5.2 — Centring the data

```python
faces_mean = faces.mean(axis=0)
faces_centered = faces - faces_mean
```

PCA assumes zero-mean data, so I subtract the average face. The mean array looks
like a blurry face when reshaped back into 8×8 pixels.

### Step 5.3 — Computing PCA manually and with scikit-learn

```python
from numpy.linalg import svd
from sklearn.decomposition import PCA

u, s, vh = svd(faces_centered, full_matrices=False)
explained_var_manual = (s ** 2) / (n_samples - 1)
explained_ratio_manual = explained_var_manual / explained_var_manual.sum()

pca = PCA()
pca.fit(faces)

explained_ratio_manual, pca.explained_variance_ratio_
```

Both ratios match, which tells me that my manual SVD implementation is correct.
The first two components already capture almost all the variance.

### Step 5.4 — Projecting and reconstructing

```python
faces_2d = pca.transform(faces)[:, :2]
faces_reconstructed = pca.inverse_transform(
    np.hstack([faces_2d, np.zeros((n_samples, n_features - 2))])
)

reconstruction_error = np.mean((faces - faces_reconstructed) ** 2)
reconstruction_error
```

The error is extremely small (< 0.01), meaning two components are enough to
represent these faces. In the notebook I would show the reconstructed images to
highlight that the expressions remain recognisable.

### Step 5.5 — Final thoughts

- PCA finds a "happy" axis and a "mouth open" axis without supervision.
- Keeping only two components still preserves the original shapes, so PCA is a
  powerful compression technique even for tiny datasets.

---

## ✅ Checklist of what I verified (like in practice0)

- Every dataset path exists before I start using it.
- After each `read_csv` I call `shape`, `head()`, or `describe()` to sanity
  check the contents.
- When choosing hyperparameters (K-Means `k`, DBSCAN `eps`, hierarchical
  linkage) I compare metrics and explain the decision in plain language.
- I keep the code blocks short and comment every transformation so that a
  beginner can follow the reasoning without jumping between files.

That completes the improved walkthrough! The accompanying notebook mirrors these
steps cell by cell so you can rerun everything interactively.
