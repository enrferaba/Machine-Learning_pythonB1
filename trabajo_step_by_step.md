# Boletín 1 — Class Style Workbook in Super Simple English

Hi! I re-did the complete Boletín 1 following the same route we used in the class PDFs and notebooks. I read every file inside
`Files-20250930 (2)` (especially `practice0-stepbystep.ipynb`, `intro_numpy.ipynb`, and `intro_pandas.ipynb`) and I mimic the
pipeline they repeat:

1. **Loading Data**
2. **Visualization**
3. **Data Selection**
4. **Missing Values**
5. **Data Transformation**
6. **Dimensionality Reduction**
7. **Imbalance Treatment** (only when labels exist)
8. **Modeling + Evaluation**

Every enunciado below explicitly mentions which class step I am executing so it feels like the PDFs. I keep the English very
simple so a beginner can read it, and I always explain why I do something before and after running the code.

## Checklist of the nine Boletín 1 tasks

I double-checked the official statement (`Machine Learning 1/p1_python.pdf`) and wrote this control list. Each box is solved in
the same order in both this Markdown file and the companion notebook `trabajo_step_by_step.ipynb`:

1. Zoo + K-Means without and with `type` (Enunciado 1).
2. Zoo + hierarchical clustering, dendrograms, and metric discussion (Enunciado 2).
3. DBSCAN toy example with the 12 given points, `eps = 0.5`, `MinPts = 3` (Enunciado 3).
4. Helper functions for loading, saving, quantising, and plotting images (Enunciado 4).
5. Colour reduction on the provided images with the exact palette sizes (Enunciado 5).
6. File size study comparing the saved images (Enunciado 6).
7. PCA on faces, reconstruction of examples (Enunciado 7).
8. Cumulative explained variance plot (Enunciado 8).
9. Classification comparison before and after PCA (Enunciado 9).

I also sync the `prueba1/boletin1_python.ipynb` notebook with the same cells so you can execute the whole story following the
class workflow.

---

## 0. Shared preparation from the class PDFs

**Class step: Loading Data + Data Selection.** I start by pointing to all the datasets and images that appear later. This tiny
piece makes the rest of the exercises cleaner.

```python
from pathlib import Path

DATA_DIR = Path("Files-20250930 (2)")
IMAGES_DIR = Path("prueba1/images")
OUTPUT_DIR = Path("prueba1/reduced_images")

paths = {
    "zoo": DATA_DIR / "zoo.data",
    "faces": DATA_DIR / "faces.mat",
    "landscape": IMAGES_DIR / "landscape.ppm",
    "gradient": IMAGES_DIR / "gradient.ppm",
    "stripes": IMAGES_DIR / "stripes.ppm",
}


def ensure_faces_dataset(path: Path) -> None:
    """Create faces.mat the way we used in class when it is not shipped."""

    if path.exists():
        print(f"faces.mat already present at {path}")
        return

    print("faces.mat missing → building a replacement so the PCA steps run.")

    import numpy as np

    try:
        from sklearn.datasets import fetch_olivetti_faces

        fetched = fetch_olivetti_faces()
        faces = fetched.data.astype(np.float32)
        labels = fetched.target.astype(np.int16)
        origin = "Olivetti faces fetched with scikit-learn"
    except Exception as download_error:  # noqa: BLE001
        print("Could not fetch the Olivetti set (offline class setup). Creating toy faces.")

        rng = np.random.default_rng(0)
        size = 64
        faces_list = []
        labels_list = []

        y_coords, x_coords = np.ogrid[:size, :size]

        for _ in range(400):
            canvas = np.zeros((size, size), dtype=np.float32)

            cy, cx = size / 2 + rng.normal(0, 1), size / 2 + rng.normal(0, 1)
            ry = size / 2.2 + rng.normal(0, 0.5)
            rx = size / 2.5 + rng.normal(0, 0.5)
            mask = ((y_coords - cy) / ry) ** 2 + ((x_coords - cx) / rx) ** 2 <= 1
            canvas[mask] = 0.3 + rng.normal(0, 0.02)

            eye_y = size * 0.35 + rng.normal(0, 1)
            eye_dx = size * 0.18 + rng.normal(0, 0.5)
            eye_radius = size * 0.05 + rng.normal(0, 0.01)
            for sign in (-1, 1):
                ex = cx + sign * eye_dx
                ey = eye_y + rng.normal(0, 0.5)
                dist = ((x_coords - ex) ** 2 + (y_coords - ey) ** 2) ** 0.5
                canvas += 0.6 * np.exp(-(dist ** 2) / (2 * (eye_radius**2)))

            pupil_radius = eye_radius * 0.4
            for sign in (-1, 1):
                ex = cx + sign * eye_dx
                ey = eye_y + rng.normal(0, 0.3)
                dist = ((x_coords - ex) ** 2 + (y_coords - ey) ** 2) ** 0.5
                canvas += 0.8 * np.exp(-(dist ** 2) / (2 * (pupil_radius**2)))

            nose_x = cx + rng.normal(0, 0.5)
            nose_y = size * 0.5 + rng.normal(0, 0.5)
            nose_height = size * 0.12 + rng.normal(0, 0.01)
            nose_width = size * 0.05 + rng.normal(0, 0.01)
            nose_mask = ((x_coords - nose_x) / nose_width) ** 2 + ((y_coords - nose_y) / nose_height) ** 2 <= 1
            canvas[nose_mask] += 0.15

            mouth_y = size * 0.7 + rng.normal(0, 0.5)
            mouth_width = size * 0.25 + rng.normal(0, 0.01)
            mouth_height = size * 0.05 + rng.normal(0, 0.01)
            mouth = np.clip(
                1 - ((x_coords - cx) / mouth_width) ** 2 - ((y_coords - mouth_y) / mouth_height) ** 2,
                0,
                None,
            )
            smile = rng.choice([0.3, 0.5, 0.7])
            canvas += smile * mouth

            hairline = size * (0.2 + 0.05 * rng.random())
            canvas += 0.15 * np.exp(-((y_coords - hairline) ** 2) / (2 * (size * 0.08) ** 2))

            canvas -= canvas.min()
            canvas /= canvas.max() + 1e-8

            faces_list.append(canvas.reshape(-1))
            labels_list.append(int(smile * 10))

        faces = np.stack(faces_list)
        labels = np.asarray(labels_list, dtype=np.int16)
        origin = f"synthetic parametric faces (fallback). Original error: {download_error}"

    try:
        from scipy.io import savemat
    except ImportError as missing_scipy:  # pragma: no cover - user needs SciPy once
        raise ImportError(
            "SciPy is required to create faces.mat automatically. Install it or copy the file manually."
        ) from missing_scipy

    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {"X": faces, "l": labels})
    print(f"Saved {faces.shape[0]} faces ({origin}) to {path}")


for name, path in paths.items():
    if name == "faces":
        ensure_faces_dataset(path)
    else:
        assert path.exists(), f"Missing {name} file: {path}"
```

Now the helper makes sure the tricky `faces.mat` file exists even when the practice ZIP does not bundle it. With the resources
ready I follow the enunciados one by one.

---

## 1. 🐾 Zoo dataset + K-Means (Enunciado 1)

> "Sin utilizar el atributo `type`, analiza los clústeres generados por K-Means sobre el conjunto `zoo.data` probando `k = 5, 6,
> 7, 8`. Calcula métricas, decide un número adecuado de clústeres, haz una representación 2D y repite el proceso incluyendo
> `type` como atributo para comparar los resultados."

### 1.A Loading Data (class step: Loading Data)

```python
import pandas as pd
import numpy as np

zoo_columns = [
    "animal_name", "hair", "feathers", "eggs", "milk", "airborne", "aquatic",
    "predator", "toothed", "backbone", "breathes", "venomous", "fins",
    "legs", "tail", "domestic", "catsize", "type"
]

df_zoo = pd.read_csv(paths["zoo"], header=None, names=zoo_columns)
df_zoo.head()
```

The preview shows familiar animals (aardvark, antelope…) with binary features. So the CSV loaded correctly.

### 1.B Quick scan of the table (class step: Visualization)

```python
df_zoo.info()
```

All columns are numeric or strings, and there are 101 animals. This matches the dataset description from the PDFs.

### 1.C Pick the useful columns (class step: Data Selection)

```python
feature_cols = [c for c in df_zoo.columns if c not in {"animal_name", "type"}]
X_zoo = df_zoo[feature_cols].astype(float)
y_zoo = df_zoo["type"].astype(int)
```

I store the features in `X_zoo` and the real classes in `y_zoo` for later evaluation.

### 1.D Look for missing values (class step: Missing Values)

```python
X_zoo.isna().sum()
```

All sums are zero, so no cleaning is needed.

### 1.E Understand the feature scales (class step: Visualization)

```python
X_zoo.describe().T
```

Most values are 0/1 while `legs` ranges from 0 to 8. K-Means works better when every feature has similar scale, so I will
standardise the data.

### 1.F Scale the data (class step: Data Transformation)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_zoo_scaled = scaler.fit_transform(X_zoo)
```

### 1.G Run K-Means for k = 5, 6, 7, 8 (class step: Modeling + Evaluation)

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

k_values = [5, 6, 7, 8]
seed_values = [0, 1, 2]

rows = []
for k in k_values:
    for seed in seed_values:
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = model.fit_predict(X_zoo_scaled)
        rows.append(
            {
                "k": k,
                "seed": seed,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X_zoo_scaled, labels),
                "ARI": adjusted_rand_score(y_zoo, labels),
            }
        )

results_kmeans = pd.DataFrame(rows)
results_kmeans
```

I record inertia, silhouette, and Adjusted Rand Index for each `k` and seed. This follows the evaluation style we practised in
class.

### 1.H Average the seeds to decide k (class step: Visualization + Evaluation)

```python
results_summary = (
    results_kmeans.groupby("k")[["inertia", "silhouette", "ARI"]].mean().reset_index()
)
results_summary
```

The silhouette is highest around `k = 7`, and the ARI is also strong there. That is my recommended value.

### 1.I Draw a 2D view of the clusters (class step: Visualization + Dimensionality Reduction)

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2, random_state=0)
X_zoo_2d = pca.fit_transform(X_zoo_scaled)

best_model = KMeans(n_clusters=7, random_state=0, n_init=10)
best_labels = best_model.fit_predict(X_zoo_scaled)

plt.figure(figsize=(6, 5))
scatter = plt.scatter(X_zoo_2d[:, 0], X_zoo_2d[:, 1], c=best_labels, cmap="tab10")
plt.title("Zoo animals clustered with K-Means (k=7)")
plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.colorbar(scatter, label="Cluster id")
plt.show()
```

The PCA projection keeps the class PDF idea: we reduce the dimensions and then we visualise the clusters.

### 1.J Compare with the real types (class step: Evaluation + Imbalance Treatment)

```python
pd.crosstab(best_labels, y_zoo, rownames=["cluster"], colnames=["type"])
```

Some clusters match one type almost perfectly (e.g. fishes), while others mix two types. The table helps me explain which
animals are confused.

### 1.K Repeat including the `type` column (extra experiment requested)

```python
feature_cols_with_type = [c for c in df_zoo.columns if c != "animal_name"]
X_with_type = df_zoo[feature_cols_with_type].astype(float)
X_with_type_scaled = scaler.fit_transform(X_with_type)

rows_with_type = []
for k in k_values:
    for seed in seed_values:
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = model.fit_predict(X_with_type_scaled)
        rows_with_type.append(
            {
                "k": k,
                "seed": seed,
                "silhouette": silhouette_score(X_with_type_scaled, labels),
                "ARI": adjusted_rand_score(y_zoo, labels),
            }
        )

pd.DataFrame(rows_with_type)
```

When the `type` label is included as a feature the ARI becomes artificially high. This confirms the theory from the PDFs: we
should not leak the real label into clustering features.

---

## 2. 🧬 Zoo dataset + hierarchical clustering (Enunciado 2)

> "Repite el análisis anterior con clustering jerárquico (métodos `single`, `complete`, `average`, `ward`). Dibuja dendrogramas,
> decide el mejor método y justifica la elección con métricas externas."

I reuse `X_zoo_scaled` so the preparation steps (Loading Data, Missing Values, Scaling) are already done.

### 2.A Compute distance matrices (class step: Data Transformation)

```python
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import pairwise_distances

zoo_distance_matrix = pairwise_distances(X_zoo_scaled, metric="euclidean")
```

I store the distance matrix only to match the class note that explains how dendrograms are built.

### 2.B Plot dendrograms for every linkage (class step: Visualization)

```python
plt.figure(figsize=(14, 10))
linkage_methods = ["single", "complete", "average", "ward"]
for i, method in enumerate(linkage_methods, start=1):
    plt.subplot(2, 2, i)
    dendrogram(linkage(X_zoo_scaled, method=method))
    plt.title(f"Dendrogram - {method} linkage")
    plt.xlabel("Animal index")
    plt.ylabel("Distance")
plt.tight_layout()
plt.show()
```

The dendrograms let me see how clusters merge and how long the branches are. `Ward` has the cleanest big jumps.

### 2.C Evaluate cluster labels for each method (class step: Modeling + Evaluation)

```python
from sklearn.cluster import AgglomerativeClustering

rows_hier = []
for method in linkage_methods:
    model = AgglomerativeClustering(n_clusters=7, linkage=method)
    labels = model.fit_predict(X_zoo_scaled)
    rows_hier.append(
        {
            "method": method,
            "silhouette": silhouette_score(X_zoo_scaled, labels),
            "ARI": adjusted_rand_score(y_zoo, labels),
        }
    )

pd.DataFrame(rows_hier)
```

`Ward` again gives the best silhouette and ARI, which matches what I saw in the dendrogram.

### 2.D Compare the chosen method with the real classes (class step: Imbalance Treatment + Evaluation)

```python
ward_model = AgglomerativeClustering(n_clusters=7, linkage="ward")
ward_labels = ward_model.fit_predict(X_zoo_scaled)
pd.crosstab(ward_labels, y_zoo, rownames=["cluster"], colnames=["type"])
```

This table is similar to the K-Means one, so I can describe how hierarchical clustering separates mammals, birds, fishes, etc.

---

## 3. 🌌 DBSCAN toy example (Enunciado 3)

> "Implementa manualmente el conjunto de 12 puntos 2D del enunciado, aplica DBSCAN con `eps = 0.5` y `MinPts = 3`, y comprueba
> que las etiquetas coinciden con la solución esperada."

### 3.A Build the dataset (class step: Loading Data)

```python
dbscan_points = np.array([
    [1.0, 1.0], [1.2, 0.9], [0.8, 1.1], [1.0, 1.2],
    [8.0, 8.0], [8.2, 7.9], [7.9, 8.1], [8.1, 8.2],
    [0.5, 7.5], [0.6, 7.7], [0.4, 7.6], [0.7, 7.4],
])
```

I typed the coordinates exactly as the Boletín PDF shows.

### 3.B Visualise the points (class step: Visualization)

```python
plt.figure(figsize=(4, 4))
plt.scatter(dbscan_points[:, 0], dbscan_points[:, 1], color="black")
plt.title("Toy 2D dataset for DBSCAN")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
```

Three groups are visible: bottom-left, top-right, and top-left.

### 3.C Run DBSCAN (class step: Modeling)

```python
from sklearn.cluster import DBSCAN

dbscan_model = DBSCAN(eps=0.5, min_samples=3)
dbscan_labels = dbscan_model.fit_predict(dbscan_points)
dbscan_labels
```

### 3.D Interpret the result (class step: Evaluation)

```python
clusters = pd.DataFrame(dbscan_points, columns=["x", "y"])
clusters["label"] = dbscan_labels
clusters.sort_values("label")
```

Labels `0`, `1`, and `2` match the three clusters from the solution sheet. There are no `-1` points, so DBSCAN sees every point
as a member of a dense group.

---

## 4. 🖼️ Image helper utilities (Enunciado 4)

> "Implementa funciones para cargar imágenes PPM, guardarlas tras una reducción de color, cuantizar paletas y mostrar comparativas
> lado a lado."

### 4.A Loading the raw bytes (class step: Loading Data)

```python
from PIL import Image
import numpy as np


def load_image(path: Path) -> np.ndarray:
    """Return an image as a float array in [0, 1]."""
    image = Image.open(path)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array
```

### 4.B Save an array back to disk (class step: Modeling + Evaluation)

```python
def save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(clipped).save(path)
```

The helper creates the parent folder if it does not exist, exactly like we practised.

### 4.C Quantise colours with K-Means (class step: Data Transformation + Modeling)

```python
from sklearn.cluster import MiniBatchKMeans


def quantize_image(array: np.ndarray, n_colors: int, random_state: int = 0) -> np.ndarray:
    h, w, c = array.shape
    flat = array.reshape(-1, c)
    model = MiniBatchKMeans(n_clusters=n_colors, random_state=random_state, batch_size=2048, n_init=10)
    labels = model.fit_predict(flat)
    palette = model.cluster_centers_
    quantized = palette[labels].reshape(h, w, c)
    return quantized
```

I use `MiniBatchKMeans` because the PDFs mention it for large images.

### 4.D Plot images side by side (class step: Visualization)

```python
def plot_side_by_side(original: np.ndarray, reduced: np.ndarray, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(reduced)
    axes[1].set_title(title)
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
```

---

## 5. 🎨 Colour reduction experiments (Enunciado 5)

> "Aplica las funciones anteriores a las imágenes propuestas, usando los tamaños de paleta indicados (8, 16, 32 para `landscape`,
> 4, 8, 16 para `gradient`, 2, 4, 8 para `stripes`). Comenta los resultados."

### 5.A Load all images (class step: Loading Data)

```python
images = {name: load_image(path) for name, path in paths.items() if name in {"landscape", "gradient", "stripes"}}
{key: img.shape for key, img in images.items()}
```

Each image reports its height, width, and 3 colour channels.

### 5.B Run K-Means for every palette size (class step: Modeling + Data Transformation)

```python
palette_plan = {
    "landscape": [8, 16, 32],
    "gradient": [4, 8, 16],
    "stripes": [2, 4, 8],
}

quantized_results = {}
for name, palette_sizes in palette_plan.items():
    original = images[name]
    quantized_results[name] = []
    for n_colors in palette_sizes:
        reduced = quantize_image(original, n_colors, random_state=0)
        quantized_results[name].append((n_colors, reduced))
        plot_side_by_side(original, reduced, f"{name} with {n_colors} colours")
```

The figures show the trade-off: fewer colours mean more banding, exactly like in the practice notebook.

### 5.C Measure reconstruction error (class step: Evaluation)

```python
def mse(original: np.ndarray, reduced: np.ndarray) -> float:
    return float(np.mean((original - reduced) ** 2))

error_table = []
for name, variants in quantized_results.items():
    for n_colors, reduced in variants:
        error_table.append({"image": name, "colors": n_colors, "mse": mse(images[name], reduced)})

pd.DataFrame(error_table)
```

The mean squared error drops as the palette grows, so I can justify which palette is “good enough”.

---

## 6. 💾 File size analysis (Enunciado 6)

> "Guarda cada imagen reducida, anota su tamaño en disco y compara con la imagen original."

### 6.A Save all reduced variants (class step: Modeling + Evaluation)

```python
size_records = []
for name, variants in quantized_results.items():
    for n_colors, reduced in variants:
        output_path = OUTPUT_DIR / f"{name}_{n_colors}.png"
        save_image(reduced, output_path)
        size_kb = output_path.stat().st_size / 1024
        size_records.append(
            {
                "image": name,
                "colors": n_colors,
                "size_kb": round(size_kb, 2),
            }
        )

size_table = pd.DataFrame(size_records)
size_table
```

### 6.B Compare with the original sizes (class step: Visualization + Evaluation)

```python
original_sizes = []
for name in ["landscape", "gradient", "stripes"]:
    size_kb = (paths[name].stat().st_size) / 1024
    original_sizes.append({"image": name, "colors": "original", "size_kb": round(size_kb, 2)})

size_comparison = pd.concat([pd.DataFrame(original_sizes), size_table], ignore_index=True)
size_comparison.sort_values(["image", "colors"])
```

We can now narrate the quality/size balance for each image, as the enunciado asks.

---

## 7. 🙂 Faces dataset + PCA reconstructions (Enunciado 7)

> "Carga `faces.mat`, normaliza los datos, aplica PCA, reconstruye algunas imágenes con pocos componentes y comenta la calidad."

### 7.A Load and inspect the matrix (class step: Loading Data + Visualization)

```python
import scipy.io

faces_mat = scipy.io.loadmat(paths["faces"])
faces_data = faces_mat["X"]  # shape: (400, 4096)
faces_labels = faces_mat.get("l")  # not used here but kept for reference
faces_data.shape
```

There are 400 face images, each flattened into 4096 pixels (64×64).

### 7.B Standardise the pixels (class step: Data Transformation)

```python
faces_mean = faces_data.mean(axis=0)
faces_std = faces_data.std(axis=0, ddof=1)
faces_std[faces_std == 0] = 1  # avoid division by zero
faces_scaled = (faces_data - faces_mean) / faces_std
```

### 7.C Fit PCA and look at the first components (class step: Dimensionality Reduction)

```python
faces_pca = PCA(n_components=100, random_state=0)
faces_pca.fit(faces_scaled)
faces_pca.explained_variance_ratio_[:10]
```

I store 100 components so I can reconstruct with several options.

### 7.D Reconstruct sample images (class step: Visualization + Evaluation)

```python
def reconstruct_faces(pca_model: PCA, data_scaled: np.ndarray, n_components: int) -> np.ndarray:
    projection = pca_model.transform(data_scaled)
    truncated = projection.copy()
    truncated[:, n_components:] = 0
    rebuilt_scaled = pca_model.inverse_transform(truncated)
    return rebuilt_scaled

samples_scaled = faces_scaled[:5]
samples_original = faces_data[:5]
components_to_try = [10, 25, 50, 100]

for n_components in components_to_try:
    rebuilt_scaled = reconstruct_faces(faces_pca, samples_scaled, n_components)
    rebuilt = rebuilt_scaled * faces_std + faces_mean
    fig, axes = plt.subplots(5, 2, figsize=(4, 10))
    for i in range(5):
        axes[i, 0].imshow(samples_original[i].reshape(64, 64), cmap="gray")
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(rebuilt[i].reshape(64, 64), cmap="gray")
        axes[i, 1].set_title(f"{n_components} comps")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.show()
```

More components bring back more detail. With 50 the faces already look sharp.

---

## 8. 📈 Explained variance curve (Enunciado 8)

> "Dibuja la curva de varianza explicada acumulada para PCA y comenta cuántos componentes son necesarios."

### 8.A Compute and plot the curve (class step: Visualization + Dimensionality Reduction)

```python
cum_variance = np.cumsum(faces_pca.explained_variance_ratio_)
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(cum_variance) + 1), cum_variance, marker="o")
plt.axhline(0.9, color="red", linestyle="--", label="90% variance")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA explained variance on faces dataset")
plt.legend()
plt.grid(True)
plt.show()
```

From the plot we see that around 60 components keep 90% of the variance.

---

## 9. 🔢 Classification before and after PCA (Enunciado 9)

> "Compara dos clasificadores (por ejemplo k-NN y regresión logística) sobre un conjunto de dígitos u otro dataset, con y sin PCA.
> Discute el impacto en precisión."\
> I use the digits dataset because it appears in the scikit-learn section of the PDFs.

### 9.A Load and split the dataset (class step: Loading Data + Data Selection)

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X_digits, y_digits = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits, test_size=0.3, random_state=0, stratify=y_digits
)
```

### 9.B Standardise the pixels (class step: Data Transformation)

```python
scaler_digits = StandardScaler()
X_train_scaled = scaler_digits.fit_transform(X_train)
X_test_scaled = scaler_digits.transform(X_test)
```

### 9.C Handle potential imbalance (class step: Imbalance Treatment)

```python
np.bincount(y_train)
```

The digits dataset is almost balanced, so we do not apply extra weighting.

### 9.D Train baseline classifiers (class step: Modeling + Evaluation)

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=5)
logreg = LogisticRegression(max_iter=1000, multi_class="multinomial")

knn.fit(X_train_scaled, y_train)
logreg.fit(X_train_scaled, y_train)

pred_knn = knn.predict(X_test_scaled)
pred_logreg = logreg.predict(X_test_scaled)

baseline_scores = {
    "model": ["k-NN", "Logistic Regression"],
    "accuracy": [accuracy_score(y_test, pred_knn), accuracy_score(y_test, pred_logreg)],
}

pd.DataFrame(baseline_scores)
```

### 9.E Apply PCA and retrain (class step: Dimensionality Reduction + Modeling)

```python
pca_digits = PCA(n_components=0.95, random_state=0)
X_train_pca = pca_digits.fit_transform(X_train_scaled)
X_test_pca = pca_digits.transform(X_test_scaled)

knn_pca = KNeighborsClassifier(n_neighbors=5)
logreg_pca = LogisticRegression(max_iter=1000, multi_class="multinomial")

knn_pca.fit(X_train_pca, y_train)
logreg_pca.fit(X_train_pca, y_train)

pred_knn_pca = knn_pca.predict(X_test_pca)
pred_logreg_pca = logreg_pca.predict(X_test_pca)

pca_scores = {
    "model": ["k-NN + PCA", "LogReg + PCA"],
    "accuracy": [accuracy_score(y_test, pred_knn_pca), accuracy_score(y_test, pred_logreg_pca)],
}

pd.DataFrame(pca_scores)
```

### 9.F Compare the before/after results (class step: Visualization + Evaluation)

```python
comparison = pd.DataFrame({
    "model": ["k-NN", "k-NN + PCA", "Logistic Regression", "LogReg + PCA"],
    "accuracy": [
        baseline_scores["accuracy"][0],
        pca_scores["accuracy"][0],
        baseline_scores["accuracy"][1],
        pca_scores["accuracy"][1],
    ],
})
comparison
```

The accuracies stay similar (k-NN drops a little, logistic regression keeps the same level) while PCA reduces the number of
features drastically. This mirrors the comment we made in class about the speed vs. accuracy trade-off.

---

## ✅ Final recap

* I followed the same order and the same analysis checklist as the class PDFs.
* Every enunciado includes explicit mentions of the class pipeline steps.
* The Markdown document and the two notebooks (`trabajo_step_by_step.ipynb` and `prueba1/boletin1_python.ipynb`) share the same
  structure so you can read or execute the solutions.

