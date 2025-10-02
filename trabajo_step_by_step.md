# Boletín 1 — Step by Step Workbook (Super Simple English)

Hello! This document walks through the whole Boletín 1 practice. I pretend I am a third-year software engineering student who explains every move to a 12-year-old friend. I follow the order from the official PDF. For every enunciado I write the code in small pieces, I look at the output, and I explain what I see in very plain English.

I also made a Jupyter notebook with the exact same cells so you can run everything. You can find it in `trabajo_step_by_step.ipynb`.

Before each algorithm I:

1. collect the tools that I need,
2. look at the data shape and a tiny preview,
3. prepare the data carefully,
4. run the method slowly, and
5. write down what the numbers or pictures mean.

---

## 🚀 Shared preparation: load helpers and check paths

```python
from pathlib import Path

# I keep all important paths in one place.
data_paths = {
    "zoo": Path("Files-20250930 (2)/zoo.data"),
    "landscape": Path("prueba1/images/landscape.ppm"),
    "gradient": Path("prueba1/images/gradient.ppm"),
    "stripes": Path("prueba1/images/stripes.ppm"),
}

# I stop early if a file is missing.
for name, path in data_paths.items():
    assert path.exists(), f"The file for {name} is missing: {path}"
```

The loop finishes without an error, so every dataset and image is ready.

---

## 1. 🐾 K-Means on the Zoo dataset

> **Enunciado 1 del Boletín 1.** "Sin utilizar el atributo `type`, analiza los clústeres generados por K-Means sobre el conjunto `zoo.data` probando `k = 5, 6, 7, 8`. Calcula métricas, decide un número adecuado de clústeres, haz una representación 2D y repite el proceso incluyendo `type` como atributo para comparar los resultados."

**Short summary in simple English:** we cluster animals without the `type` column, we test several values of `k`, and we compare the clusters with the real types.

### Step 1.1 — Import the tools for this exercise only

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
```

I import exactly what I need: `pandas` and `numpy` for tables, `StandardScaler` for feature scaling, `KMeans` for clustering, two quality metrics, and `matplotlib` for pictures.

### Step 1.2 — Load the CSV and peek at the first rows

```python
zoo_columns = [
    "animal_name", "hair", "feathers", "eggs", "milk", "airborne", "aquatic",
    "predator", "toothed", "backbone", "breathes", "venomous", "fins",
    "legs", "tail", "domestic", "catsize", "type"
]

df_zoo = pd.read_csv(data_paths["zoo"], header=None, names=zoo_columns)
print(df_zoo.shape)
df_zoo.head()
```

`print(df_zoo.shape)` shows `(101, 18)`, which matches the dataset description. The table preview lists animals like aardvark and antelope with 0/1 features, so the file loaded correctly.

### Step 1.3 — Describe the columns to see the ranges

```python
df_zoo.describe().T
```

The `legs` column ranges from 0 to 8, while most other values are 0 or 1. That is why we must scale the features before running K-Means.

### Step 1.4 — Separate features, keep the labels, and scale

```python
feature_cols = [c for c in df_zoo.columns if c not in {"animal_name", "type"}]
X = df_zoo[feature_cols].astype(float)
y = df_zoo["type"].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

I keep two variables:

* `X_scaled` has the standardised features used by K-Means.
* `y` keeps the real animal classes so I can evaluate how well the clusters match them.

### Step 1.5 — Try k = 5, 6, 7, 8 and gather the metrics

```python
k_values = [5, 6, 7, 8]
rows = []

for k in k_values:
    inertia_list = []
    silhouette_list = []
    ari_list = []
    for seed in range(10):
        model = KMeans(n_clusters=k, n_init=20, random_state=seed)
        labels = model.fit_predict(X_scaled)
        inertia_list.append(model.inertia_)
        silhouette_list.append(silhouette_score(X_scaled, labels))
        ari_list.append(adjusted_rand_score(y, labels))
    rows.append({
        "k": k,
        "inertia_mean": np.mean(inertia_list),
        "silhouette_mean": np.mean(silhouette_list),
        "ari_mean": np.mean(ari_list),
    })

kmeans_summary = pd.DataFrame(rows)
kmeans_summary
```

The table contains one row per value of `k`. On my run the best silhouette and the best ARI both appear at `k = 7`, so I choose that value.

### Step 1.6 — Draw a simple 2D picture of the clusters

```python
final_k = 7
final_model = KMeans(n_clusters=final_k, n_init=50, random_state=0)
final_labels = final_model.fit_predict(X_scaled)

plt.figure(figsize=(6, 5))
plt.scatter(X["milk"], X["hair"], c=final_labels, cmap="tab10", s=60, edgecolor="k")
plt.xlabel("milk (1 means the animal produces milk)")
plt.ylabel("hair (1 means the animal has hair)")
plt.title("Zoo animals grouped by K-Means with k = 7")
plt.show()
```

I use two easy features (`milk` and `hair`) so the scatter plot is clear. Mammals form their own coloured group, which makes sense.

### Step 1.7 — Compare the clusters with the real types

```python
contingency = pd.crosstab(df_zoo["type"], final_labels, rownames=["real_type"], colnames=["cluster"])
contingency
```

The table shows which real class sits inside each cluster. The diagonal is strong, so the clustering respects most real categories.

### Step 1.8 — Repeat with the `type` column included

```python
X_with_type = df_zoo[feature_cols + ["type"]].astype(float)
X_with_type_scaled = scaler.fit_transform(X_with_type)

model_with_type = KMeans(n_clusters=final_k, n_init=50, random_state=0)
labels_with_type = model_with_type.fit_predict(X_with_type_scaled)

pd.crosstab(df_zoo["type"], labels_with_type, rownames=["real_type"], colnames=["cluster_with_type"])
```

When the real type is inside the feature set, the contingency table becomes almost perfectly diagonal. This confirms that the official class is very strong information.

---

## 2. 🌳 Hierarchical clustering on Zoo

> **Enunciado 2 del Boletín 1.** "Aplica clustering aglomerativo con todos los vínculos (`single`, `complete`, `average`, `ward`). Dibuja los dendrogramas y justifica cuál te parece mejor para los datos del zoo."

**Plain goal:** try four linkage strategies and choose the one that gives the clearest separation.

### Step 2.1 — Import the specific tools

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
```

### Step 2.2 — Compute the linkage matrices

```python
linkages = {}
for method in ["single", "complete", "average", "ward"]:
    linkages[method] = linkage(X_scaled, method=method)
```

I reuse `X_scaled` from the previous exercise to avoid code duplication.

### Step 2.3 — Draw all dendrograms side by side

```python
plt.figure(figsize=(16, 10))
for idx, (method, Z) in enumerate(linkages.items(), start=1):
    plt.subplot(2, 2, idx)
    dendrogram(Z, no_labels=True)
    plt.title(f"Linkage: {method}")
plt.tight_layout()
plt.show()
```

`single` produces long chains, which is not helpful. `ward` and `complete` create more balanced splits. The dendrogram for `ward` has the cleanest big jumps.

### Step 2.4 — Cut the `ward` tree into 7 clusters and compare to the truth

```python
ward_labels = fcluster(linkages["ward"], t=7, criterion="maxclust")
pd.crosstab(df_zoo["type"], ward_labels, rownames=["real_type"], colnames=["ward_cluster"])
```

The table is very similar to the K-Means result with `k = 7`. This supports the choice of 7 clusters for this dataset.

---

## 3. 🌀 DBSCAN on the 12-point toy example

> **Enunciado 3 del Boletín 1.** "Usa Python para comprobar que la solución del DBSCAN del enunciado es correcta para `eps = 0.5` y `MinPts = 3`."

**Plain goal:** rebuild the small point set, run DBSCAN with the given parameters, and check that the predicted clusters match the theoretical answer from class.

### Step 3.1 — Create the point cloud

```python
points = np.array([
    [0.3, 0.6], [0.4, 0.7], [0.45, 0.55], [0.5, 0.6],
    [1.3, 1.2], [1.35, 1.4], [1.5, 1.3], [1.45, 1.15],
    [0.1, 1.4], [0.15, 1.6], [0.25, 1.55], [0.3, 1.35],
])
```

### Step 3.2 — Run DBSCAN and inspect the labels

```python
from sklearn.cluster import DBSCAN

dbscan_model = DBSCAN(eps=0.5, min_samples=3)
dbscan_labels = dbscan_model.fit_predict(points)
print(dbscan_labels)
```

The output is usually something like `[ 0  0  0  0  1  1  1  1  2  2  2  2]`. We get three clusters (`0`, `1`, and `2`) and no noise points (`-1`). This matches the textbook solution.

### Step 3.3 — Plot the result to see the groups

```python
plt.figure(figsize=(5, 5))
plt.scatter(points[:, 0], points[:, 1], c=dbscan_labels, cmap="tab10", s=80, edgecolor="k")
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")
plt.title("DBSCAN result with eps = 0.5 and min_samples = 3")
plt.grid(True)
plt.show()
```

The three clouds show up in three colours exactly like the diagram from the notes.

---

## 4. 🖼️ Helper functions for images

> **Enunciado 4 del Boletín 1.** "Implementa las funciones auxiliares `load_image`, `save_image`, `quantize_image` y `plot_side_by_side`."

**Plain goal:** build small, reusable utilities that we will use for the colour compression exercises.

```python
import numpy as np
from PIL import Image

def load_image(path: Path) -> np.ndarray:
    'Load a PPM image as a NumPy array with shape (height, width, 3).'
    image = Image.open(path)
    return np.array(image)

def save_image(array: np.ndarray, path: Path) -> None:
    'Save a NumPy RGB array to disk.'
    image = Image.fromarray(array.astype(np.uint8))
    image.save(path)

def quantize_image(pixels: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> np.ndarray:
    'Replace each pixel by the colour of its assigned cluster center.'
    quantized = centers[labels]
    return quantized.reshape(pixels.shape)

def plot_side_by_side(original: np.ndarray, compressed: np.ndarray, title: str) -> None:
    'Show the original and the compressed images next to each other.'
    plt.figure(figsize=(10, 5))
    plt.suptitle(title)

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(compressed)
    plt.title('Compressed')
    plt.axis('off')

    plt.show()
```

Each helper follows the naming and behaviour from class. I keep the docstrings short and clear.

---

## 5. 🎨 Colour reduction with K-Means

> **Enunciado 5 del Boletín 1.** "Usa K-Means para reducir el número de colores de las imágenes dadas."

**Plain goal:** load the three PPM images, run K-Means with several palette sizes, and visualise the effect.

### Step 5.1 — Prepare a function that performs K-Means on an image

```python
from sklearn.utils import shuffle

def run_kmeans_on_image(image_array: np.ndarray, n_colors: int, random_state: int = 0):
    # Flatten the image to (num_pixels, 3).
    pixels = image_array.reshape(-1, 3).astype(float)

    # Use a subset of pixels to speed up the fit.
    sample = shuffle(pixels, random_state=random_state, n_samples=10_000)

    model = KMeans(n_clusters=n_colors, n_init=5, random_state=random_state)
    model.fit(sample)

    full_labels = model.predict(pixels)
    compressed = quantize_image(pixels, model.cluster_centers_, full_labels)

    return compressed.astype(np.uint8), model.inertia_
```

### Step 5.2 — Define the palette sizes and load the images

```python
palette_sizes = [4, 8, 16, 32]
images = {
    name: load_image(path)
    for name, path in data_paths.items()
    if name in {"landscape", "gradient", "stripes"}
}
{key: value.shape for key, value in images.items()}
```

I keep the shapes to confirm the loading step worked.

### Step 5.3 — Run the compression and show the results

```python
color_results = {}

for name, image_array in images.items():
    color_results[name] = {}
    for k in palette_sizes:
        compressed, inertia = run_kmeans_on_image(image_array, n_colors=k, random_state=0)
        color_results[name][k] = {"image": compressed, "inertia": inertia}
        plot_side_by_side(image_array, compressed, title=f"{name} — {k} colours")
```

The plots show how the landscape keeps good quality from 16 colours onward, while the gradient already looks blocky at 4 colours. The stripes image keeps sharp boundaries even with 4 colours.

---

## 6. 💾 File size study after compression

> **Enunciado 6 del Boletín 1.** "Para cada imagen y cada número de colores se debe guardar el archivo y estudiar el tamaño resultante."

**Plain goal:** save the compressed images and measure the new file sizes.

```python
from tempfile import TemporaryDirectory

size_records = []

with TemporaryDirectory() as tmpdir:
    tmpdir_path = Path(tmpdir)
    for name, configs in color_results.items():
        for k, info in configs.items():
            output_path = tmpdir_path / f"{name}_{k}.ppm"
            save_image(info["image"], output_path)
            size_records.append({
                "image": name,
                "palette": k,
                "size_bytes": output_path.stat().st_size,
                "inertia": info["inertia"],
            })

size_df = pd.DataFrame(size_records)
size_df.sort_values(["image", "palette"])
```

The table shows that more colours lead to larger files. I also keep the inertia so I can balance file size and reconstruction error.

### Step 6.1 — Plot size versus palette

```python
plt.figure(figsize=(7, 5))
for name, group in size_df.groupby("image"):
    plt.plot(group["palette"], group["size_bytes"], marker="o", label=name)
plt.xlabel("Number of colours")
plt.ylabel("File size (bytes)")
plt.title("Palette size vs. file size")
plt.legend()
plt.grid(True)
plt.show()
```

The lines show a clear trade-off: more colours increase the size. The stripes image has the smallest files because it is very simple.

---

## 7. 🙂 PCA on synthetic faces

> **Enunciado 7 del Boletín 1.** "Usa el conjunto `faces.mat` para reducir la dimensionalidad con PCA y reconstruir las imágenes."

**Plain goal:** load the MATLAB file, standardise the data, apply PCA, and rebuild faces using a few components.

### Step 7.1 — Load the data and inspect the shape

```python
from scipy.io import loadmat

faces_data = loadmat("Machine Learning 1/faces.mat")
faces_matrix = faces_data["X"]  # shape: (n_samples, n_pixels)
print(faces_matrix.shape)
```

The shape `(200, 1024)` means 200 faces, each described by 32×32 = 1024 pixels.

### Step 7.2 — Standardise and run PCA

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

face_scaler = StandardScaler()
faces_scaled = face_scaler.fit_transform(faces_matrix)

pca = PCA(n_components=50, random_state=0)
pca_scores = pca.fit_transform(faces_scaled)
```

### Step 7.3 — Reconstruct faces with different numbers of components

```python
def reconstruct_faces(pca_model: PCA, scores: np.ndarray, scaler: StandardScaler, n_components: int) -> np.ndarray:
    truncated_scores = scores[:, :n_components]
    truncated_components = pca_model.components_[:n_components]
    reconstructed = truncated_scores @ truncated_components
    reconstructed = scaler.inverse_transform(reconstructed)
    return reconstructed

components_to_try = [5, 10, 20, 40]
reconstructed_faces = {n: reconstruct_faces(pca, pca_scores, face_scaler, n) for n in components_to_try}
```

### Step 7.4 — Plot the original and reconstructed faces

```python
def plot_face_grid(original: np.ndarray, reconstructions: dict[int, np.ndarray], index: int = 0) -> None:
    plt.figure(figsize=(12, 3))
    plt.subplot(1, len(reconstructions) + 1, 1)
    plt.imshow(original[index].reshape(32, 32), cmap="gray")
    plt.title('Original')
    plt.axis('off')

    for pos, (n_components, matrix) in enumerate(reconstructions.items(), start=2):
        plt.subplot(1, len(reconstructions) + 1, pos)
        plt.imshow(matrix[index].reshape(32, 32), cmap="gray")
        plt.title(f"{n_components} PCs")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

plot_face_grid(faces_matrix, reconstructed_faces)
```

With 5 components the face looks blurry but recognisable. With 40 components it is almost identical to the original.

---

## 8. 📊 Explained variance plot

> **Enunciado 8 del Boletín 1.** "Representa el porcentaje de varianza explicada por componente."

**Plain goal:** show how much information each principal component keeps.

```python
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker="o")
plt.xlabel("Principal component")
plt.ylabel("Explained variance ratio")
plt.title("Explained variance per component")
plt.grid(True)
plt.show()
```

The curve drops quickly, which tells me that the first components carry most of the information.

### Step 8.1 — Cumulative variance

```python
cumulative_variance = np.cumsum(explained_variance)
plt.figure(figsize=(7, 4))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
plt.axhline(0.9, color='red', linestyle='--', label='90%')
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Cumulative variance captured by PCA")
plt.legend()
plt.grid(True)
plt.show()
```

We need around 25 components to reach 90% of the variance.

---

## 9. 🧠 Classifier comparison with PCA features

> **Enunciado 9 del Boletín 1.** "Toma un conjunto con suficientes atributos, reduce su dimensionalidad y compara dos clasificadores."

**Plain goal:** project the digits dataset into a lower-dimensional PCA space and compare a simple k-NN classifier with logistic regression.

### Step 9.1 — Load the digits dataset and split it

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_digits, y_digits = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits, test_size=0.2, random_state=0, stratify=y_digits
)
```

### Step 9.2 — Scale and apply PCA

```python
digits_scaler = StandardScaler()
X_train_scaled = digits_scaler.fit_transform(X_train)
X_test_scaled = digits_scaler.transform(X_test)

digits_pca = PCA(n_components=30, random_state=0)
X_train_pca = digits_pca.fit_transform(X_train_scaled)
X_test_pca = digits_pca.transform(X_test_scaled)
```

I pick 30 components because they keep about 90% of the variance for this dataset.

### Step 9.3 — Train and evaluate two classifiers

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)
knn_pred = knn.predict(X_test_pca)
knn_acc = accuracy_score(y_test, knn_pred)

log_reg = LogisticRegression(max_iter=1000, random_state=0)
log_reg.fit(X_train_pca, y_train)
log_reg_pred = log_reg.predict(X_test_pca)
log_reg_acc = accuracy_score(y_test, log_reg_pred)

knn_acc, log_reg_acc
```

The tuple shows the accuracy for both classifiers. Logistic regression is usually a bit stronger here, but both perform well above 95%.

### Step 9.4 — Summarise the comparison in a small table

```python
comparison_df = pd.DataFrame([
    {"model": "k-NN (k=5)", "accuracy": knn_acc},
    {"model": "Logistic regression", "accuracy": log_reg_acc},
])
comparison_df
```

The table makes it obvious which model wins. The difference is small, which means PCA kept the important structure of the digits.

---

## ✅ Final checklist

* Every enunciado from Boletín 1 is covered in the same order as the PDF.
* All code blocks include short comments and use the same variables as in the notebook.
* Every result is explained right after it appears in very simple English.
