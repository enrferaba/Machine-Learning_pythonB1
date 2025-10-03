from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("Files-20250930 (2)")
IMAGES_DIR = Path("prueba1/images")
OUTPUT_DIR = Path("prueba1/reduced_images")

paths: Dict[str, Path] = {
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
        faces_list: List[np.ndarray] = []
        labels_list: List[int] = []

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
                canvas += 0.6 * np.exp(-(dist**2) / (2 * (eye_radius**2)))

            pupil_radius = eye_radius * 0.4
            for sign in (-1, 1):
                ex = cx + sign * eye_dx
                ey = eye_y + rng.normal(0, 0.3)
                dist = ((x_coords - ex) ** 2 + (y_coords - ey) ** 2) ** 0.5
                canvas += 0.8 * np.exp(-(dist**2) / (2 * (pupil_radius**2)))

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
    except ImportError as missing_scipy:  # pragma: no cover - user installs SciPy once
        raise ImportError(
            "SciPy is required to create faces.mat automatically. Install it or copy the file manually."
        ) from missing_scipy

    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {"X": faces, "l": labels})
    print(f"Saved {faces.shape[0]} faces ({origin}) to {path}")


def ensure_practice_paths() -> None:
    for name, target in paths.items():
        if name == "faces":
            ensure_faces_dataset(target)
        else:
            if not target.exists():
                raise FileNotFoundError(f"Missing {name} file: {target}")


ZOO_COLUMNS: List[str] = [
    "animal_name",
    "hair",
    "feathers",
    "eggs",
    "milk",
    "airborne",
    "aquatic",
    "predator",
    "toothed",
    "backbone",
    "breathes",
    "venomous",
    "fins",
    "legs",
    "tail",
    "domestic",
    "catsize",
    "type",
]


def load_zoo(include_type: bool = False) -> Tuple[pd.DataFrame, List[str], pd.DataFrame, pd.Series]:
    ensure_practice_paths()
    df = pd.read_csv(paths["zoo"], header=None, names=ZOO_COLUMNS)
    if include_type:
        feature_cols = [col for col in ZOO_COLUMNS if col != "animal_name"]
    else:
        feature_cols = [col for col in ZOO_COLUMNS if col not in {"animal_name", "type"}]
    X = df[feature_cols].astype(float)
    y = df["type"].astype(int)
    return df, feature_cols, X, y


def scale_features(X: pd.DataFrame) -> Tuple[StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled


def grid_search_kmeans(
    X: np.ndarray,
    y: Iterable[int],
    k_values: Iterable[int],
    seed_values: Iterable[int],
) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    rows = []
    for k in k_values:
        for seed in seed_values:
            model = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = model.fit_predict(X)
            rows.append(
                {
                    "k": int(k),
                    "seed": int(seed),
                    "inertia": float(model.inertia_),
                    "silhouette": float(silhouette_score(X, labels)),
                    "ARI": float(adjusted_rand_score(y, labels)),
                }
            )
    return pd.DataFrame(rows)


def load_faces() -> Tuple[np.ndarray, np.ndarray]:
    ensure_practice_paths()
    try:
        from scipy.io import loadmat
    except ImportError as missing_scipy:  # pragma: no cover - SciPy required
        raise ImportError("SciPy is required to load faces.mat") from missing_scipy

    mat = loadmat(paths["faces"])
    X = np.asarray(mat["X"], dtype=np.float32)
    y = np.asarray(mat["l"], dtype=np.int16).ravel()
    return X, y


def load_digits_split(test_size: float = 0.3, random_state: int = 0):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def load_image_array(name: str) -> np.ndarray:
    ensure_practice_paths()
    try:
        from PIL import Image
    except ImportError as missing_pillow:  # pragma: no cover - Pillow required for images
        raise ImportError("Pillow is required to load the PPM images") from missing_pillow

    with Image.open(paths[name]) as img:
        return np.asarray(img, dtype=np.float32) / 255.0


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
