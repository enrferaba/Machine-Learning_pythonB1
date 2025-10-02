# Bulletin 1 – Step-by-Step Walkthrough (Friendly 12-year-old Edition)

Hi! I'm pretending to be a third-year software engineering student who explains everything like I am twelve. That means I will show every tiny step, write the code in small chunks, and tell you what I notice after each action. I am re-solving the original bulletin exercises in the same order as the class syllabus.

---

## Part 1 – K-Means on the Zoo Dataset

### Step 1.1 – Peek at the raw CSV so I know what I am touching
```python
from pathlib import Path

zoo_path = Path('Files-20250930 (2)/zoo.data')
with zoo_path.open() as handle:
    first_rows = [next(handle).strip() for _ in range(5)]
for line in first_rows:
    print(line)
```
When I print the first five rows I see the animal name followed by 16 numbers. The last number is the **type** column that tells me the class of the animal. 【F:trabajo_step_by_step.md†L12-L20】【b44acb†L6-L13】

### Step 1.2 – Turn the text into numbers and measure the columns
```python
import csv, math

names, raw_matrix, raw_labels = [], [], []
with zoo_path.open() as handle:
    reader = csv.reader(handle)
    for row in reader:
        names.append(row[0])
        raw_matrix.append([float(x) for x in row[1:-1]])  # 16 features
        raw_labels.append(int(row[-1]))                  # class from 1 to 7

feature_means = []
feature_stds = []
for col in range(len(raw_matrix[0])):
    values = [animal[col] for animal in raw_matrix]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance) or 1.0  # I replace 0 with 1 so I do not divide by 0 later
    feature_means.append(mean)
    feature_stds.append(std)

print('We have', len(names), 'animals and', len(raw_matrix[0]), 'features')
print('First feature mean:', feature_means[0], 'std:', feature_stds[0])
```
After computing the mean and standard deviation for every feature I know how to standardise later. For example, feature 1 ("hair") has mean 0.4257 and standard deviation 0.4945. 【F:trabajo_step_by_step.md†L22-L42】【74bcea†L1-L23】

### Step 1.3 – Standardise features so K-Means does not get confused
```python
def standardise(matrix, means, stds):
    scaled = []
    for animal in matrix:
        scaled.append([(value - means[idx]) / stds[idx] for idx, value in enumerate(animal)])
    return scaled

scaled_matrix = standardise(raw_matrix, feature_means, feature_stds)
```
Now every column has mean 0 and variance 1 (or close), which keeps the Euclidean distances fair. 【F:trabajo_step_by_step.md†L44-L53】

### Step 1.4 – Build K-Means piece by piece
```python
def squared_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def kmeans(matrix, k, seed, max_iter=100):
    rng = random.Random(seed)
    centroids = [matrix[i][:] for i in rng.sample(range(len(matrix)), k)]
    assignments = [None] * len(matrix)

    for _ in range(max_iter):
        changed = False
        # Step A – assign each animal to its nearest centroid
        for idx, animal in enumerate(matrix):
            best_cluster = None
            best_distance = None
            for cluster_idx, centroid in enumerate(centroids):
                distance = squared_distance(animal, centroid)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_cluster = cluster_idx
            if assignments[idx] != best_cluster:
                assignments[idx] = best_cluster
                changed = True

        # Step B – recompute centroids using simple averages
        new_centroids = [[0.0] * len(matrix[0]) for _ in range(k)]
        counts = [0] * k
        for idx, cluster in enumerate(assignments):
            counts[cluster] += 1
            for j, value in enumerate(matrix[idx]):
                new_centroids[cluster][j] += value
        for cluster in range(k):
            if counts[cluster] > 0:
                for j in range(len(matrix[0])):
                    new_centroids[cluster][j] /= counts[cluster]
        centroids = new_centroids

        if not changed:
            break  # I stop early when the labels stop moving

    sse = sum(squared_distance(matrix[i], centroids[assignments[i]]) for i in range(len(matrix)))
    return assignments, centroids, sse
```
I wrote the algorithm exactly like we did in class, and I kept many comments to remind myself of the sub-steps. 【F:trabajo_step_by_step.md†L55-L99】

### Step 1.5 – Try several values of k and seeds
```python
seeds = [2, 7, 11]
k_values = [5, 6, 7, 8]
records = []
for k in k_values:
    row = {'k': k}
    sses = []
    for seed in seeds:
        _, _, sse = kmeans(scaled_matrix, k, seed)
        row[f'seed{seed}'] = sse
        sses.append(sse)
    row['average'] = sum(sses) / len(sses)
    records.append(row)

for row in records:
    print(row)
```
This prints the same information as the following table (rounded to 3 decimals):

| k | SSE (seed=2) | SSE (seed=7) | SSE (seed=11) | Average SSE |
|---|--------------|--------------|---------------|-------------|
| 5 | 695.889 | 667.176 | 695.202 | 686.089 |
| 6 | 663.707 | 657.975 | 646.078 | 655.920 |
| 7 | 595.692 | 593.235 | 584.896 | 591.274 |
| 8 | 544.119 | 542.710 | 560.478 | 549.102 |

The inertia keeps shrinking when k grows, but the improvement from 7 to 8 is smaller than before, so k = 7 looks nice. 【F:trabajo_step_by_step.md†L101-L124】【9030e5†L1-L6】

### Step 1.6 – Describe each cluster like a human
```python
best_k = 7
assignments, centroids, _ = kmeans(scaled_matrix, best_k, seed=2)

cluster_summary = []
for cluster in range(best_k):
    members = [idx for idx, label in enumerate(assignments) if label == cluster]
    milk_mean = sum(raw_matrix[idx][3] for idx in members) / len(members)
    legs_mean = sum(raw_matrix[idx][12] for idx in members) / len(members)
    examples = ', '.join(names[idx] for idx in members[:5])
    cluster_summary.append({'cluster': cluster,
                            'size': len(members),
                            'milk_mean': milk_mean,
                            'legs_mean': legs_mean,
                            'examples': examples})

for info in cluster_summary:
    print(info)
```
I look at the average milk flag and number of legs to see what kind of animals ended up inside. For instance, cluster 2 has milk_mean = 1 and legs around 3.4, so it clearly contains mammals. 【F:trabajo_step_by_step.md†L126-L150】【7934cf†L40-L46】

### Step 1.7 – Cross-check with the real classes to make sure I understand the mistakes
```python
type_names = {1: 'Mammal', 2: 'Bird', 3: 'Reptile', 4: 'Fish', 5: 'Amphibian', 6: 'Bug', 7: 'Invertebrate'}

confusion = {name: [0] * best_k for name in type_names.values()}
for idx, real_type in enumerate(raw_labels):
    confusion[type_names[real_type]][assignments[idx]] += 1

for animal_type, counts in confusion.items():
    print(animal_type, counts)
```
The confusion table shows, for example, that all 41 mammals are split between cluster 2 and cluster 5, while all fishes are in cluster 0. That means the clusters are meaningful. 【F:trabajo_step_by_step.md†L152-L167】【7934cf†L47-L53】

---

## Part 2 – Agglomerative Hierarchical Clustering

### Step 2.1 – Create a distance dictionary that stores every pair of animals
```python
def pairwise_distances(matrix):
    distances = {}
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            distances[(i, j)] = squared_distance(matrix[i], matrix[j])
    return distances
```
I reuse the same squared distance from Part 1 so that all clustering methods compare apples with apples. 【F:trabajo_step_by_step.md†L169-L177】

### Step 2.2 – Build the agglomerative model with the four linkage options
```python
class AgglomerativeClusterer:
    def __init__(self, matrix, linkage):
        self.matrix = matrix
        self.linkage = linkage
        self.distances = pairwise_distances(matrix)
        self.active = {i: {i} for i in range(len(matrix))}
        self.centroids = {i: matrix[i][:] for i in range(len(matrix))}
        self.sizes = {i: 1 for i in range(len(matrix))}
        self.next_id = len(matrix)
        self.merges = []

    def _cluster_distance(self, a, b):
        points_a = self.active[a]
        points_b = self.active[b]
        if self.linkage == 'single':
            return min(self.distances[(min(i, j), max(i, j))] for i in points_a for j in points_b)
        if self.linkage == 'complete':
            return max(self.distances[(min(i, j), max(i, j))] for i in points_a for j in points_b)
        if self.linkage == 'average':
            total = sum(self.distances[(min(i, j), max(i, j))] for i in points_a for j in points_b)
            return total / (len(points_a) * len(points_b))
        if self.linkage == 'ward':
            size_a = self.sizes[a]
            size_b = self.sizes[b]
            centroid_a = self.centroids[a]
            centroid_b = self.centroids[b]
            return (size_a * size_b) / (size_a + size_b) * squared_distance(centroid_a, centroid_b)
        raise ValueError('Unknown linkage')

    def _step(self):
        candidates = list(self.active.keys())
        best_pair, best_distance = None, None
        for idx_a in range(len(candidates)):
            for idx_b in range(idx_a + 1, len(candidates)):
                a, b = candidates[idx_a], candidates[idx_b]
                distance = self._cluster_distance(a, b)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_pair = (a, b)
        a, b = best_pair
        new_id = self.next_id
        self.next_id += 1
        merged_points = self.active[a] | self.active[b]
        self.active[new_id] = merged_points
        del self.active[a]
        del self.active[b]
        self.sizes[new_id] = len(merged_points)
        total = [0.0] * len(self.matrix[0])
        for idx in merged_points:
            for j, value in enumerate(self.matrix[idx]):
                total[j] += value
        self.centroids[new_id] = [value / len(merged_points) for value in total]
        self.merges.append((a, b, best_distance, len(merged_points)))

    def fit(self, n_clusters):
        while len(self.active) > n_clusters:
            self._step()
        clusters = list(self.active.values())
        assignments = [None] * len(self.matrix)
        for cluster_idx, indices in enumerate(clusters):
            for point in indices:
                assignments[point] = cluster_idx
        return assignments, self.merges
```
This class mimics scikit-learn but with the formulas we wrote by hand in the lecture. I carefully added comments to keep track of what each attribute stores. 【F:trabajo_step_by_step.md†L179-L239】

### Step 2.3 – Define the external metrics (Rand, Adjusted Rand, mutual information, homogeneity, completeness, V-measure) and the silhouette
```python
def contingency_matrix(labels_true, labels_pred):
    classes = sorted(set(labels_true))
    clusters = sorted(set(labels_pred))
    table = {cls: {cluster: 0 for cluster in clusters} for cls in classes}
    for truth, pred in zip(labels_true, labels_pred):
        table[truth][pred] += 1
    return table


def rand_scores(labels_true, labels_pred):
    table = contingency_matrix(labels_true, labels_pred)
    total_pairs = math.comb(len(labels_true), 2)
    sum_comb_c = sum(math.comb(sum(row.values()), 2) for row in table.values())
    cluster_sums = {}
    for row in table.values():
        for cluster, count in row.items():
            cluster_sums[cluster] = cluster_sums.get(cluster, 0) + count
    sum_comb_k = sum(math.comb(count, 2) for count in cluster_sums.values())
    sum_comb = sum(math.comb(count, 2) for row in table.values() for count in row.values())
    expected = (sum_comb_c * sum_comb_k) / total_pairs if total_pairs else 0.0
    max_index = 0.5 * (sum_comb_c + sum_comb_k)
    ari = (sum_comb - expected) / (max_index - expected) if max_index != expected else 0.0
    ri = (sum_comb + (total_pairs - sum_comb_c - sum_comb_k + sum_comb)) / total_pairs if total_pairs else 0.0
    return ri, ari


def entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    result = 0.0
    for count in counts:
        if count == 0:
            continue
        p = count / total
        result -= p * math.log(p, 2)
    return result


def mutual_information(labels_true, labels_pred):
    table = contingency_matrix(labels_true, labels_pred)
    n = len(labels_true)
    clusters = sorted(set(labels_pred))
    class_totals = {cls: sum(row.values()) for cls, row in table.items()}
    cluster_totals = {cluster: sum(table[cls][cluster] for cls in table) for cluster in clusters}
    mi = 0.0
    for cls, row in table.items():
        for cluster, count in row.items():
            if count == 0:
                continue
            mi += (count / n) * math.log((count * n) / (class_totals[cls] * cluster_totals[cluster]), 2)
    return mi


def homogeneity_completeness(labels_true, labels_pred):
    table = contingency_matrix(labels_true, labels_pred)
    n = len(labels_true)
    classes = list(table.keys())
    clusters = sorted(set(labels_pred))
    class_totals = {cls: sum(table[cls].values()) for cls in classes}
    cluster_totals = {cluster: sum(table[cls][cluster] for cls in classes) for cluster in clusters}
    h_c = entropy(class_totals.values())
    h_k = entropy(cluster_totals.values())
    conditional_c = 0.0
    for cluster in clusters:
        for cls in classes:
            count = table[cls][cluster]
            if count == 0 or cluster_totals[cluster] == 0:
                continue
            conditional_c -= (count / n) * math.log(count / cluster_totals[cluster], 2)
    conditional_k = 0.0
    for cls in classes:
        for cluster in clusters:
            count = table[cls][cluster]
            if count == 0 or class_totals[cls] == 0:
                continue
            conditional_k -= (count / n) * math.log(count / class_totals[cls], 2)
    homogeneity = 1 - conditional_c / h_c if h_c > 0 else 1.0
    completeness = 1 - conditional_k / h_k if h_k > 0 else 1.0
    v_measure = (2 * homogeneity * completeness / (homogeneity + completeness)) if (homogeneity + completeness) > 0 else 0.0
    return homogeneity, completeness, v_measure


def silhouette_score(matrix, labels_pred):
    distances = pairwise_distances(matrix)
    def dist(i, j):
        if i == j:
            return 0.0
        return distances[(i, j)] if i < j else distances[(j, i)]

    silhouettes = []
    for i in range(len(matrix)):
        cluster = labels_pred[i]
        same_cluster = [j for j in range(len(matrix)) if labels_pred[j] == cluster and j != i]
        a = sum(dist(i, j) for j in same_cluster) / len(same_cluster) if same_cluster else 0.0
        b = None
        for other in sorted(set(labels_pred)):
            if other == cluster:
                continue
            members = [j for j in range(len(matrix)) if labels_pred[j] == other]
            if not members:
                continue
            avg = sum(dist(i, j) for j in members) / len(members)
            if b is None or avg < b:
                b = avg
        if b is None:
            silhouettes.append(0.0)
        else:
            silhouettes.append(0.0 if max(a, b) == 0 else (b - a) / max(a, b))
    return sum(silhouettes) / len(silhouettes)
```
These helpers let me compute the same quality numbers as scikit-learn. 【F:trabajo_step_by_step.md†L241-L343】

### Step 2.4 – Run the four linkages and compare the metrics
```python
def evaluate_linkage(matrix, labels_true, linkage, n_clusters=7):
    model = AgglomerativeClusterer(matrix, linkage)
    assignments, merges = model.fit(n_clusters)
    ri, ari = rand_scores(labels_true, assignments)
    mi = mutual_information(labels_true, assignments)
    h, c, v = homogeneity_completeness(labels_true, assignments)
    sil = silhouette_score(matrix, assignments)
    return {
        'ri': ri,
        'ari': ari,
        'mi': mi,
        'homogeneity': h,
        'completeness': c,
        'v_measure': v,
        'silhouette': sil,
        'assignments': assignments,
        'merges': merges
    }

for linkage in ['single', 'complete', 'average', 'ward']:
    metrics = evaluate_linkage(scaled_matrix, raw_labels, linkage)
    print(linkage, metrics['ri'], metrics['ari'], metrics['silhouette'])
```
Formatted nicely, the results look like this:

| Linkage | RI | ARI | Silhouette |
|---------|-----|------|------------|
| single | 0.745 | 0.478 | 0.375 |
| complete | 0.968 | 0.911 | 0.565 |
| average | 0.894 | 0.716 | 0.567 |
| ward | 0.895 | 0.680 | 0.535 |

The complete linkage wins in both Rand Index and silhouette, which means the clusters are both faithful and well separated. 【F:trabajo_step_by_step.md†L345-L377】【ec45ba†L1-L4】

### Step 2.5 – Study how the Ward silhouette changes with the number of clusters
```python
ward_silhouettes = []
for clusters in range(2, 11):
    model = AgglomerativeClusterer(scaled_matrix, 'ward')
    labels_pred, _ = model.fit(clusters)
    score = silhouette_score(scaled_matrix, labels_pred)
    ward_silhouettes.append((clusters, score))
    print(clusters, score)
```
This is the curve I observe:

| Clusters | Silhouette |
|----------|------------|
| 2 | 0.394 |
| 3 | 0.467 |
| 4 | 0.550 |
| 5 | 0.560 |
| 6 | 0.490 |
| 7 | 0.535 |
| 8 | 0.529 |
| 9 | 0.536 |
| 10 | 0.557 |

The silhouette peaks around 5 clusters and stays stable afterwards, which matches the zoo classes fairly well. 【F:trabajo_step_by_step.md†L379-L404】【28bafa†L1-L9】

### Step 2.6 – Glance at the first merges of the dendrogram (complete linkage)
```python
complete_merges = evaluate_linkage(scaled_matrix, raw_labels, 'complete')['merges']
for merge in complete_merges[:10]:
    a, b, dist, size = merge
    print(f'Merge {a} with {b} at distance {dist:.3f} -> size {size}')
```
The earliest merges combine animals that are almost identical (distance 0), which reassures me that the algorithm is doing its job. 【F:trabajo_step_by_step.md†L406-L415】【a607c0†L1-L10】

---

## Part 3 – Manual DBSCAN Example (Problem 5 Verification)

### Step 3.1 – Place the 12 points in 2D space
```python
points = {
    'P1': (1.0, 1.2), 'P2': (0.8, 1.1), 'P3': (1.2, 0.9),
    'P4': (8.0, 8.5), 'P5': (8.2, 8.3), 'P6': (7.9, 8.1),
    'P7': (5.0, 1.0), 'P8': (5.2, 1.1), 'P9': (5.1, 0.9),
    'P10': (3.0, 6.0), 'P11': (3.1, 6.2), 'P12': (2.9, 5.9)
}
```
These four tiny clusters are the same ones from the worksheet: three groups at the corners and one around (3, 6). 【F:trabajo_step_by_step.md†L417-L426】

### Step 3.2 – Implement the algorithm exactly like in class and run it
```python
import math

order = list(points.keys())
coords = [points[label] for label in order]


def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def region_query(index, eps):
    return [j for j in range(len(coords)) if euclidean(coords[index], coords[j]) <= eps]


def dbscan(eps, min_pts):
    neighbour_cache = [region_query(i, eps) for i in range(len(coords))]
    core_points = {i for i in range(len(coords)) if len(neighbour_cache[i]) >= min_pts}
    visited = [False] * len(coords)
    labels = [None] * len(coords)
    cluster_id = 0

    for i in range(len(coords)):
        if visited[i]:
            continue
        visited[i] = True
        if i not in core_points:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        queue = [j for j in neighbour_cache[i] if j != i]
        while queue:
            j = queue.pop(0)
            if not visited[j]:
                visited[j] = True
                if j in core_points:
                    for q in neighbour_cache[j]:
                        if q not in queue:
                            queue.append(q)
            if labels[j] in (None, -1):
                labels[j] = cluster_id
        cluster_id += 1

    for i in range(len(coords)):
        if labels[i] is None:
            labels[i] = -1

    border = {i for i in range(len(coords)) if labels[i] >= 0 and i not in core_points}
    noise = {i for i, label in enumerate(labels) if label == -1}
    return labels, core_points, border, noise

labels_found, core_set, border_set, noise_set = dbscan(eps=0.5, min_pts=3)
clusters = {}
for idx, cluster in enumerate(labels_found):
    clusters.setdefault(cluster, []).append(order[idx])
print('clusters', clusters)
print('cores', [order[i] for i in sorted(core_set)])
print('border', [order[i] for i in sorted(border_set)])
print('noise', [order[i] for i in sorted(noise_set)])
```
All twelve points are core points, and DBSCAN discovers the four groups {P1,P2,P3}, {P4,P5,P6}, {P7,P8,P9}, {P10,P11,P12} with zero noise, exactly like the theoretical example promised. 【F:trabajo_step_by_step.md†L428-L476】【326e0e†L1-L27】

---

## Part 4 – Image Compression with K-Means

### Step 4.1 – Generate three simple images (gradient, stripes, landscape)
```python
def gradient_image(width, height):
    image = []
    for y in range(height):
        row = []
        for x in range(width):
            r = int(255 * x / (width - 1))
            g = int(255 * y / (height - 1))
            b = int(255 * (x + y) / (width + height - 2))
            row.append((r, g, b))
        image.append(row)
    return image


def stripes_image(width, height):
    image = []
    for y in range(height):
        row = []
        for x in range(width):
            stripe = (x // max(1, width // 8)) % 3
            if stripe == 0:
                color = (220, 20, 60)
            elif stripe == 1:
                color = (30, 144, 255)
            else:
                color = (255, 215, 0)
            row.append(color)
        image.append(row)
    return image


def landscape_image(width, height):
    image = []
    horizon = height // 2
    for y in range(height):
        row = []
        for x in range(width):
            if y < horizon:
                b = int(200 + 55 * y / max(1, horizon))
                g = int(150 + 80 * y / max(1, horizon))
                r = int(120 + 30 * y / max(1, horizon))
            else:
                factor = (y - horizon) / max(1, height - horizon - 1)
                g = int(120 + 80 * (1 - factor))
                r = int(40 + 40 * factor)
                b = int(20 + 60 * (1 - factor))
            row.append((r, g, b))
        image.append(row)
    return image
```
I keep the functions tiny and descriptive so I can tweak the colours easily if something looks odd. 【F:trabajo_step_by_step.md†L478-L531】

### Step 4.2 – Reuse the K-Means function to quantise colours and check the reconstruction
```python
import random


def compress_image(image, k, seed=42):
    height = len(image)
    width = len(image[0])
    flat_pixels = [list(pixel) for row in image for pixel in row]
    assignments, centroids, sse = kmeans(flat_pixels, k, seed)
    idx = 0
    new_image = []
    for _ in range(height):
        row = []
        for _ in range(width):
            centroid = centroids[assignments[idx]]
            row.append(tuple(int(round(value)) for value in centroid))
            idx += 1
        new_image.append(row)
    return new_image, sse

images = {
    'gradient': gradient_image(32, 32),
    'stripes': stripes_image(32, 32),
    'landscape': landscape_image(48, 32)
}
portrait_k = [3, 5, 10, 16, 20, 32, 50, 64]
landscape_k = [5, 10, 20]

results = {}
for name, img in images.items():
    ks = landscape_k if name == 'landscape' else portrait_k
    entries = []
    for k in ks:
        _, sse = compress_image(img, k)
        entries.append((k, sse))
    results[name] = entries

for name, entries in results.items():
    print(name, [(k, round(sse, 2)) for k, sse in entries])
```
I get the following error (SSE) values:

- Gradient: [(3, 5 748 816.72), (5, 3 090 741.53), (10, 1 477 751.76), (16, 931 578.78), (20, 736 586.03), (32, 469 494.70), (50, 304 810.28), (64, 240 669.42)]
- Stripes: [(3, 6 581 760.00), then 0 for any k ≥ 5 because the three stripe colours are captured perfectly]
- Landscape: [(5, 290 146.29), (10, 124 312.34), (20, 36 816.00)]

Even a child can see that the gradient needs more colours than the stripes to look smooth. 【F:trabajo_step_by_step.md†L533-L587】【c1ec3d†L1-L5】

### Step 4.3 – Peek at the ASCII version of the gradient compressed with k = 5
```python
preview, _ = compress_image(images['gradient'], 5)
for row in preview[:1]:  # first row
    print(row[:8])
```
The first row becomes eight identical pixels around RGB (50, 62, 56), so the transition looks blocky, which matches my expectation for such a low k. 【F:trabajo_step_by_step.md†L589-L594】【c1ec3d†L1-L5】

---

## Part 5 – PCA on Synthetic Faces

### Step 5.1 – Create base patterns and generate noisy samples
```python
def generate_base_patterns(size=8):
    patterns = []
    pattern1 = [(y + 1) / size for y in range(size) for x in range(size)]
    pattern2 = [(x + 1) / size for y in range(size) for x in range(size)]
    pattern3 = [(x + y) / (2 * size) for y in range(size) for x in range(size)]
    return [pattern1, pattern2, pattern3]

base_patterns = generate_base_patterns()


def add_noise(pattern, level=0.1, rng=None):
    rng = rng or random.Random()
    noisy = []
    for value in pattern:
        noise = (rng.random() - 0.5) * 2 * level
        noisy.append(min(max(value + noise, 0.0), 1.0))
    return noisy


def generate_face_dataset(samples=90, size=8, seed=123):
    rng = random.Random(seed)
    data, labels = [], []
    for idx in range(samples):
        base_idx = idx % len(base_patterns)
        data.append(add_noise(base_patterns[base_idx], level=0.1, rng=rng))
        labels.append(base_idx)
    return data, labels

faces, face_labels = generate_face_dataset()
print('Dataset size:', len(faces))
```
I build three smooth base faces and add tiny noise so every sample still looks like the right prototype. 【F:trabajo_step_by_step.md†L596-L640】

### Step 5.2 – Compute the mean vector, covariance matrix, and eigenpairs with power iteration
```python
def mean_vector(data):
    mean = [0.0] * len(data[0])
    for vector in data:
        for i, value in enumerate(vector):
            mean[i] += value
    return [value / len(data) for value in mean]


def center_data(data, mean):
    return [[value - mean[i] for i, value in enumerate(vector)] for vector in data]


def covariance_matrix(data):
    n = len(data)
    dim = len(data[0])
    cov = [[0.0] * dim for _ in range(dim)]
    for vector in data:
        for i in range(dim):
            for j in range(i, dim):
                cov[i][j] += vector[i] * vector[j]
    for i in range(dim):
        for j in range(i, dim):
            cov_val = cov[i][j] / (n - 1 if n > 1 else 1)
            cov[i][j] = cov_val
            cov[j][i] = cov_val
    return cov


def mat_vec_mul(matrix, vector):
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def normalize(vector):
    norm = math.sqrt(sum(x * x for x in vector))
    return vector[:] if norm == 0 else [x / norm for x in vector]


def power_iteration(matrix, iterations=1000, tolerance=1e-9, seed=0):
    rng = random.Random(seed)
    vec = normalize([rng.random() for _ in range(len(matrix))])
    for _ in range(iterations):
        next_vec = normalize(mat_vec_mul(matrix, vec))
        diff = max(abs(next_vec[i] - vec[i]) for i in range(len(vec)))
        vec = next_vec
        if diff < tolerance:
            break
    eigenvalue = dot(vec, mat_vec_mul(matrix, vec))
    return eigenvalue, vec


def deflate(matrix, eigenvalue, eigenvector):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            matrix[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j]


def pca(data, components_count):
    mean = mean_vector(data)
    centered = center_data(data, mean)
    cov = covariance_matrix(centered)
    working = [row[:] for row in cov]
    eigenvalues, components = [], []
    for idx in range(components_count):
        eigenvalue, eigenvector = power_iteration(working, seed=idx)
        eigenvalues.append(eigenvalue)
        components.append(eigenvector)
        deflate(working, eigenvalue, eigenvector)
    return mean, components, eigenvalues
```
This is the longest chunk, but each helper mirrors what we wrote on the whiteboard, so it is approachable even if you are young. 【F:trabajo_step_by_step.md†L642-L718】

### Step 5.3 – Project, reconstruct, and inspect the explained variance
```python
mean_face, components, eigenvalues = pca(faces, components_count=10)
projections = project(faces, mean_face, components)
reconstructed = reconstruct(projections, mean_face, components)

total_variance = sum(eigenvalues)
variance_report = []
accumulated = 0.0
for eigenvalue in eigenvalues:
    ratio = eigenvalue / total_variance if total_variance else 0.0
    accumulated += ratio
    variance_report.append((eigenvalue, ratio, accumulated))

print('First five components:', [(round(ev, 4), round(ratio * 100, 2), round(acc * 100, 2))
                                 for ev, ratio, acc in variance_report[:5]])
```
The first component already captures ~86% of the variance and the first three together capture ~97%, which is why PCA is such a powerful compressor. 【F:trabajo_step_by_step.md†L720-L745】【f0453f†L1-L3】

### Step 5.4 – Compare the k-NN accuracy before and after PCA
```python
def project(dataset, mean, components):
    centered = center_data(dataset, mean)
    return [[dot(vector, component) for component in components] for vector in centered]


def reconstruct(projections, mean, components):
    reconstructions = []
    dim = len(mean)
    for coords in projections:
        vector = mean[:]
        for weight, component in zip(coords, components):
            for i in range(dim):
                vector[i] += weight * component[i]
        reconstructions.append(vector)
    return reconstructions


def train_test_split(data, labels, test_ratio=0.3, seed=321):
    indices = list(range(len(data)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    split = int(len(data) * (1 - test_ratio))
    train_idx = indices[:split]
    test_idx = indices[split:]
    train_data = [data[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_data = [data[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    return train_data, train_labels, test_data, test_labels


def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def knn_predict(train_data, train_labels, sample, k=3):
    distances = [(euclidean_distance(vector, sample), label) for vector, label in zip(train_data, train_labels)]
    distances.sort(key=lambda item: item[0])
    votes = {}
    for _, label in distances[:k]:
        votes[label] = votes.get(label, 0) + 1
    return max(votes.items(), key=lambda item: (item[1], -item[0]))[0]


def accuracy(train_data, train_labels, test_data, test_labels, k=3):
    hits = 0
    for sample, label in zip(test_data, test_labels):
        if knn_predict(train_data, train_labels, sample, k) == label:
            hits += 1
    return hits / len(test_labels)

train_data, train_labels, test_data, test_labels = train_test_split(faces, face_labels)
baseline_acc = accuracy(train_data, train_labels, test_data, test_labels)
proj_train = project(train_data, mean_face, components[:3])
proj_test = project(test_data, mean_face, components[:3])
pca_acc = accuracy(proj_train, train_labels, proj_test, test_labels)
print('baseline:', baseline_acc, 'after PCA:', pca_acc)
```
Both accuracies are 1.0 because the dataset is very clean, so PCA did not hurt performance. 【F:trabajo_step_by_step.md†L747-L815】【f0453f†L2-L3】

### Step 5.5 – Display one face before and after reconstruction to understand the effect
```python
def vector_to_ascii(vector, size=8, shades=" .:-=+*#%@"):
    rows = []
    levels = len(shades) - 1
    for row in range(size):
        chars = []
        for col in range(size):
            value = vector[row * size + col]
            index = max(0, min(levels, int(round(value * levels))))
            chars.append(shades[index])
        rows.append(''.join(chars))
    return '\n'.join(rows)

print('Original face:\n', vector_to_ascii(faces[0]))
print('\nReconstructed face:\n', vector_to_ascii(reconstructed[0]))
```
The ASCII art shows the same patterns with slightly smoother gradients, confirming that 10 components are enough to rebuild the image. 【F:trabajo_step_by_step.md†L817-L846】【0bd95a†L1-L10】

---

## Final Thoughts (because even twelve-year-olds reflect!)

- Clustering the zoo animals works best with **k = 7** and the complete linkage also aligns well with the labelled classes.
- The DBSCAN toy example is a nice sanity check: identical points within dense regions form clusters with no noise.
- K-Means colour quantisation trades off smooth gradients for smaller files; stripes behave differently because they already use a small palette.
- PCA keeps almost all the information using only three principal components, and k-NN does not lose accuracy on the synthetic faces.

And that is the entire bulletin solved again, this time with every step exposed. 【F:trabajo_step_by_step.md†L848-L857】
