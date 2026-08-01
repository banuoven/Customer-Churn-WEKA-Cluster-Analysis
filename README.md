# Cluster Analysis on Customer Churn Dataset

**Authors:** Banu Öven, Eyüp Erman Erman

## 1. Introduction

The objective of this project is to perform a comprehensive, unsupervised cluster analysis on
the Customer Churn dataset, discovering inherent grouping structures within customer usage
patterns. To preserve strict unsupervised integrity, the actual class label (`Churn`) was
entirely masked during clustering and used only *a posteriori* for cluster validation and
quality assessment.

## 2. Data Preprocessing & Methodology

The following pipeline was executed in **Weka 3.8.7**:

- **Attribute transformation:** the numerical churn indicator (1/0) was converted to a
  categorical format via the `NumericToNominal` filter, preventing algorithms from treating
  the class label as continuous.
- **Normalization strategy:** for distance-reliant algorithms (e.g. K-Means), Weka's internal
  standardization was applied so that features with larger ranges (like `Seconds of Use`) do
  not dominate the Euclidean distance calculations.
- **Evaluation mode:** clustering was run under "Classes to clusters evaluation" to generate
  an accurate confusion matrix post-clustering.

## 3. Theoretical Background and Implemented Algorithms

Five clustering paradigms were deployed:

### Hierarchical Clustering (Agglomerative)

Builds a hierarchy of clusters bottom-up. Three linkage criteria were evaluated:

- **Single Linkage** — minimum distance between clusters.
- **Complete Linkage** — maximum distance between clusters (less sensitive to outliers).
- **Ward's Method** — minimizes total within-cluster variance.

| Single Linkage | Complete Linkage | Ward's Method |
|---|---|---|
|<img width="1205" height="639" alt="01_dendrogram_single" src="https://github.com/user-attachments/assets/2e5ac455-2f5b-49fd-94e2-919c64148e13" />
 |<img width="1208" height="636" alt="02_dendrogram_complete" src="https://github.com/user-attachments/assets/481ee1f2-20b2-4b2d-9ad0-1430d57c3dd2" />
 |<img width="1208" height="636" alt="03_dendrogram_ward" src="https://github.com/user-attachments/assets/fa9eb66b-60a1-435d-ae4d-dadb5b569bf6" />
 |

### K-Means

A partition-based centroid algorithm configured for **k = 2**, iteratively minimizing the
within-cluster sum of squares (WCSS) using the standard Euclidean distance formula:

| <img width="248" height="32" alt="05_euclidean_distance_formula" src="https://github.com/user-attachments/assets/e420115a-5238-4675-9597-5daf9d587d26" />
)| <img width="1208" height="643" alt="04_kmeans_weka_output" src="https://github.com/user-attachments/assets/9ca31f31-d9c5-4396-bd8c-c72a9be14b4d" />
| <img width="1208" height="643" alt="04_kmeans_weka_output" src="https://github.com/user-attachments/assets/80f60926-5b94-4d64-a16f-fc6c8803d55a" />
|<img width="1208" height="643" alt="06_kmeans_visualize_scatter" src="https://github.com/user-attachments/assets/0b2ca14f-bb85-4cb2-8885-b79913c48381" /> |

### EM (Expectation-Maximization)

A probabilistic mixture model that assigns a probability distribution to each instance rather
than a hard cluster boundary.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

Groups densely packed points rather than relying on centroids. Parameters: **Epsilon = 0.9**
(neighborhood radius), **MinPoints = 6** (minimum instances to form a dense region).

### FarthestFirst

Selected as the supplementary method — a K-Means variant that places initial cluster centers
at the point farthest from existing centers, speeding up convergence.

## 4. Results and Cluster Quality Assessment

Clustering quality was assessed by mapping algorithmically generated clusters against the
actual `Churn` distribution (misclassification / error rate):

| Clustering Algorithm | Number of Clusters | Error Rate (%) |
|---|---|---|
| Hierarchical (Single Linkage) | 2 | 15.68% |
| K-Means | 2 | 17.05% |
| Hierarchical (Ward's Method) | 2 | 17.05% |
| FarthestFirst | 2 | 17.52% |
| EM (Expectation-Maximization) | 2 | 23.57% |
| Hierarchical (Complete Linkage) | 2 | 28.44% |
| DBSCAN (ε=0.9, MinPts=6) | Algorithm-determined | 29.17% |

**Weka outputs for each method:**

| Hierarchical (Single) | Hierarchical (Complete) | Hierarchical (Ward) |
|---|---|---|
| <img width="1207" height="636" alt="07_weka_hierarchical_single" src="https://github.com/user-attachments/assets/c2f7a479-8498-4d9c-9817-0e33fdf37af3" />
 |<img width="1209" height="636" alt="08_weka_hierarchical_complete" src="https://github.com/user-attachments/assets/3c347670-6518-463c-b864-8ab244c6850e" />
|<img width="1207" height="641" alt="09_weka_hierarchical_ward" src="https://github.com/user-attachments/assets/ef162978-3a1b-44ab-871b-369d0140a2c5" />
|

| EM | DBSCAN | FarthestFirst |
|---|---|---|
|<img width="1208" height="642" alt="10_weka_em_output" src="https://github.com/user-attachments/assets/b2944ec3-48f9-459d-965d-e955f4a9e4d5" /> | <img width="1208" height="638" alt="11_weka_dbscan_output" src="https://github.com/user-attachments/assets/19f46d41-7363-49b2-b575-c33ff5f04c1b" />
 | <img width="1209" height="638" alt="12_weka_farthestfirst_output" src="https://github.com/user-attachments/assets/c68fbb3e-694e-409f-bdcb-c7c23dfc52db" />
 |

## 5. Detailed Conclusion and Internal Quality Evaluation

- **The Single Linkage illusion (imbalance trap):** Single Linkage achieved the lowest
  external error rate (15.68%), but its cluster distribution is severely imbalanced (e.g.
  3,149 instances in Cluster 0 vs. 1 instance in Cluster 1) — a classic "majority class
  anomaly." The algorithm merely isolated a single outlier rather than discovering
  meaningful churn profiles, giving it zero actionable business value despite the
  deceivingly low error metric.

- **Internal validity and structural cohesion:** unsupervised models should not be judged on
  external labels alone. Internal quality metrics — conceptually aligned with Silhouette
  Score and Within-Cluster Sum of Squares (WCSS) — measure cohesion (similarity within a
  cluster) vs. separation (dissimilarity between clusters). Methods that fail to balance
  these (Single and Complete Linkage, both sensitive to outliers) show poor internal
  validity.

- **The true optimal models — K-Means & Ward's Method:** considering both balanced instance
  distribution and structural cohesion, K-Means and Ward's Method emerge as the best models
  for this dataset. Both achieved a stable 17.05% error rate while producing meaningful,
  proportionate clusters, since both minimize intra-cluster variance (WCSS). FarthestFirst's
  comparable result (17.52%) further supports the robustness of variance-minimizing
  algorithms here.

- **Distributional and density limitations (EM & DBSCAN):** DBSCAN (29.17%) and EM (23.57%)
  struggled significantly. DBSCAN's failure suggests high noise and varying local densities
  that prevent clear separation without heavy parameter tuning; EM's lower quality suggests
  the underlying customer features do not follow a strict Gaussian mixture distribution.

**Bottom line:** relying purely on external error rates can be misleading on imbalanced
datasets. Accounting for internal cluster cohesion, proportional distribution, and variance
minimization, **K-Means and Ward's Method** stand out as the most reliable clustering
techniques for analyzing customer churn behavior.
