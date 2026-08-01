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
|<img width="1205" height="639" alt="01_dendrogram_single" src="https://github.com/user-attachments/assets/00f5c7fd-e9b4-469d-9c1c-dd8f49c97697" />
<img width="1208" height="636" alt="02_dendrogram_complete" src="https://github.com/user-attachments/assets/35f5d1f4-127d-488a-a846-c76e0000ee2e" />


| <img width="1208" height="636" alt="03_dendrogram_ward" src="https://github.com/user-attachments/assets/23cbb554-6ad6-4742-8226-c1f5c9e0a5b9" />
|

### K-Means

A partition-based centroid algorithm configured for **k = 2**, iteratively minimizing the
within-cluster sum of squares (WCSS) using the standard Euclidean distance formula:

<img width="248" height="32" alt="05_euclidean_distance_formula" src="https://github.com/user-attachments/assets/2c1e9438-4e6a-47ce-9728-dbeb2b18fcac" />

<img width="1208" height="643" alt="04_kmeans_weka_output" src="https://github.com/user-attachments/assets/caf3c338-953f-4af7-8089-d694ad6ab74b" />

<img width="1208" height="643" alt="06_kmeans_visualize_scatter" src="https://github.com/user-attachments/assets/192fb697-6f24-4f03-aa2b-86a8d9264633" />

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
| <img width="1207" height="636" alt="07_weka_hierarchical_single" src="https://github.com/user-attachments/assets/61314552-9271-4aa5-a3d3-f34935a769e4" /> |<img width="1209" height="636" alt="08_weka_hierarchical_complete" src="https://github.com/user-attachments/assets/2a48ac45-8093-4418-8069-35bef5ac6e05" /> | <img width="1207" height="641" alt="09_weka_hierarchical_ward" src="https://github.com/user-attachments/assets/ddecac0b-48c8-46ba-80cd-9da7422d7e0e" />|

| EM | DBSCAN | FarthestFirst |
|---|---|---|
|<img width="1208" height="642" alt="10_weka_em_output" src="https://github.com/user-attachments/assets/de365e13-9c62-4bb4-a665-b563b6916d15" />
 | <img width="1208" height="638" alt="11_weka_dbscan_output" src="https://github.com/user-attachments/assets/bda6ccbf-6cd5-4117-9ce7-2188c593cef3" />
 | <img width="1209" height="638" alt="12_weka_farthestfirst_output" src="https://github.com/user-attachments/assets/96a1b2bc-ec37-444b-aed0-611190079575" />
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
