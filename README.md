# 🛍️ Retail Store Customer Clustering
Customer segmentation analysis using unsupervised machine learning
on the Mall Customers dataset.

## Algorithms Used
- KMeans (optimal k via Silhouette Score)
- Agglomerative Hierarchical Clustering
- DBSCAN (density-based with noise detection)

## Results — 6 Customer Segments
| Cluster | Segment | Avg Income | Avg Spend |
|---|---|---|---|
| 0 | 💎 Premium Shoppers | $86k | 83 |
| 1 | 💼 Conservative Earners | $89k | 16 |
| 2 | 🌱 Frugal Savers | $26k | 19 |
| 3 | 🌱 Frugal Savers | $54k | 50 |
| 4 | 🛍️ Budget Enthusiasts | $26k | 77 |
| 5 | 🌱 Frugal Savers | $58k | 48 |

## Output Charts
![K Selection](outputs/01_k_selection_metrics.png)
![PCA Clusters](outputs/03_pca_2d_all_algorithms.png)
![Radar Profiles](outputs/05_radar_cluster_profiles.png)

## How to Run
pip install scikit-learn matplotlib seaborn pandas numpy scipy
python customer_clustering_upgraded.py

## Dataset
Mall Customers dataset from Kaggle.
