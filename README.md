# Customer Segmentation (K-Means Clustering)

## Overview
This project analyzes a Mall Customer dataset to identify distinct customer segments based on their Annual Income and Spending Score. It uses unsupervised learning techniques to group similar customers together for targeted marketing strategies.

## Dataset
- **Source:** `mc.csv` (Mall Customers)
- **Key Features:**
  - `Annual Income (k$)`
  - `Spending Score (1-100)`

## Methodology
1. **Elbow Method:** Determines the optimal number of clusters ($k$).
2. **K-Means Clustering:** Groups customers into $k=5$ distinct segments.
3. **DBSCAN (Bonus):** Implements density-based clustering to find outliers and non-linear clusters.

## Technologies Used
- Python
- Scikit-Learn (KMeans, DBSCAN)
- Matplotlib & Seaborn (Visualization)
- Pandas

## Visualizations
- **Elbow Graph:** Shows the optimal $k$ value.
- **Cluster Map:** A 2D scatter plot coloring customers by their assigned segment.
