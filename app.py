import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="Customer Segmentation Toolkit", layout="wide")

st.title("🛍️ Customer Segmentation Dashboard")
st.markdown("Upload your dataset to explore K-Means and DBSCAN clustering.")

# 1. File Upload
uploaded_file = st.sidebar.file_uploader("Upload 'mc.csv'", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # 2. Data Selection Logic
    try:
        X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    except KeyError:
        st.warning("Standard columns not found. Using columns at index 3 and 4.")
        X = df.iloc[:, [3, 4]]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Sidebar Tabs for Algorithm selection
    algo = st.sidebar.selectbox("Select Algorithm", ["K-Means", "DBSCAN"])

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Dataset Preview")
        st.write(df.head())
        
        if algo == "K-Means":
            st.divider()
            st.subheader("K-Means Parameters")
            k_value = st.slider("Select Number of Clusters (k)", 2, 10, 5)
            
            # Run KMeans
            kmeans = KMeans(n_clusters=k_value, init='k-means++', random_state=42)
            y_kmeans = kmeans.fit_predict(X_scaled)
            df['Cluster'] = y_kmeans

        else:
            st.divider()
            st.subheader("DBSCAN Parameters")
            eps = st.slider("Epsilon (eps)", 0.1, 1.0, 0.3)
            min_samples = st.slider("Min Samples", 1, 10, 5)
            
            # Run DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X_scaled)
            df['Cluster'] = clusters

    with col2:
        st.subheader(f"{algo} Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plotting
        scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df['Cluster'], 
                            cmap='viridis', s=100, edgecolor='k')
        
        ax.set_xlabel("Annual Income")
        ax.set_ylabel("Spending Score")
        plt.colorbar(scatter, ax=ax, label='Cluster ID')
        
        st.pyplot(fig)

        # Cluster Distribution
        st.subheader("Cluster Counts")
        st.bar_chart(df['Cluster'].value_counts())

else:
    st.info("Please upload a CSV file to get started.")
