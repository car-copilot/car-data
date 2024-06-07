from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

import logging

logger = logging.getLogger(__name__)
    
class Preprocessor:
    """
    A class for preprocessing data for ecological driving score prediction.
    """

    def __init__(self):
        """
        Initializes the preprocessor with hyperparameters.

        Args:
            n_clusters (int, optional): Number of clusters for KMeans clustering. Defaults to 4.
            pca_components (float, optional): Percentage of variance to retain with PCA. Defaults to 0.8.
        """
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        
    def find_best_k(self, X):
        """
        Finds the optimal number of clusters for KMeans using silhouette score within 2 to 8 clusters.

        Args:
            X (np.ndarray): The data to be clustered.

        Returns:
            int: The optimal number of clusters.
        """
        best_k = None
        best_score = -1

        for k in range(2, 8 + 1):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            silhouette = silhouette_score(X, kmeans.labels_)

            if silhouette > best_score:
                best_k = k
                best_score = silhouette

        return best_k

    def fit_transform(self, X):
        """
        Performs preprocessing steps on the input data.

        Args:
            X (np.ndarray): The data to be preprocessed.

        Returns:
            np.ndarray: The preprocessed data with added cluster and PCA components.
        """
        
        # Standardize data
        X_scaled = self.scaler.fit_transform(X)

        # Find the optimal number of clusters
        best_k = self.find_best_k(X_scaled)
        
        # Perform KMeans clustering and add cluster labels as a new feature
        self.kmeans = KMeans(n_clusters=best_k)
        self.kmeans.fit(X_scaled)
        X_with_clusters = np.hstack((X_scaled, self.kmeans.labels_.reshape(-1, 1)))
        
        # # Apply PCA to the data with clusters
        # self.pca.fit(X_with_clusters)
        # X_reduced = self.pca.transform(X_with_clusters)

        # get number of features
        n_features = X.shape[1]
        n_clusters = best_k
        n_pca_components = X_with_clusters.shape[1]
        logger.info(f"Data preprocessed with {n_features} features, {n_clusters} clusters, and {n_pca_components} PCA components.")
        return X_with_clusters