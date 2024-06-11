from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tensorflow.keras.models import load_model # type: ignore
from influxdb_client import InfluxDBClient, Point
import logging
import numpy as np

def create_data_point(row):
    """
    Creates an InfluxDB data point from a row of data.

    Args:
        row (pd.Series): A row of data.

    Returns:
        (Point): An InfluxDB data point.
    """
    point = Point("trip")
    point.tag("trip_id", row["trip_id"])
    point.field("score", row["score"])
    return point

def find_best_k(X):
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

def add_env_with_K_means(X):
    """
    Performs preprocessing steps on the input data.

    Args:
        X (np.ndarray): The data to be preprocessed.

    Returns:
        np.ndarray: The preprocessed data with added cluster.
    """

    # Find the optimal number of clusters
    best_k = find_best_k(X)
    
    # Perform KMeans clustering and add cluster labels as a new feature
    kmeans = KMeans(n_clusters=best_k)
    kmeans.fit(X)
    X_with_clusters = np.hstack((X, kmeans.labels_.reshape(-1, 1)))
    
    # # Apply PCA to the data with clusters
    # pca.fit(X_with_clusters)
    # X_reduced = pca.transform(X_with_clusters)

    # get number of features
    n_features_X = X.shape[1]
    n_clusters = best_k
    n_pca_components = X_with_clusters.shape[1]
    logger.info(f"Data preprocessed with {n_features_X} features in X, {n_clusters} clusters, and {n_pca_components} PCA components.")
        
    return X_with_clusters

logger = logging.getLogger(__name__) 

def add_score(influxdb_data):
    """
    Adds a driving score to the data for a single trip.

    Args:
        influxdb_data (pd.DataFrame): The data for a single trip from InfluxDB.
    """
    # Preprocess the data
    scaler = StandardScaler()
    preprocessed_data = scaler.fit_transform(influxdb_data)
    
    if 'environment' not in influxdb_data.columns:
        # Add environmental features with KMeans clustering
        preprocessed_data = add_env_with_K_means(preprocessed_data)
        
        # Add the env to data
        influxdb_data["environment"] = preprocessed_data[:, -1].astype(int)
    
    # Load the model
    model = load_model("out/weights.keras")
    
    # Predict the driving score
    score = model.predict(preprocessed_data)
    
    # Add the score to the data
    influxdb_data["style"] = score.astype(int)
    logger.info(f"Score added to trip {influxdb_data.describe()}.")
    
    
    
    # Save the data to InfluxDB
    write_api = client.write_api()
    write_api.write_points(bucket, points=[create_data_point(row) for _, row in df.iterrows()])
    
    logger.info(f"Score added to trip {influxdb_data['trip_id'].iloc[0]}.")