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

def add_env(data):
    """
    Adds environmental features to the data using model prediction
    
    Args:
        data (np.array): The preprocessed data.
        
    Returns:
        np.array: The data with environmental features added.
    """
    # Load the model
    environment_model = load_model("out/environment_weights.keras")

    # Predict the environmental features
    env = environment_model.predict(data)
    
    # Add the environmental features to the data
    data = np.concatenate((data, env), axis=1)
    
    return data

logger = logging.getLogger(__name__) 

def add_score(influxdb_data):
    """
    Adds a driving score to the data for a single trip.

    Args:
        influxdb_data (pd.DataFrame): The data for a single trip from InfluxDB.
    """
    # Preprocess the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(influxdb_data)
    logger.info(f"Data scaled: {influxdb_data.describe()}")
    if 'environment' not in influxdb_data.columns:
        # Add environmental features
        data_with_env = add_env(scaled_data)
        
        # Add the env to data
        influxdb_data["environment"] = data_with_env[:, -1].astype(int)
    
    # Load the model
    score_model = load_model("out/score_weights.keras")
    
    # Predict the driving score
    score = score_model.predict(scaled_data)
    
    # Add the score to the data
    influxdb_data["style"] = score.astype(int)
    logger.info(f"Score added to trip {influxdb_data.describe()}.")
    
    
    # Save the data to InfluxDB
    write_api = client.write_api()
    write_api.write_points(bucket, points=[create_data_point(row) for _, row in df.iterrows()])
    
    logger.info(f"Score added to trip {influxdb_data['trip_id'].iloc[0]}.")