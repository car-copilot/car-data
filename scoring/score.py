import pprint
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from tensorflow.keras.models import load_model # type: ignore
from influxdb_client import InfluxDBClient, Point
import logging
import numpy as np

from get_data.get_data_from_influx import get_influxdb_data_one_trip
from get_data.utils import load_config

def write_in_influxdb(config_file, bucket_name, data):
    """
    Writes data to InfluxDB.

    Args:
        config_file (str): The path to the configuration file.
        bucket_name (str): The name of the bucket.
        data (pd.DataFrame): The data to write to InfluxDB.
    """
    data = data.unstack().reset_index()
    
    config = load_config(config_file)
    client = InfluxDBClient(url=config['influx']['url'], token=config['influx']['token'], org=config['influx']['org'])
    write_api = client.write_api()
    for i, row in data.iterrows():
        if row["measurement"] in ["style", "environment"]:
            # show progress
            if i % 100 == 0:
                print(f"Writing row {i} of {len(data)}")
            data_point = Point(row["measurement"]).time(row["time"])
            for key, value in row.items():
                if key not in ['time', 'measurement']:
                    data_point = data_point.field("value", value)

            write_api.write(bucket=bucket_name, org=config['influx']['org'], record=data_point)

def add_env(influxdb_data, scaled_data):
    """
    Adds environmental features to the data using model prediction

    Args:
        influxdb_data (pd.DataFrame): The data for a single trip from InfluxDB.
        scaled_data (np.array): The scaled data for a single trip.

    Returns:
        np.array: The data with environmental features added.
    """
    logger.info(f"Data columns before if: {influxdb_data.columns}")
    if 'environment' not in influxdb_data.columns:
        # Load the model
        environment_model = load_model("out/environment_weights.keras")
        
        # Check model input shape
        if environment_model.input_shape[1] != scaled_data.shape[1] and 'Gear engaged' in influxdb_data.columns:
            influxdb_data_copy = influxdb_data.copy()
            influxdb_data_copy.drop(columns=['Gear engaged'], inplace=True)
            scaled_data = StandardScaler().fit_transform(influxdb_data_copy)
        
        logger.info(f"Data shape: {scaled_data.shape}")   
        # Predict the environmental features
        pred = environment_model.predict(scaled_data)

        predicted_environment_labels = np.argmax(pred, axis=1)  # Get index of max probability class

        # Add the env to data
        influxdb_data["environment"] = predicted_environment_labels
        
    return influxdb_data

def add_score(influxdb_data, scaled_data):
    """
    Adds a driving score to the data for a single trip.

    Args:
        influxdb_data (pd.DataFrame): The data for a single trip from InfluxDB.
        scaled_data (np.array): The scaled data for a single trip.

    Returns:
        np.array: The data with the driving score added.
    """
    if 'style' not in influxdb_data.columns:
        # Load the model
        score_model = load_model("out/score_weights.keras")
        
        # Check model input shape
        if score_model.input_shape[1] != scaled_data.shape[1] and 'Gear engaged' in influxdb_data.columns:
            influxdb_data_copy = influxdb_data.copy()
            influxdb_data_copy.drop(columns=['environment'], inplace=True)
            influxdb_data_copy.drop(columns=['Gear engaged'], inplace=True)
            scaled_data = StandardScaler().fit_transform(influxdb_data_copy)

        # Predict the driving score
        score = score_model.predict(scaled_data)

        # Add the score to data
        influxdb_data["style"] = score.astype(int)
        logger.info(f"Score added to trip {influxdb_data.describe()}.")

logger = logging.getLogger(__name__)

def add_predictions(config_file, bucket_name, influxdb_data):
    """
    Adds a driving score to the data for a single trip.

    Args:
        influxdb_data (pd.DataFrame): The data for a single trip from InfluxDB.
    """
    # Preprocess the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(influxdb_data)
    # Add environmental features
    add_env(influxdb_data, scaled_data)

    # Add the driving score
    add_score(influxdb_data, scaled_data)

    # Save the data to InfluxDB
    write_in_influxdb(config_file, bucket_name, influxdb_data)

    # logger.info(f"Score added to trip {influxdb_data['trip_id'].iloc[0]}.")
    # logger.info(f"Data with score: {get_influxdb_data_one_trip(config_file, influxdb_data['trip_id'].iloc[0])}")
