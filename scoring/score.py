from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model # type: ignore
from influxdb_client import InfluxDBClient, Point
import logging
import numpy as np
from get_data.utils import load_config

WINDOW_SIZE = 7

def create_windows(df, window_size):
    """
    This function creates sequences (windows) of data from a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing your data.
        window_size (int): The length of the window (number of time steps).
        feature_columns (list): A list of column names representing features.

    Returns:
        list: A list of NumPy arrays, each representing a sequence (window) of data.
    """
    sequences = []
    for i in range(len(df) - window_size + 1):
        sequence = df.iloc[i:i+window_size]  # Select relevant features
        sequences.append(sequence)
    
    # Pad the end of the sequences with zeros
    for i in range(window_size - 1):
        sequences.append(np.zeros_like(sequences[-1]))    
    return sequences

def data_chunks(data, chunk_size):
    """
    Splits the data into chunks of a specified size.

    Args:
        data (pd.DataFrame): The data to split into chunks.
        chunk_size (int): The size of each chunk.

    Returns:
        list: A list of data chunks.
    """
    for i in range(0, len(data), chunk_size):
        yield data.iloc[i:i + chunk_size]

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
            data_point = Point(row["measurement"]).time(row["time"])
            for key, value in row.items():
                if key not in ['time', 'measurement']:
                    data_point = data_point.field("value", value)

            write_api.write(bucket=bucket_name, org=config['influx']['org'], record=data_point)
    print(f"Data written to InfluxDB.")
    client.close()

def add_env(influxdb_data, scaled_data):
    """
    Adds environmental features to the data using model prediction

    Args:
        influxdb_data (pd.DataFrame): The data for a single trip from InfluxDB.
        scaled_data (np.array): The scaled data for a single trip.

    Returns:
        np.array: The data with environmental features added.
    """
    if 'environment' not in influxdb_data.columns:
        # Load the model
        environment_model = load_model("out/environment_weights.keras")
        
        # Predict the environmental features
        pred = environment_model.predict(scaled_data)

        predicted_environment_labels = np.argmax(pred, axis=1)  # Get index of max probability class

        # Add the env to data
        influxdb_data["environment"] = predicted_environment_labels

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

        # Predict the driving score
        score = score_model.predict(scaled_data)[:, 0]

        # Add the score to data
        influxdb_data["style"] = score.astype(int)
        logger.info(f"Score added to trip {influxdb_data.describe()}.")

logger = logging.getLogger(__name__)

def moving_average_smoothing(data, window_size):
    smoothed_data = np.zeros_like(data)
    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        smoothed_data[i] = np.mean(data[start_idx:end_idx])
    return smoothed_data

def add_predictions(config_file, bucket_name, influxdb_data):
    """
    Adds a driving score to the data for a single trip.

    Args:
        influxdb_data (pd.DataFrame): The data for a single trip from InfluxDB.
    """
    sequences = create_windows(influxdb_data, WINDOW_SIZE)
    X_new = np.array(sequences)
    X_new = np.stack(X_new)
    
    # Preprocess the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X_new.reshape(-1, X_new.shape[2]))
    scaled_data = scaled_data.reshape(-1, WINDOW_SIZE, X_new.shape[2])
    # Add environmental features
    add_env(influxdb_data, scaled_data)

    # Add the driving score
    add_score(influxdb_data, scaled_data)

    smoothing_window_size = 10  # Smoothing window size
    smoothed_data = moving_average_smoothing(influxdb_data['style'], smoothing_window_size)
    smoothed_data2 = moving_average_smoothing(influxdb_data['environment'], smoothing_window_size)
    
    influxdb_data['style'] = smoothed_data
    influxdb_data['environment'] = smoothed_data2
    
    # Round the values
    influxdb_data['style'] = influxdb_data['style'].round()
    influxdb_data['environment'] = influxdb_data['environment'].round()
    
    # Save the data to InfluxDB
    write_in_influxdb(config_file, bucket_name, influxdb_data)
