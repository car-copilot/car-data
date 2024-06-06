from get_data.utils import *
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def influxdb_data_to_list(influxdb_data):
    """
    Converts the data retrieved from InfluxDB to a list format.

    Args:
        influxdb_data (list): A list of data points retrieved from InfluxDB.

    Returns:
        list: A list of data points in a list format.
    """
    data_list = []
    for table in influxdb_data:
        for record in table.records:
            # Extract relevant data from the dictionary
            measurement_name = record['_measurement']
            value = record['_value']
            timestamp = record['_time']
            # Assuming all tables have the same 'id' (adjust if needed)
            data_list.append([measurement_name, timestamp, value])
    return data_list

def create_clean_dataframe(data_list):
    """
    Creates a Pandas DataFrame from the data retrieved from InfluxDB.

    Args:
        data_list (list): A list of data points retrieved from InfluxDB.

    Returns:
        pandas.DataFrame: A DataFrame containing the data points.
    """
    columns = ['measurement', 'time', 'value']
    df = pd.DataFrame(data_list, columns=columns)

    df = df.pivot_table(
        values='value',
        index='time',
        columns='measurement',
        aggfunc='first'
    )
    df['Altitude difference'] = df['Altitude'].diff()
    df.drop(columns=['Altitude'], inplace=True)
    df = fill_nan(df)
    if 'Gear engaged' in df.columns:
        df['Gear engaged'] = df['Gear engaged'].astype(int)
    df = df.sort_index(axis=1)
    logger.info(f"Data cleaned : {df.columns}")
    return df

def get_influxdb_data_one_trip(config_file, trip_id):
    """
    Retrieves data from InfluxDB and PostgreSQL for a specific trip.

    Args:
        config_file (str): Path to the YAML file containing connection details.
        trip_id (int): The ID of the trip for which data is to be retrieved.

    Returns:
        tuple: A tuple containing the data retrieved from InfluxDB and PostgreSQL.
    """
    config = load_config(config_file)
    influxdb_client = connect_to_influxdb(config)
    postgres_conn = connect_postgres(config)
    data_from_postgres = get_data_from_postgres_one_trip(postgres_conn, trip_id)
    influxdb_data = query_influxdb(influxdb_client, data_from_postgres[0])
    data_list = influxdb_data_to_list(influxdb_data)
    df = create_clean_dataframe(data_list)
    return df

def get_influxdb_data_all_trips(config_file):
    """
    Retrieves data from InfluxDB and PostgreSQL for all trips.

    Args:
        config_file (str): Path to the YAML file containing connection details.

    Returns:
        list: A list of data retrieved from InfluxDB and PostgreSQL.
    """
    config = load_config(config_file)
    influxdb_client = connect_to_influxdb(config)
    postgres_conn = connect_postgres(config)
    data_from_postgres = get_data_from_postgres(postgres_conn)
    data_list = []
    for trip in data_from_postgres:
        influxdb_data = query_influxdb(influxdb_client, trip)
        data_list.append(influxdb_data_to_list(influxdb_data))
    flat_data_list = [record for trip in data_list for record in trip]
    df = create_clean_dataframe(flat_data_list)
    return df

