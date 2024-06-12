import yaml
from influxdb_client import InfluxDBClient
import pandas as pd
import psycopg2

def load_config(config_file):
    """
    Loads configuration details from a YAML file.

    Args:
        config_file (str): Path to the YAML file containing InfluxDB connection details.

    Returns:
        dict: A dictionary containing the loaded configuration details.
    """
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def connect_to_influxdb(config):
    """
    Connects to InfluxDB using the provided configuration.

    Args:
        config (dict): A dictionary containing InfluxDB connection details.

    Returns:
        influxdb_client.InfluxDBClient: A client object for interacting with InfluxDB.
    """
    url = config['influx']['url']
    token = config['influx']['token']
    org = config['influx']['org']
    return InfluxDBClient(url=url, token=token, org=org)

def query_influxdb(client, data_from_postgres):
    """
    Executes a query on the InfluxDB database.

    Args:
        client (influxdb_client.InfluxDBClient): A client object for interacting with InfluxDB.
        query (str): The InfluxDB query to be executed.

    Returns:
        list: A list of data points retrieved from the query.
    """
    start_time = int(data_from_postgres[1].timestamp())
    stop_time = int(data_from_postgres[2].timestamp())
    bucket_name = data_from_postgres[3]
    query = f"""
        from(bucket: "{bucket_name}")
        |> range(start: {start_time}, stop: {stop_time})
        |> filter(fn: (r) => r["_measurement"] == "Vehicle speed" or r["_measurement"] == "Instant fuel flow" or r["_measurement"] == "Boost pressure actual" or r["_measurement"] == "Altitude" or r["_measurement"] == "Calculated engine load value" or r["_measurement"] == "Engine RPM" or r["_measurement"] == "Gear engaged" or r["_measurement"] == "Instant engine power" or r["_measurement"] == "Throttle position" or r["_measurement"] == "Vehicle acceleration" or r["_measurement"] == "environment" or r["_measurement"] == "style")
        |> filter(fn: (r) => r["_field"] == "value")
    """
    query_api = client.query_api()
    return query_api.query(query=query, org=client.org)

def connect_postgres(config):
    """
    Connects to a PostgreSQL database using the provided configuration.

    Args:
        config (dict): A dictionary containing PostgreSQL connection details.

    Returns:
        psycopg2.extensions.connection: A connection object for interacting with the PostgreSQL database.
    """
    return psycopg2.connect(
        host=config['postgres']['host'],
        port=config['postgres']['port'],
        database=config['postgres']['db'],
        user=config['postgres']['user'],
        password=config['postgres']['mdp']
    )
  
def get_data_from_postgres_one_trip(conn, trip_id):
    """
    Retrieves data from a PostgreSQL database for a specific trip.

    Args:
        conn (psycopg2.extensions.connection): A connection object for interacting with the PostgreSQL database.
        trip_id (int): The ID of the trip for which data is to be retrieved.
        
    Returns:
        list: A list of data records retrieved from the database.
    """
    cursor = conn.cursor()
    query = """
        SELECT 
            t.trip_id, 
            t.begin_timestamp, 
            t.end_timestamp,
            o.bucket
        FROM 
            public.trip AS t
        JOIN 
            public.owners AS o
        ON 
            t.owner_id = o.owner_id
        WHERE 
            trip_id = %s
    """
    cursor.execute(query, (trip_id,))
    return cursor.fetchall()

import logging

logger = logging.getLogger(__name__)
def get_data_from_postgres(conn):
    """
    Retrieves data from a PostgreSQL database.

    Args:
        conn (psycopg2.extensions.connection): A connection object for interacting with the PostgreSQL database.

    Returns:
        list: A list of data records retrieved from the database.
    """
    cursor = conn.cursor()
    query = """
        SELECT 
            t.trip_id, 
            t.begin_timestamp, 
            t.end_timestamp,
            o.bucket
        FROM 
            public.trip AS t
        JOIN 
            public.owners AS o
        ON 
            t.owner_id = o.owner_id
    """
    cursor.execute(query)
    return cursor.fetchall()

def fill_nan(df):
    """
    Fills missing values in a DataFrame using interpolation and forward/backward filling.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data points.

    Returns:
        pandas.DataFrame: The DataFrame with missing values filled.
    """
    for column in df.columns[df.isna().any()]:
        df[column] = df[column].interpolate(method='time')
        df[column] = df[column].ffill()
        df[column] = df[column].bfill()
    return df