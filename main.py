from get_data.get_data_from_influx import get_influxdb_data_one_trip, get_influxdb_data_all_trips
from AI.train import train_model
from fastapi import FastAPI
import logging

config_file = "config.yaml"
logger = logging.getLogger(__name__)
api = FastAPI()

@api.get("/add_score_in_trip/{trip_id}")
async def add_score_in_trip(trip_id: int):
    influxdb_data = get_influxdb_data_one_trip(config_file, trip_id)
    
    add_score_in_trip(influxdb_data)
    return "OK"

@api.get("/retrain_model")
async def retrain_model():
    influxdb_data_all_trips = get_influxdb_data_all_trips(config_file)
    # save to csv
    influxdb_data_all_trips.to_csv("out/influxdb_data_all_trips.csv", index=False)
    train_model(influxdb_data_all_trips)
    return "OK"