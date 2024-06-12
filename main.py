import os
import pandas as pd
from get_data.get_data_from_influx import get_influxdb_data_one_trip, get_influxdb_data_all_trips
from AI.train import train_model
from scoring.score import add_predictions
from fastapi import FastAPI
import logging

config_file = "config.yaml"
logger = logging.getLogger(__name__)
api = FastAPI()

@api.get("/add_score_in_trip/{trip_id}")
async def add_score_in_trip(trip_id: int):
    influxdb_data, bucket_name = get_influxdb_data_one_trip(config_file, trip_id)
    
    add_predictions(config_file, bucket_name, influxdb_data)
    return "OK"

@api.get("/retrain_model")
async def retrain_model():
    influxdb_data_all_trips = get_influxdb_data_all_trips(config_file)
    train_model(influxdb_data_all_trips)
    return "OK"