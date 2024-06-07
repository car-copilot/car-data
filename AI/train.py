from sklearn.model_selection import train_test_split
from AI.preprocessing import Preprocessor
from AI.models import create_model
import numpy as np

import logging

logger = logging.getLogger(__name__)

def train_model(influxdb_data_all_trips):
    """
    Train the model with the data from all trips
    """
    X = np.array(influxdb_data_all_trips)
    
    # Split data into training, validation, and test sets (70% train, 15% validation, 15% test)
    X_train, X_test_val, y_train, y_test_val = train_test_split(X[:, :-1], X[:, -1], test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)
    
    # Preprocess the data
    preprocessor = Preprocessor()
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.fit_transform(X_val)
    
    logger.info(f"Data preprocessed: {X_train.shape}")
    # Train the model
    model = create_model(X_train.shape[1])
    model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))
    
    # Evaluate the model
    loss, mae, mse = model.evaluate(X_val, y_val)
    logger.info(f"Test Loss: {loss}, Test MSE: {mse}, Test MAE: {mae}")
    
    # Save the model
    model.save("out/weights.h5")
    
    return influxdb_data_all_trips