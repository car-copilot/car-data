from models.preprocessing import Preprocessor
from models.model import create_model


def train_model(influxdb_data_all_trips):
    """
    Train the model with the data from all trips
    """
    preprocessor = Preprocessor()
    X_preprocessed = preprocessor.fit_transform(influxdb_data_all_trips)
    nb_features = X_preprocessed.shape[1]
    # Train the model
    model = create_model(nb_features)
    model.summary()
    
    return influxdb_data_all_trips