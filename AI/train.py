from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from AI.models import create_model_for_score, create_model_for_environment, create_model_CNN_RNN
import numpy as np
from tensorflow.keras.utils import to_categorical # type: ignore

import logging

logger = logging.getLogger(__name__)

# def plot_learning_curves(history, title=""):
#         loss = history.history['loss']
#         val_loss = history.history['val_loss']

#         epochs = range(len(loss))

#         plt.figure(figsize=(15,5))
#         plt.plot(epochs, loss, 'bo', label='Entraînement')
#         plt.plot(epochs, val_loss, 'b', label='Validation')
#         plt.xlabel("epochs")
#         plt.ylabel("Perte MAE")
#         plt.title(title)
#         plt.legend()
        
#         # save the plot
#         plt.savefig("out/learning_curves.png")
        
def create_windows(df, window_size, feature_names):
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
        sequence = df[feature_names].iloc[i:i+window_size]  # Select relevant features
        sequences.append(sequence.values)
    
    # Pad the end of the sequences with zeros
    for i in range(window_size - 1):
        sequences.append(np.zeros_like(sequences[-1]))    
    return sequences

def train_model(influxdb_data_all_trips):
    """
    Train the model with the data from all trips
    """    
    X = np.array(influxdb_data_all_trips)
    y_environment = X[:, -2]
    y_driving_score = X[:, -1]
    
    sequences = create_windows(influxdb_data_all_trips, 5, influxdb_data_all_trips.columns[:-2])
    features = np.array(sequences)
    features = np.stack(features)

    X_train, X_test, y_environment_train, y_environment_test, y_driving_score_train, y_driving_score_test = train_test_split(features, y_environment, y_driving_score, test_size=0.2, random_state=42)

    # Preprocess the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[2]))
    X_test = scaler.fit_transform(X_test.reshape(-1, X_test.shape[2]))
    X_train = X_train.reshape(-1, 5, X_train.shape[1])
    X_test = X_test.reshape(-1, 5, X_test.shape[1])
    y_environment_train = to_categorical(y_environment_train, num_classes=4)
    y_environment_test = to_categorical(y_environment_test, num_classes=4)
    
    # Train the environment model
    environment_model = create_model_for_environment(features.shape[2], 5)
    environment_model.summary()
    environment_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    environment_model.fit(X_train, y_environment_train, epochs=20, batch_size=64, validation_data=(X_test, y_environment_test), verbose=1)  
    
    # Evaluate the environment_model
    environment_model.evaluate(X_test, y_environment_test)
    
    
    # Train the driving score model
    driving_score_model = create_model_for_score(features.shape[2], 5)
    driving_score_model.summary()
    driving_score_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    driving_score_model.fit(X_train, y_driving_score_train, epochs=20, batch_size=64, validation_data=(X_test, y_driving_score_test), verbose=1)
    
    # Evaluate the driving_score_model
    driving_score_model.evaluate(X_test, y_driving_score_test)

#     plot_learning_curves(history, title="Modèle MLP (avec une couche cachée, 32 neurones)")    
    
    # Save the driving_score_model
    driving_score_model.save("out/score_weights.keras")
    
    environment_model.save("out/environment_weights.keras")
    
    return influxdb_data_all_trips