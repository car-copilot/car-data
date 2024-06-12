from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from AI.models import create_model_for_score, create_model_for_environment, create_inference_model, create_model_CNN_RNN
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
        
        
def train_model(influxdb_data_all_trips):
    """
    Train the model with the data from all trips
    """    
    X = np.array(influxdb_data_all_trips)
    features = X[:, :-2]
    y_environment = X[:, -2]
    y_driving_score = X[:, -1]
    
    logger.info(f"X shape: {X.shape}")
    logger.info(f"{influxdb_data_all_trips.describe()}")

    X_train, X_test, y_environment_train, y_environment_test, y_driving_score_train, y_driving_score_test = train_test_split(features, y_environment, y_driving_score, test_size=0.2, random_state=42)

    # Preprocess the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    y_environment_train = to_categorical(y_environment_train, num_classes=4)
    y_environment_test = to_categorical(y_environment_test, num_classes=4)
    
    # Train the environment model
    environment_model = create_model_for_environment(features.shape[1])
    environment_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    environment_model.fit(X_train, y_environment_train, epochs=30, batch_size=64, validation_data=(X_test, y_environment_test), verbose=1)  
    
    # Evaluate the environment_model
    environment_model.evaluate(X_test, y_environment_test)
    environment_model.summary()
    
    # Train the driving score model
    driving_score_model = create_model_for_score(features.shape[1])
    driving_score_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    driving_score_model.fit(X_train, y_driving_score_train, epochs=30, batch_size=64, validation_data=(X_test, y_driving_score_test), verbose=1)
    
    # Evaluate the driving_score_model
    driving_score_model.evaluate(X_test, y_driving_score_test)
    driving_score_model.summary()

#     plot_learning_curves(history, title="Modèle MLP (avec une couche cachée, 32 neurones)")

    # Save the driving_score_model
    driving_score_model.save("out/score_weights.keras")
    
    environment_model.save("out/environment_weights.keras")
    
    return influxdb_data_all_trips