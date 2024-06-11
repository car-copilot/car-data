from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from AI.models import create_model, create_inference_model, create_model_CNN_RNN
import numpy as np

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
    # X = influxdb_data_all_trips.sample(frac=1).reset_index(drop=True)
    X = np.array(influxdb_data_all_trips)
    
    # Split data into training, validation, and test sets (70% train, 15% validation, 15% test)
    X_train, X_test, y_train, y_test = train_test_split(X[:, :-1], X[:, -1], test_size=0.2, random_state=42)
    
    # Preprocess the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train the model
    model = create_model(X_train.shape[1])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose=1)
    
    # Evaluate the model
    loss, mae = model.evaluate(X_test, y_test)
    logger.info(f"Test Loss: {loss}, Test MAE: {mae}")

#     plot_learning_curves(history, title="Modèle MLP (avec une couche cachée, 32 neurones)")
    # Save the model
    model.save("out/weights.keras")
    
    return influxdb_data_all_trips