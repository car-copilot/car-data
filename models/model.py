from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_model(input_dim):
    """
    Defines a neural network architecture for ecological driving score prediction.

    Args:
        input_dim (int): The number of features in the input layer.

    Returns:
        tensorflow.keras.models.Model: The compiled neural network model.
    """
    model = Sequential()

    # Input layer with 9 neurons (assuming 9 features after preprocessing)
    model.add(Dense(units=input_dim, activation="relu", input_shape=(input_dim,)))

    # Hidden layer with 16 neurons, ReLU activation, and Dropout for regularization
    model.add(Dense(units=16, activation="relu"))
    model.add(Dropout(rate=0.2))  # Experiment with different dropout rates

    # Hidden layer with 8 neurons, ReLU activation, and Dropout for regularization
    model.add(Dense(units=8, activation="relu"))
    model.add(Dropout(rate=0.2))  # Experiment with different dropout rates

    # Output layer with a single neuron and sigmoid activation for score between 0 and 1
    model.add(Dense(units=1, activation="sigmoid"))

    # Compile the model with optimizer, loss function, and metrics
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    return model