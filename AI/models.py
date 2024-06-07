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

    # Hidden layer with 32 neurons, ReLU activation, and Dropout for regularization
    model.add(Dense(units=32, activation="relu"))
    model.add(Dropout(rate=0.2))  # Experiment with different dropout rates

    # Hidden layer with 16 neurons, ReLU activation, and Dropout for regularization
    model.add(Dense(units=16, activation="relu"))
    model.add(Dropout(rate=0.2))  # Experiment with different dropout rates

    # Output layer with a single neuron and sigmoid activation for score between 0 and 1
    model.add(Dense(units=1, activation="sigmoid"))

    # Compile the model with optimizer, loss function, and metrics
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    return model

def create_inference_model(input_dim=11):  # Model for inference with original features
    """
    Defines a neural network architecture for inference on original data.

    Args:
        input_dim (int, optional): The number of features in the original data. Defaults to 11.

    Returns:
        tensorflow.keras.models.Model: The compiled model for inference.
    """
    model = Sequential()

    # Input layer with number of neurons equal to input_dim (original features)
    model.add(Dense(units=input_dim, activation="relu", input_shape=(input_dim,)))
    model.add(Dropout(rate=0.2))  # Regularization with dropout
    
    # Hidden layer with 32 neurons, ReLU activation, and Dropout for regularization
    model.add(Dense(units=32, activation="relu"))
    model.add(Dropout(rate=0.2))  # Regularization with dropout
    
    # Hidden layer with 16 neurons, ReLU activation, and Dropout for regularization
    model.add(Dense(units=16, activation="relu"))
    
    # Output layer with a single neuron and sigmoid activation for binary classification
    model.add(Dense(units=1, activation="sigmoid"))
    
    # Compile the model with optimizer, loss function, and metrics
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    return model