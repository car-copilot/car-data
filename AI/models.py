from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input # type: ignore
import tensorflow as tf

def create_model_for_score(input_dim):
    """
    Defines a neural network architecture for ecological driving score prediction.

    Args:
        input_dim (int): The number of features in the input layer.

    Returns:
        tensorflow.keras.models.Model: The compiled neural network model.
    """
    model = Sequential()

    # Input layer with number of neurons equal to input_dim
    model.add(Dense(units=input_dim, activation="relu", input_shape=(input_dim,)))
    
    # Hidden layer with 32 neurons, ReLU activation, and Dropout for regularization
    model.add(Dense(units=32, activation="relu"))
    model.add(Dropout(rate=0.2))  # Regularization with dropout
    
    # Hidden layer with 16 neurons, ReLU activation, and Dropout for regularization
    # model.add(Dense(units=16, activation="relu"))
    # model.add(Dropout(rate=0.2))  # Regularization with dropout
       
    # Output layer with a single neuron and linear activation for regression
    model.add(Dense(units=1, activation="linear"))

    # Compile the model with optimizer, loss function, and metrics
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    return model

def create_model_for_environment(input_dim):
    """
    Defines a neural network architecture for environment prediction.

    Args:
        input_dim (int): The number of features in the input layer.

    Returns:
        tensorflow.keras.models.Model: The compiled neural network model.
    """
    model = Sequential()

    # Input layer with number of neurons equal to input_dim
    model.add(Dense(units=input_dim, activation="relu", input_shape=(input_dim,)))
    
    # Hidden layer with 32 neurons, ReLU activation, and Dropout for regularization
    model.add(Dense(units=32, activation="relu"))
    model.add(Dropout(rate=0.2))  # Regularization with dropout
    
    # Output layer with a single neuron and softmax activation for classification
    model.add(Dense(units=4, activation="softmax"))

    # Compile the model with optimizer, loss function, and metrics
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
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
    model.add(Dense(units=Input(shape=(input_dim,)), activation="relu", input_shape=(input_dim,)))
    model.add(Dropout(rate=0.2))  # Regularization with dropout
    
    # Hidden layer with 32 neurons, ReLU activation, and Dropout for regularization
    model.add(Dense(units=64, activation="relu"))
    model.add(Dropout(rate=0.2))  # Regularization with dropout
    
    # Hidden layer with 16 neurons, ReLU activation, and Dropout for regularization
    model.add(Dense(units=16, activation="relu"))
    
    # Output layer with a single neuron and sigmoid activation for binary classification
    model.add(Dense(units=1, activation="sigmoid"))
    
    # Compile the model with optimizer, loss function, and metrics
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    return model

def create_model_CNN_RNN(input_shape):
    """
    Defines a neural network architecture for time series prediction using CNN and RNN layers.

    Args:
        input_shape (tuple): The shape of the input data.

    Returns:
        tensorflow.keras.models.Model: The compiled neural network model.
    """
    model_cnn_rnn = Sequential()
    
    # Convolutional layer with 32 filters, kernel size of 3, ReLU activation, and input shape
    model_cnn_rnn.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(input_shape, 1)))
    
    # Convolutional layer with 64 filters, kernel size of 3, and ReLU activation
    model_cnn_rnn.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    
    # Max pooling layer with pool size of 2
    model_cnn_rnn.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    
    model_cnn_rnn.add(tf.keras.layers.Flatten())
    
    # # Dense layer with 128 neurons and ReLU activation
    # model_cnn_rnn.add(tf.keras.layers.Dense(128, activation='relu'))
    
    # GRU layer with 32 neurons and recurrent dropout for regularization
    # model_cnn_rnn.add(tf.keras.layers.GRU(32, recurrent_dropout=0.2))
    
    # Dropout layer for regularization
    # model_cnn_rnn.add(tf.keras.layers.Dropout(0.2))
    
    # Output layer with a single neuron and linear activation for regression
    model_cnn_rnn.add(tf.keras.layers.Dense(1, activation='linear'))
    
    return model_cnn_rnn