import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers
import matplotlib.pyplot as plt
import datetime
from scipy.signal import savgol_filter
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.signal import resample
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from absorption_spectrum import *

# Load THz data from .txt file
def load_thz_data(file_path):
    data = np.loadtxt(file_path)
    time = data[:, 0]  # Time data
    amplitude = data[:, 1]  # THz signal amplitude
    return time, amplitude

# AWGN Noise function
def add_awgn_noise(signal, snr_db):
    """
    Adds AWGN noise to the signal with a given SNR (in dB).
    """
    signal_power = np.mean(signal**2)
    snr_linear = 10**(snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.normal(size=signal.shape)
    noisy_signal = signal + noise
    return noisy_signal

# Define peak detection function (dummy example for labels)
def find_peaks(data, threshold=1):
    peaks = np.where(data > threshold)[0]
    return peaks

def standardize(data):
    return (data - np.mean(data)) / np.std(data)

# Load the data from the provided file
# time, amplitude = load_thz_data('C:/Users/leots/OneDrive/Desktop/master EIT/masterarbeit/RNN_Like_ISTA/Spektrum_THz.txt')

# Define custom loss function combining L2 loss and L1 regularization
class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_value):
        super(CustomLoss, self).__init__()
        self.lambda_value = lambda_value

    def call(self, y_true, y_pred):
        # L2 loss (squared error)
        l2_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        # L1 regularization (sparsity penalty)
        l1_loss = self.lambda_value * tf.reduce_sum(tf.abs(y_pred))
        # Combined loss
        return l2_loss + l1_loss

class SoftThresholdLayer(layers.Layer):
    def __init__(self, threshold):
        super(SoftThresholdLayer, self).__init__()
        self.threshold = tf.constant(threshold, dtype=tf.float32)

    def call(self, inputs):
        return tf.sign(inputs) * tf.maximum(tf.abs(inputs) - self.threshold, 0.0)

# Define the RNN-like ISTA model with Dropout
class ISTA_RNN(tf.keras.Model):
    def __init__(self, hidden_size, dropout_rate=0.3, threshold=0.1):
        super(ISTA_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.We = layers.Dense(hidden_size, activation='relu', kernel_regularizer=regularizers.l2(0.01))  # Linear transformation
        self.S = layers.Dense(hidden_size, activation='relu', kernel_regularizer=regularizers.l2(0.01))   # Recurrent transformation
        self.dropout = layers.Dropout(dropout_rate)  # Dropout layer for regularization
        self.output_layer = layers.Dense(1, activation='linear')  # Output layer predicting the recovered signal
        self.soft_threshold = SoftThresholdLayer(threshold)  # the soft-thresholding

    def call(self, inputs, hidden_state=None, training=None):
        batch_size = tf.shape(inputs)[0]  # Dynamic batch size handling
        if hidden_state is None:
            hidden_state = tf.zeros((batch_size, self.hidden_size))
        
        if len(inputs.shape) == 1:
            inputs = tf.expand_dims(inputs, axis=-1)  # Add dimension if needed

        # Linear transformation (ISTA step)
        x = self.We(inputs)
        
        # Recurrent transformation with dropout
        hidden_state = self.S(hidden_state) + x
        hidden_state = self.soft_threshold(hidden_state)

        # Apply dropout during training
        if training:
            hidden_state = self.dropout(hidden_state, training=training)
        
        # Predict the recovered signal
        recovered_signal = self.output_layer(hidden_state)
        return recovered_signal, hidden_state

# Hyperparameters
hidden_size = 64  # Number of hidden units
batch_size = 32
num_epochs = 10000
learning_rate = 1e-4  # Increased for faster convergence
dropout_rate = 0.2  # Dropout rate for regularization
lambda_value = 0.01  # Reduced L1 regularization weight (sparsity term)
distance_losses = []  # To store loss values

# Plot of the original THz trace
plt.figure(figsize=(10, 6))
plt.plot(trace, label='Original received THz Signal')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('original received THz Signal')
plt.grid(True)
plt.show()

test_data = np.real(np.fft.irfft(np.fft.rfft(trace) * transfer_functions))
test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1])
test_data = standardize(test_data)
test_data = tf.convert_to_tensor(test_data, dtype=tf.float32)

transfer_functions_val = []
distances = np.arange(0.3, 0.595 + 0.005, 0.001) # 30 cm to 60 cm, step 1 mm
for distance in distances:
    # Compute transfer function for the current distance
    transfer_function_val = np.exp(distance * 1j * complex_refractive_index * w_vector / speed_of_light) * \
                        np.exp(distance * 1j * 1.0027 * w_vector / speed_of_light)
    transfer_function_val = np.abs(transfer_function_val) * np.exp(-1j * np.angle(transfer_function_val))
    transfer_functions_val.append(transfer_function_val)

val_data = np.real(np.fft.irfft(np.fft.rfft(trace) * transfer_functions_val))
val_data = val_data.reshape(val_data.shape[0], 1, val_data.shape[1])
val_data = standardize(val_data)

# Labels
max_indices = np.argmax(test_data, axis=2)
labels = tf.convert_to_tensor(max_indices, dtype=tf.float32)

print(f"test_data shape: {test_data.shape}")
print(f"val_data shape: {val_data.shape}")
print(f"labels shape: {labels.shape}")

# Instantiate the model
ista_rnn_model = ISTA_RNN(hidden_size, dropout_rate)

# Compile the model
ista_rnn_model.compile(optimizer=optimizers.Adam(learning_rate), loss=CustomLoss(lambda_value))

# Define early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

# Train the model on the data
max_indices = np.argmax(val_data, axis=2)
labels_2 = tf.convert_to_tensor(max_indices, dtype=tf.float32)
X_val = val_data
Y_val = labels_2
history = ista_rnn_model.fit(test_data, labels, batch_size=batch_size, epochs=num_epochs, validation_data=(X_val, Y_val), callbacks=[early_stopping, lr_scheduler])

# Print the history
print(history.history.keys())

# Evaluate the model
test_loss = ista_rnn_model.evaluate(val_data)
print(f"Test Loss: {test_loss}")

# Predict the recovered signal
val_size = int(test_data.shape[2] * 0.33)
#X_val = test_data[-val_size:]
#Y_val = labels[-val_size:]
predictions, _ = ista_rnn_model.predict(X_val)
#predictions = np.squeeze(predictions)

# Plot the original noisy THz signal and the recovered signal
for i in range(val_data.shape[0]):
    plt.plot(val_data[i, 0, :], label='THz Signal', alpha=0.7)
    plt.scatter(predictions[i,0], 0, color='r', marker='x', s=100, label='Predicted Pulse Position')
    plt.scatter(Y_val[i], 0, color='y', marker='o', s=100, label='True Pulse Position')
    plt.legend()
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title(f'Predicted Pulse position at {round(distances[i] * 1e2, 1)}cm')
    plt.grid(True)
    plt.show()

# Summarize training history for loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Confusion matrix and metrics
predicted_classes = (predictions.flatten() > 0.5).astype(int)  # Threshold to get binary predictions
cm = confusion_matrix(labels, predicted_classes)
precision = precision_score(labels, predicted_classes, zero_division=0)
recall = recall_score(labels, predicted_classes, zero_division=0)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Present", "Absent"])
disp.plot(cmap='Blues', values_format=".4%")
plt.title(f"Precision={precision:.3f}, Recall={recall:.3f}")
plt.show()