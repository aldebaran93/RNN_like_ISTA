import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers
import matplotlib.pyplot as plt
import datetime
from scipy.signal import savgol_filter
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization 
from tensorflow.keras.callbacks import EarlyStopping

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

# Load the data from the provided file
time, amplitude = load_thz_data('C:/Users/leots/OneDrive/Desktop/master EIT/masterarbeit/RNN_Like_ISTA/Spektrum_THz.txt')
test_data = np.concatenate([amplitude,amplitude,amplitude])
#x_test_data = np.arange(len(test_data))

# Add AWGN noise to the amplitude data
snr_db = 10  # Desired SNR in dB
noisy_test_data = add_awgn_noise(test_data, snr_db)

# Normalize the noisy amplitude data
amplitude_noisy = (add_awgn_noise(amplitude, snr_db) - np.mean(add_awgn_noise(amplitude, snr_db))) / np.std(add_awgn_noise(amplitude, snr_db))
# Normalize the noisy test_data
test_data = (noisy_test_data - np.mean(noisy_test_data)) / np.std(noisy_test_data)

# Define peak detection function (dummy example for labels)
def find_peaks(data, threshold=3):
    peaks = np.where(data > threshold)[0]
    return peaks

# Create labels (peak positions)
peak_positions = find_peaks(test_data)
labels = np.zeros(test_data.shape)
labels[peak_positions] = 1  # Set peak positions as 1, rest as 0
labels_test = np.zeros(test_data.shape)

# Prepare input and output data
x_data = test_data.reshape(-1, 1)  # Reshape to match input shape (samples, features)
y_data = amplitude.reshape(-1, 1)  # Original clean signal to be recovered

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

# Define the RNN-like ISTA model with Dropout
class ISTA_RNN(tf.keras.Model):
    def __init__(self, hidden_size, dropout_rate=0.3):
        super(ISTA_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.We = layers.Dense(hidden_size, activation=None, kernel_regularizer=regularizers.l2(0.01))  # Linear transformation
        self.S = layers.Dense(hidden_size, activation=None, kernel_regularizer=regularizers.l2(0.01))   # Recurrent transformation
        self.dropout = layers.Dropout(dropout_rate)  # Dropout layer for regularization
        self.output_layer = layers.Dense(1, activation='sigmoid')  # Output layer predicting the recovered signal
        self.soft_threshold = layers.ReLU()  # ReLU as the soft-thresholding

    def call(self, inputs, hidden_state=None, training=False):
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
hidden_size = 16  # Number of hidden units
batch_size = 8
num_epochs = 10
learning_rate = 1e-4  # Increased for faster convergence
dropout_rate = 0.2  # Dropout rate for regularization
lambda_value = 0.1  # L1 regularization weight (sparsity term)

# Instantiate the model
ista_rnn_model = ISTA_RNN(hidden_size, dropout_rate)

# Compile the model
ista_rnn_model.compile(optimizer=optimizers.Adam(learning_rate), loss=CustomLoss(lambda_value))
# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)

# Train the model on the noisy data
history = ista_rnn_model.fit(labels, test_data, batch_size=batch_size, epochs=num_epochs, validation_data=(test_data, labels_test), callbacks=[early_stopping])

# Print the history
print(history.history.keys())

# Evaluate the model
test_loss = ista_rnn_model.evaluate(amplitude_noisy, y_data)
print(f"Test Loss: {test_loss}")

# Predict the recovered signal
predictions, _ = ista_rnn_model.predict(y_data)

# Noise Reduction on Recovered Signal using Savitzky-Golay Filter (smooth out the signal)
smoothed_predictions = savgol_filter(predictions.flatten(), window_length=11, polyorder=3)

# Define Timestamps
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Plot of the original THz trace
plt.figure(figsize=(10, 6))
plt.plot(amplitude, label='Original received THz Signal')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('original received THz Signal')
plt.grid(True)
plt.show()

# Plot the original noisy THz signal and the recovered signal
plt.plot(y_data, label='Noisy THz Signal', alpha=0.7)
plt.plot(smoothed_predictions, 'r', label='Recovered Signal (smoothed)', alpha=0.7)
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('THz Signal Recovery with Noise Reduction')
plt.grid(True)
plt.savefig(f'plots/recovered_signal_{timestamp}_{batch_size}_{hidden_size}_{learning_rate}_{num_epochs}.png', bbox_inches='tight', dpi=300)
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
plt.savefig(f'plots/loss_{timestamp}_{batch_size}_{hidden_size}_{learning_rate}_{num_epochs}.png', bbox_inches='tight', dpi=300)
plt.show()

# Summarize training history for accuracy
# plt.figure(figsize=(10, 6))
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
# plt.savefig(f'plots/plot_accuracy_{timestamp}.png', bbox_inches='tight', dpi=300)
# plt.show()

