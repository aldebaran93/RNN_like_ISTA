import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from absorption_spectrum import *


def standardize(data):
    return (data - np.mean(data)) / np.std(data)

class SoftThresholdLayer(layers.Layer):
    def __init__(self, threshold):
        super(SoftThresholdLayer, self).__init__()
        self.threshold = tf.constant(threshold, dtype=tf.float32)

    def call(self, inputs):
        return tf.sign(inputs) * tf.maximum(tf.abs(inputs) - self.threshold, 0.0)
    
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

# ISTA-RNN Model Definition
class ISTA_RNN(tf.keras.Model):
    def __init__(self, input_dim, hidden_state=None, threshold=0.1):
        super(ISTA_RNN, self).__init__()
        self.We = layers.Dense(input_dim, use_bias=False)
        self.hidden = layers.Dense(input_dim, activation='relu')
        self.dropout = layers.Dropout(0.5)
        self.reconstruction = layers.Dense(input_dim)
        self.peak_output = layers.Dense(1, activation='relu')
        self.soft_threshold = SoftThresholdLayer(threshold)  # the soft-thresholding

    def call(self, inputs, hidden_state=None, training=False):
        batch_size = inputs.shape[1]
        if hidden_state is None:
            hidden_state = tf.zeros((batch_size, input_dim))
            
        x = self.We(inputs)
        hidden_state = self.hidden(hidden_state) + x
        hidden_state = self.soft_threshold(hidden_state)
        
        # dropout während trainieren
        if training:
            hidden_state = self.dropout(hidden_state, training=training)
        #x = self.dropout(x, training=training)
        recon = self.reconstruction(x)
        peak = self.peak_output(x)
        return {'reconstruction': recon, 'peak': peak}

def windowing(data):
    window_size = 3528  # Fenstergröße
    threshold = 1       # Amplituden-Schwellenwert
    pulse_windows = []
    pulse_positions = []

    for seq_idx, sequence in enumerate(data):
        found_pulse = False
        for i in range(2000, len(sequence) - window_size + 1, window_size):
            window = sequence[i:i + window_size]
            if np.max(window) > threshold:
                pulse_windows.append(window)
                pulse_positions.append(i)
                found_pulse = True
                break  # Sobald Puls gefunden ist, stoppe für diese Sequenz
        if not found_pulse:
            # Falls kein Puls gefunden wurde, z.B. Nullfenster anhängen
            pulse_windows.append(np.zeros(window_size))
            pulse_positions.append(None)

    return np.array(pulse_windows)
    
test_data = np.real(np.fft.irfft(np.fft.rfft(trace) * transfer_functions))
labels = tf.convert_to_tensor(np.argmax(test_data, axis=1), dtype=tf.float32)

### validation data####
transfer_functions_val = []
distances_2 = np.arange(0.3, 0.329 + 0.001, 500e-6) # 30 cm to 60 cm, step 1 mm
for distance in distances_2:
    # Compute transfer function for the current distance
    transfer_function_val = np.exp(distance * 1j * complex_refractive_index * w_vector / speed_of_light) * \
                        np.exp(distance * 1j * 1.0027 * w_vector / speed_of_light)
    transfer_function_val = np.abs(transfer_function_val) * np.exp(-1j * np.angle(transfer_function_val))
    transfer_functions_val.append(transfer_function_val)

val_data = np.real(np.fft.irfft(np.fft.rfft(trace) * transfer_functions_val))

#Normalisierung der Daten
#test_data = standardize(test_data)
#val_data = standardize(val_data)

val_data = tf.convert_to_tensor(windowing(val_data), dtype=tf.float32)
train_data = tf.convert_to_tensor(windowing(test_data), dtype=tf.float32)

# Peak positions (target for peak prediction)
peak_positions_train = tf.convert_to_tensor(np.argmax(train_data, axis=1), dtype=tf.float32)
peak_positions_val = tf.convert_to_tensor(np.argmax(val_data, axis=1), dtype=tf.float32)

# Model Compile and Train
input_dim = tf.shape(train_data)[1]
ista_rnn_model = ISTA_RNN(input_dim)
ista_rnn_model.compile(
    optimizer='adam',
    loss=CustomLoss(0.01),  #{'reconstruction': 'mse', 'peak': 'mse'},
    loss_weights={'reconstruction': 1.0, 'peak': 0.01}
)

# Define early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

history = ista_rnn_model.fit(
    train_data, 
    {'reconstruction': train_data, 'peak': peak_positions_train},
    validation_data=(val_data, {'reconstruction': val_data, 'peak': peak_positions_val}),
    batch_size=32,
    epochs=500,
    callbacks=[early_stopping, lr_scheduler]
)

# Prediction and Visualization
predictions = ista_rnn_model.predict(val_data)
reconstructed_traces = predictions['reconstruction']
predicted_peaks = predictions['peak'].flatten()

# Mittelwert
mean = np.mean(peak_positions_val - predicted_peaks)

# Evaluate the model
test_loss = ista_rnn_model.evaluate(val_data, peak_positions_val)
print(f"Test Loss: {test_loss}")

x_axis = np.arange(train_data.shape[1])
plt.figure(figsize=(12, 8))
for i in range(val_data.shape[0]):
    try:
        plt.plot(x_axis, val_data[i], color='blue', alpha=0.3)
        plt.scatter(predicted_peaks[i], val_data[i, int(predicted_peaks[i])], color='red', marker='x', s=60)
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.title('Vorhersage der Max-Peak-Position auf 300 neuen Traces')
        plt.savefig(f"plots/plot_at_{i}.png", dpi=300)
        plt.show()
    except Exception as e:
        print(f"Error plotting for distance {i} with value {predicted_peaks[i]}: {e}")
        continue

# Summarize training history for loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.savefig(f"plots/loss_vs_val.png", dpi=300)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()