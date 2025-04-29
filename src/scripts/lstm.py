import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from absorption_spectrum import *

# =============== 1. Standardization Function ===============
def standardize(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / (std + 1e-8)

# =============== 2. Your Windowing Function ===============
def windowing(data):
    window_size = 3528
    threshold = 1
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
                break
        if not found_pulse:
            pulse_windows.append(np.zeros(window_size))
            pulse_positions.append(None)

    return np.array(pulse_windows)

# =============== 3. Model Definition (Peak Prediction Only) ===============
class PeakPredictor(tf.keras.Model):
    def __init__(self, input_length):
        super(PeakPredictor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1)  # Predict 1 value: peak position

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        peak_pred = self.dense3(x)
        peak_pred = tf.squeeze(peak_pred, axis=-1)  # Shape: (batch,)
        return peak_pred

# =============== 4. Load Your Data (Standardized) ===============

# Assuming you already have:
# - test_data (raw THz traces for training)
# - val_data (raw THz traces for validation)

# Step 1: Windowing
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

# Peak positions (target for peak prediction)
val_data = tf.convert_to_tensor(windowing(val_data), dtype=tf.float32)
train_data = tf.convert_to_tensor(windowing(test_data), dtype=tf.float32)

peak_positions_train = tf.convert_to_tensor(np.argmax(train_data, axis=1), dtype=tf.float32)
peak_positions_val = tf.convert_to_tensor(np.argmax(val_data, axis=1), dtype=tf.float32)

# They must be tensors
train_data = tf.convert_to_tensor(train_data, dtype=tf.float32)
val_data = tf.convert_to_tensor(val_data, dtype=tf.float32)
peak_positions_train = tf.convert_to_tensor(peak_positions_train, dtype=tf.float32)
peak_positions_val = tf.convert_to_tensor(peak_positions_val, dtype=tf.float32)

# =============== 5. Setup Model and Optimizer ===============

np.random.seed(42)
tf.random.set_seed(42)

model = PeakPredictor(input_length=3528)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# =============== 6. Training Loop (only Peak Loss) ===============

loss_history = []
val_loss_history = []

epochs = 500
batch_size = 32

for epoch in range(epochs):
    idx = tf.random.shuffle(tf.range(tf.shape(train_data)[0]))
    train_data_shuffled = tf.gather(train_data, idx)
    peak_positions_train_shuffled = tf.gather(peak_positions_train, idx)

    epoch_loss = 0.0

    for i in range(0, train_data.shape[0], batch_size):
        x_batch = train_data_shuffled[i:i+batch_size]
        peak_batch = peak_positions_train_shuffled[i:i+batch_size]

        with tf.GradientTape() as tape:
            peak_pred = model(x_batch)
            loss_peak = tf.reduce_mean(tf.square(peak_pred - peak_batch))  # Regression loss

        gradients = tape.gradient(loss_peak, model.trainable_variables)
        gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]  # Clip gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        epoch_loss += loss_peak.numpy()

    avg_epoch_loss = epoch_loss / (train_data.shape[0] // batch_size)
    loss_history.append(avg_epoch_loss)

    # Validation
    val_peak_pred = model(val_data)
    val_loss_peak = tf.reduce_mean(tf.square(val_peak_pred - peak_positions_val))
    val_loss_history.append(val_loss_peak.numpy())

    print(f"Epoch {epoch+1}/{epochs}, Train Peak Loss: {avg_epoch_loss:.6f}, Val Peak Loss: {val_loss_peak:.6f}")

# =============== 7. Plot Loss Curves ===============

plt.figure(figsize=(10,5))
plt.plot(loss_history, label='Train Peak Loss')
plt.plot(val_loss_history, label='Validation Peak Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Training and Validation Peak Prediction Loss')
plt.grid()
plt.legend()
plt.show()

# =============== 8. Predict on New Data ===============

# Assume you get new incoming THz trace
# Replace this with your real incoming trace

# Step 4: Prediction
predicted_peaks = model(standardize(val_data)).numpy()

# Step 5: Visualization
for idx in range(val_data.shape[0]):
    try:
        plt.figure(figsize=(10,5))
        plt.plot(val_data[idx].numpy(), label='Original Window', linestyle='--')
        #plt.plot(reconstructed_pulses[idx], label='Reconstructed Pulse', alpha=0.7)
        plt.axvline(predicted_peaks[idx], color='red', linestyle='--', label='Predicted Peak')
        plt.legend()
        plt.title('New THz Trace: Pulse Reconstruction and Peak Prediction')
        plt.grid()
        plt.savefig(f"plots/rnn_ista_plot_at_{idx}.png", dpi=300)
        plt.show()
    except Exception as e:
        print(f"Error plotting for distance {i} with value {predicted_peaks[i]}: {e}")
        continue

# Step 6: Save predicted peaks if needed
# np.savetxt('predicted_peaks.csv', predicted_peaks, delimiter=',')
