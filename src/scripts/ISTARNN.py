import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from absorption_spectrum import *

# =============== 1. Soft-threshold function ===============
def soft_threshold(x, theta):
    return tf.sign(x) * tf.maximum(tf.abs(x) - theta, 0.0)

# =============== 2. Model Definition ===============
class RNNLikeISTA(tf.keras.Model):
    def __init__(self, thz_pulse_init, lam, num_iterations, signal_length):
        super(RNNLikeISTA, self).__init__()
        self.lam = lam
        self.num_iterations = num_iterations
        self.signal_length = signal_length

        # Trainable THz pulse
        self.thz_pulse = tf.Variable(thz_pulse_init, dtype=tf.float32, trainable=True)

        # Precompute Lipschitz constant
        pulse_energy = tf.reduce_sum(self.thz_pulse ** 2)
        self.L = tf.Variable(pulse_energy, trainable=False, dtype=tf.float32)

        # Simple MLP to predict peak
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)  # Predict 1 value: peak position

    def call(self, y):
        # y shape: (batch, signal_length)
        thz_kernel = tf.reverse(self.thz_pulse, axis=[0])
        thz_kernel = tf.reshape(thz_kernel, (-1, 1, 1))
        y_exp = tf.expand_dims(y, axis=2)

        conv_output = tf.nn.conv1d(y_exp, filters=thz_kernel, stride=1, padding='SAME')
        conv_output = tf.squeeze(conv_output, axis=2)

        theta = self.lam / self.L

        x = tf.zeros_like(y)

        for _ in range(self.num_iterations):
            B = (1.0 / self.L) * conv_output
            c = B + x - (1.0 / self.L) * tf.nn.conv1d(
                tf.expand_dims(x, axis=2),
                filters=thz_kernel,
                stride=1,
                padding='SAME'
            )[:, :, 0]
            x = soft_threshold(c, theta)

        features = self.dense1(x)
        peak_pred = self.dense2(features)
        peak_pred = tf.squeeze(peak_pred, axis=-1)

        return x, peak_pred

# =============== 3. Your windowing function ===============
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

# =============== 4. Load your training and validation data ===============

# Assuming you already generated:
# - train_data
# - val_data
# - peak_positions_train
# - peak_positions_val
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

# =============== 5. Setup Model ===============
np.random.seed(42)
tf.random.set_seed(42)

thz_pulse_init = np.exp(-((np.linspace(-1, 1, 100))**2) * 30)

model = RNNLikeISTA(
    thz_pulse_init=thz_pulse_init,
    lam=1e-3,
    num_iterations=10,
    signal_length=3528
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# =============== 6. Training Loop ===============
loss_history = []
val_loss_history = []

epochs = 100
batch_size = 16

for epoch in range(epochs):
    idx = tf.random.shuffle(tf.range(tf.shape(train_data)[0]))
    train_data_shuffled = tf.gather(train_data, idx)
    peak_positions_train_shuffled = tf.gather(peak_positions_train, idx)

    epoch_loss = 0.0

    for i in range(0, train_data.shape[0], batch_size):
        y_batch = train_data_shuffled[i:i+batch_size]
        peak_batch = peak_positions_train_shuffled[i:i+batch_size]

        with tf.GradientTape() as tape:
            x_pred, peak_pred = model(y_batch)

            loss_recon = tf.reduce_mean(tf.square(x_pred - y_batch))
            loss_peak = tf.reduce_mean(tf.square(peak_pred - peak_batch))
            total_loss = loss_recon + 0.001 * loss_peak

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        epoch_loss += total_loss.numpy()

    avg_epoch_loss = epoch_loss / (train_data.shape[0] // batch_size)
    loss_history.append(avg_epoch_loss)

    val_x_pred, val_peak_pred = model(val_data)
    val_loss_recon = tf.reduce_mean(tf.square(val_x_pred - val_data))
    val_loss_peak = tf.reduce_mean(tf.square(val_peak_pred - peak_positions_val))
    val_total_loss = val_loss_recon + 0.001 * val_loss_peak
    val_loss_history.append(val_total_loss.numpy())

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss:.6f}, Val Loss: {val_total_loss:.6f}")

# =============== 7. Plot Loss Curves ===============
plt.figure(figsize=(10,5))
plt.plot(loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.grid()
plt.legend()
plt.show()

# =============== 8. Predict on New Data ===============

# Example: Assume you get new incoming THz data
# new_trace: shape (N,) --> you can load your new real data here
# Replace this with your real new_trace
#new_trace = np.random.randn(50000)  # Dummy example

# Step 1: Windowing
#new_windows = windowing(new_trace)

# Step 2: Tensor conversion
#new_data_tensor = tf.convert_to_tensor(new_windows, dtype=tf.float32)

# Step 3: Model prediction
reconstructed_pulses, predicted_peaks = model(val_data)

reconstructed_pulses = reconstructed_pulses.numpy()
predicted_peaks = predicted_peaks.numpy()

# Step 4: Visualize first window prediction
idx = 0
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

# Step 5: Save predicted peaks if needed
# np.savetxt('predicted_peaks.csv', predicted_peaks, delimiter=',')

learned_thz_pulse = model.thz_pulse.numpy()

plt.figure(figsize=(8,4))
plt.plot(learned_thz_pulse, label='Learned THz Pulse')
plt.title('Optimized THz Pulse after Training')
plt.grid()
plt.legend()
plt.show()

