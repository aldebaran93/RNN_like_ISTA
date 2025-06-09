import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from absorption_spectrum import *

# =============== 0. Fixes & Initialisierung ===============
# THz-Trace
transfer_functions = []
distances = np.arange(0.3, 0.329 + 0.001, 500e-6)  # 30 cm to 33 cm, step 0.5 mm
for distance in distances:
    tfct = np.exp(distance * 1j * complex_refractive_index * w_vector / speed_of_light) * \
           np.exp(distance * 1j * 1.0027 * w_vector / speed_of_light)
    tfct = np.abs(tfct) * np.exp(-1j * np.angle(tfct))
    transfer_functions.append(tfct)
transfer_functions = np.array(transfer_functions)

# =============== 1. Funktionen ===============
def awgn(signals, snr_db):
    signals = np.asarray(signals, dtype=np.float32)
    noisy_signals = np.empty_like(signals)
    for i in range(signals.shape[0]):
        signal = signals[i]
        signal_power = np.mean(signal ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
        noisy_signals[i] = signal + noise
    return noisy_signals

def get_trace_slice(t_vector, trace, t_min, t_max):
    trace = dgmm(t_vector, 100e-12, -670.81e-12, -670.39e-12, 0.19, 0.24, -5.43, 3.82)
    trace = trace.astype(np.complex128)
    mask = (t_vector >= t_min) & (t_vector <= t_max)
    return trace[mask]

def soft_threshold(x, theta):
    return tf.sign(x) * tf.maximum(tf.abs(x) - theta, 0.0)

def windowing(data):
    window_size = 3528
    threshold = 1
    pulse_windows, pulse_positions = [], []

    for seq in data:
        for i in range(2000, len(seq) - window_size + 1, window_size):
            window = seq[i:i + window_size]
            if np.max(window) > threshold:
                pulse_windows.append(window)
                pulse_positions.append(i)
                break
        else:
            pulse_windows.append(np.zeros(window_size))
            pulse_positions.append(None)
    return np.array(pulse_windows)

# =============== 2. Modell ===============
class RNNLikeISTA(tf.keras.Model):
    def __init__(self, thz_pulse_init, lam, num_iterations, signal_length):
        super().__init__()
        self.lam = lam
        self.num_iterations = num_iterations
        self.signal_length = signal_length

        self.thz_pulse = tf.Variable(thz_pulse_init, dtype=tf.float32, trainable=True)
        pulse_energy = tf.reduce_sum(self.thz_pulse ** 2)
        self.L = tf.Variable(pulse_energy, trainable=False)

        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, y):
        thz_kernel = tf.reverse(self.thz_pulse, axis=[0])
        thz_kernel = tf.reshape(thz_kernel, (-1, 1, 1))
        y_exp = tf.expand_dims(y, axis=2)
        conv_output = tf.nn.conv1d(y_exp, thz_kernel, stride=1, padding='SAME')
        conv_output = tf.squeeze(conv_output, axis=2)

        theta = self.lam / self.L
        x = tf.zeros_like(y)

        for _ in range(self.num_iterations):
            B = (1.0 / self.L) * conv_output
            c = B + x - (1.0 / self.L) * tf.nn.conv1d(tf.expand_dims(x, 2), thz_kernel, stride=1, padding='SAME')[:, :, 0]
            x = soft_threshold(c, theta)

        features = self.dense1(x)
        peak_pred = self.dense2(features)
        peak_pred = tf.squeeze(peak_pred, axis=-1) * self.signal_length
        return x, peak_pred

# =============== 3. Trainings- und Validierungsdaten ===============
test_data = np.real(np.fft.irfft(np.fft.rfft(trace) * transfer_functions))
test_data = awgn(test_data, snr_db=10)

# Validierung
transfer_functions_val = []
distances_2 = np.arange(0.3, 0.329 + 0.001, 500e-6)
for distance in distances_2:
    tfct_val = np.exp(distance * 1j * complex_refractive_index * w_vector / speed_of_light) * \
               np.exp(distance * 1j * 1.0027 * w_vector / speed_of_light)
    tfct_val = np.abs(tfct_val) * np.exp(-1j * np.angle(tfct_val))
    transfer_functions_val.append(tfct_val)
val_data = np.real(np.fft.irfft(np.fft.rfft(trace) * transfer_functions_val))
val_data_noisy = awgn(val_data, snr_db=5)

train_data = tf.convert_to_tensor(windowing(test_data), dtype=tf.float32)
# Get shape
num_samples = tf.shape(train_data)[0]
amplitudes = tf.random.uniform(shape=(num_samples, 1), minval=1, maxval=11, dtype=tf.int32)
amplitudes = tf.cast(amplitudes, tf.float32)
train_data = train_data * amplitudes

val_data = tf.convert_to_tensor(windowing(val_data_noisy), dtype=tf.float32)

peak_positions_train = tf.convert_to_tensor(np.argmax(train_data, axis=1), dtype=tf.float32)
peak_positions_val = tf.convert_to_tensor(np.argmax(val_data, axis=1), dtype=tf.float32)

# =============== 4. Modell Setup ===============
np.random.seed(42)
tf.random.set_seed(42)

thz_pulse_init = get_trace_slice(t_vector, trace, 98e-12, 105e-12)
thz_pulse_init = np.real(thz_pulse_init).astype(np.float32)

model = RNNLikeISTA(
    thz_pulse_init=thz_pulse_init + awgn(thz_pulse_init, snr_db=5),
    lam=1e-3,
    num_iterations=10,
    signal_length=3528
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# =============== 5. Training ===============
loss_history, val_loss_history = [], []
epochs, batch_size = 50, 32

for epoch in range(epochs):
    idx = tf.random.shuffle(tf.range(tf.shape(train_data)[0]))
    train_data_shuffled = tf.gather(train_data, idx)
    peak_train_shuffled = tf.gather(peak_positions_train, idx)
    epoch_loss = 0.0

    for i in range(0, train_data.shape[0], batch_size):
        y_batch = train_data_shuffled[i:i+batch_size]
        peak_batch = peak_train_shuffled[i:i+batch_size]
        with tf.GradientTape() as tape:
            x_pred, peak_pred = model(y_batch)
            loss_peak = tf.reduce_mean(tf.square(peak_pred - peak_batch))
            total_loss = 0.001 * loss_peak
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss += total_loss.numpy()

    avg_loss = epoch_loss / (train_data.shape[0] // batch_size)
    loss_history.append(avg_loss)

    val_x_pred, val_peak_pred = model(val_data)
    val_loss_recon = tf.reduce_mean(tf.square(val_x_pred - val_data))
    val_loss_peak = tf.reduce_mean(tf.square(val_peak_pred - peak_positions_val))
    val_total_loss = val_loss_recon + 0.001 * val_loss_peak
    val_loss_history.append(val_total_loss.numpy())

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.6f}, Val Loss: {val_total_loss:.6f}")

# =============== 6. Visualisierung ===============
plt.figure(figsize=(10,5))
plt.plot(loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid()
plt.show()

# =============== 7. Vorhersage und Visualisierung ===============
reconstructed_pulses, predicted_peaks = model(val_data)
reconstructed_pulses = reconstructed_pulses.numpy()
predicted_peaks = predicted_peaks.numpy()

os.makedirs("plots", exist_ok=True)

for idx in range(val_data.shape[0]):
    plt.figure(figsize=(10,5))
    plt.plot(val_data[idx].numpy(), label='Original Window', linestyle='--')
    plt.axvline(predicted_peaks[idx], color='red', linestyle='--', label='Predicted Peak')
    plt.legend()
    plt.grid()
    plt.title(f'Prediction {idx}')
    plt.savefig(f"plots/rnn_ista_plot_at_{idx}.png", dpi=300)
    plt.show()

# =============== 8. THz-Puls anzeigen ===============
learned_thz_pulse = model.thz_pulse.numpy()
plt.figure(figsize=(8,4))
plt.plot(learned_thz_pulse, label='Learned THz Pulse')
plt.title('Learned Pulse')
plt.grid()
plt.legend()
plt.show()

mae = tf.reduce_mean(tf.abs(predicted_peaks - peak_positions_val.numpy()))
print("===============================================")
print(f"Peak Prediction MAE: {mae.numpy():.2f} samples")
# =============== 9. Metrik: Trefferquote innerhalb ±5 Samples ===============
tolerance = 50
errors = np.abs(predicted_peaks - peak_positions_val.numpy())
correct_within_tolerance = np.sum(errors <= tolerance)
total = len(errors)
accuracy_within_tolerance = correct_within_tolerance / total
print(f"Peaks innerhalb ±{tolerance} Samples korrekt: {correct_within_tolerance} von {total}")
print(f"Trefferquote innerhalb ±{tolerance} Samples: {accuracy_within_tolerance * 100:.2f}%")
print("===============================================")