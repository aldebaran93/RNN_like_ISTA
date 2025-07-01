import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os
import seaborn as sns
from scipy.signal import find_peaks
from absorption_spectrum import *
from dataset import *

# =============== 0. Fixes & Initialisierung ===============

# =============== 1. Funktionen ===============

def soft_threshold(x, theta):
    return tf.sign(x) * tf.maximum(tf.abs(x) - theta, 0.0)

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
        
peak_positions_val = tf.convert_to_tensor(np.argmax(val_dataset, axis=1), dtype=tf.float32)

# =============== 4. Modell Setup ===============
np.random.seed(42)
tf.random.set_seed(42)

thz_pulse_init = get_trace_slice(t_vector, trace, 98e-12, 105e-12)
thz_pulse_init = np.real(thz_pulse_init).astype(np.float32)

model = RNNLikeISTA(
    thz_pulse_init=thz_pulse_init + awgn(thz_pulse_init, snr_db=0),
    lam=1e-3,
    num_iterations=10,
    signal_length=3528
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# =============== 5. Training ===============
loss_history, val_loss_history = [], []
epochs, batch_size = 50, 16

for epoch in range(epochs):
    idx = tf.random.shuffle(tf.range(tf.shape(train_dataset)[0]))
    train_data_shuffled = tf.gather(train_dataset, idx)
    peak_train_shuffled = tf.gather(peak_positions_train, idx)
    epoch_loss = 0.0

    for i in range(0, train_dataset.shape[0], batch_size):
        y_batch = train_data_shuffled[i:i+batch_size]
        peak_batch = peak_train_shuffled[i:i+batch_size]
        with tf.GradientTape() as tape:
            x_pred, peak_pred = model(y_batch)
            loss_peak = tf.reduce_mean(tf.square(peak_pred - peak_batch))
            total_loss = 0.001 * loss_peak
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss += total_loss.numpy()

    avg_loss = epoch_loss / (train_dataset.shape[0] // batch_size)
    loss_history.append(avg_loss)

    val_x_pred, val_peak_pred = model(val_dataset)
    val_loss_recon = tf.reduce_mean(tf.square(val_x_pred - val_dataset))
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
reconstructed_pulses, predicted_peaks = model(val_dataset)
reconstructed_pulses = reconstructed_pulses.numpy()
predicted_peaks = predicted_peaks.numpy()

pred_peak = get_peaks(reconstructed_pulses)

os.makedirs("plots/RNN_like_ISTA", exist_ok=True)

for row_idx in range(val_dataset.shape[0]):
    plt.figure(figsize=(10,5))
    signal = val_dataset[row_idx]
    peak_indices = pred_peak[row_idx]
    plt.figure()
    plt.plot(signal, label="Signal")
    for idx in peak_indices:
        if idx != 0:
            plt.axvline(x=idx, color='r', linestyle='--', label=f"Peak at {idx}")
    plt.grid()
    plt.title(f'Prediction {row_idx}')
    plt.savefig(f"plots/RNN_like_ISTA/rnn_ista_plot_at_{row_idx}.png", dpi=300)
    plt.show()
    
# =============== 8. THz-Puls anzeigen ===============
learned_thz_pulse = model.thz_pulse.numpy()
plt.figure(figsize=(8,4))
plt.plot(learned_thz_pulse, label='Learned THz Pulse')
plt.title('Learned Pulse')
plt.grid()
plt.legend()
plt.show()

peak_positions_val = get_peaks(val_dataset)
peak_positions_val = tf.convert_to_tensor(peak_positions_val, dtype=tf.float32)
mae = tf.reduce_mean(tf.abs(pred_peak - peak_positions_val.numpy()))
print("===============================================")
print(f"Peak Prediction MAE: {mae.numpy():.2f} samples")
# =============== 9. Metrik: Trefferquote innerhalb ±5 Samples ===============
tolerance = 100
errors = np.abs(pred_peak - peak_positions_val.numpy())
correct_within_tolerance = np.sum(errors <= tolerance)
total = np.count_nonzero(peak_positions_val) #len(errors)
accuracy_within_tolerance = correct_within_tolerance / total
print(f"Peaks innerhalb ±{tolerance} Samples korrekt: {correct_within_tolerance} von {total}")
print(f"Trefferquote innerhalb ±{tolerance} Samples: {accuracy_within_tolerance * 100:.2f}%")
print("===============================================")

y_pred, y_true, metrics = metric(pred_peak, peak_positions_val)
confusion_matrice(y_pred, y_true)