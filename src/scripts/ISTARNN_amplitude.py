import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import os
import seaborn as sns
from absorption_spectrum import *
from dataset import *

# =============== 0. Fixes & Initialisierung ===============

# =============== 1. Funktionen ===============

def get_trace_slice(t_vector, trace, t_min, t_max):
    trace = dgmm(t_vector, 100e-12, -670.81e-12, -670.39e-12, 0.19, 0.24, -5.43, 3.82)
    trace = trace.astype(np.complex128)
    mask = (t_vector >= t_min) & (t_vector <= t_max)
    return trace[mask]

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

def metric(predicted_peaks, peak_positions_val):
    predicted_peaks = np.array(predicted_peaks)
    true_peaks = np.array(peak_positions_val)
    
    y_true = []
    y_pred = []
    
    # Track matched ground-truth peaks
    matched_true = set()
    
    # Evaluate predicted peaks
    for pred in predicted_peaks:
        match_found = False
        for i, true in enumerate(true_peaks):
            if i not in matched_true and abs(pred - true) <= 50:
                # Match found within ±50 samples
                match_found = True
                matched_true.add(i)
                break
        if match_found:
            y_true.append(1)  # There was a real peak
            y_pred.append(1)  # And we detected it
        else:
            y_true.append(0)  # No real peak matched
            y_pred.append(1)  # But we predicted one = False Positive
    
    # Now add False Negatives: ground-truth peaks with no matching prediction
    for i in range(len(true_peaks)):
        if i not in matched_true:
            y_true.append(1)  # Real peak
            y_pred.append(0)  # But missed = False Negative
    
    # Metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    a = print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    return y_pred, y_true, a

def confusion_matrice(y_pred, y_true):

    # Get raw confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Convert to percentages
    cm_percent = cm / np.sum(cm) * 100
    
    # Create annotated labels
    labels = np.array([["{0:.4f}%".format(val) for val in row] for row in cm_percent])
    
    # Plotting
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percent, annot=labels, fmt='', cmap='Blues', cbar=True,
                xticklabels=["Present", "Absent"],
                yticklabels=["Present", "Absent"])
    
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("(a)", loc='center', fontsize=12)
    plt.tight_layout()
    plt.show()
# =============== 3. Trainings- und Validierungsdaten ===============

peak_positions_train = tf.convert_to_tensor(np.argmax(train_dataset, axis=1), dtype=tf.float32)
peak_positions_val = tf.convert_to_tensor(np.argmax(val_dataset, axis=1), dtype=tf.float32)

# =============== 4. Modell Setup ===============
np.random.seed(42)
tf.random.set_seed(42)

thz_pulse_init = get_trace_slice(t_vector, trace, 98e-12, 105e-12)
thz_pulse_init = np.real(thz_pulse_init).astype(np.float32)

model = RNNLikeISTA(
    thz_pulse_init=thz_pulse_init + awgn(thz_pulse_init, snr_db=30),
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

os.makedirs("plots", exist_ok=True)

for idx in range(val_dataset.shape[0]):
    plt.figure(figsize=(10,5))
    plt.plot(val_dataset[idx], label='Original Window', linestyle='--')
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

y_pred, y_true, a = metric(predicted_peaks, peak_positions_val)
confusion_matrice(y_pred, y_true)